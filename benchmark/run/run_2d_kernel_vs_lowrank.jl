"""
Compare `HighOrderOp_Kernel` and `HighOrderLowRankOp` on the same 2D dataset.

The benchmark measures a realistic one-shot reconstruction:

    operator setup + solver creation + fixed-iteration CG reconstruction

It deliberately excludes MAT-file I/O and data preprocessing from the timed
region. Each repeat constructs a fresh operator and releases the cached
LowRank normal backend, so timings do not depend on a prior CG backend.

Run from the repository root, for example:

    julia --project=. --threads=4 benchmark/run/run_2d_kernel_vs_lowrank.jl

Configuration can be overridden with environment variables, for example:

    HIGHORDER_BENCHMARK_GPUS=4,5 HIGHORDER_BENCHMARK_RANK=10 \
    HIGHORDER_BENCHMARK_REPEATS=5 julia --project=. --threads=4 \
    benchmark/run/run_2d_kernel_vs_lowrank.jl

For peak memory, record `nvidia-smi` externally while this script runs;
CUDA.jl's memory pool intentionally retains freed allocations for reuse.
"""
# using Revise

using HighOrderMRI
using AbstractNFFTs
using CUDA
using Dates
using LinearAlgebra
using MAT
using Printf
using RegularizedLeastSquares
using Statistics

include(joinpath(@__DIR__, "..", "common", "benchmark_utils.jl"))


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

const T = Float32
const DATA_FILE = get(
    ENV,
    "HIGHORDER_BENCHMARK_DATA",
    # "/home/jyzhang/Desktop/HighOrderLowRankOp/2D/7T_2D_Spiral_1p0_200_r4.mat",
    "/home/jyzhang/Desktop/Julia_pkg/HighOrderMRI-benchmark/demo/7T_2D_EPI_1p0_200_r4.mat",
)
const OUTPUT_DIR = normpath(joinpath(@__DIR__, "..", "results"))

# Use the identical GPU set for both operators. Exclude GPUs occupied by other
# jobs; a competing process makes both timing and peak-memory results invalid.
const GPU_IDS = parse_env_gpus("HIGHORDER_BENCHMARK_GPUS", [4, 5, 6, 7])
const N_REPEATS = parse_env_int("HIGHORDER_BENCHMARK_REPEATS", 3)
const WARMUP = parse_env_bool("HIGHORDER_BENCHMARK_WARMUP", true)

N_REPEATS >= 1 || throw(ArgumentError("N_REPEATS must be at least one to compute reconstruction-quality metrics"))

const RECON_TERMS = "0111"
const KERNEL_NBLOCK = Int64(50)
const LOWRANK_RANK = parse_env_int("HIGHORDER_BENCHMARK_RANK", 10)
const RSVD_SEED = parse_env_int("HIGHORDER_BENCHMARK_RSVD_SEED", 1234)
const RSVD_CHUNK = parse_env_int("HIGHORDER_BENCHMARK_RSVD_CHUNK", 4096)
const RSVD_OVERSAMPLE = parse_env_int("HIGHORDER_BENCHMARK_RSVD_OVERSAMPLE", 5)
const SHARED_RANK_MAX = parse_env_int("HIGHORDER_BENCHMARK_SHARED_RANK_MAX", 32)
const SHARED_BASIS_TOL = T(parse(Float64, get(ENV, "HIGHORDER_BENCHMARK_SHARED_BASIS_TOL", "1e-2")))
const CG_ITERATIONS = parse_env_int("HIGHORDER_BENCHMARK_CG_ITERATIONS", 20)
const REGULARIZATION = T(parse(Float64, get(ENV, "HIGHORDER_BENCHMARK_REGULARIZATION", "1e-9")))
const NFFT_CENTER_CORRECTION = parse_env_bool("HIGHORDER_BENCHMARK_NFFT_CENTER_CORRECTION", true)
const SAVE_IMAGES = parse_env_bool("HIGHORDER_BENCHMARK_SAVE_IMAGES", true)
const IMAGE_VMAX_PERCENTILE = parse(
    Float64,
    get(ENV, "HIGHORDER_BENCHMARK_IMAGE_VMAX_PERCENTILE", "99.9"),
)
const DIFFERENCE_DISPLAY_SCALE = parse(
    Float64,
    get(ENV, "HIGHORDER_BENCHMARK_DIFFERENCE_SCALE", "10"),
)
DIFFERENCE_DISPLAY_SCALE > 0 || throw(ArgumentError("HIGHORDER_BENCHMARK_DIFFERENCE_SCALE must be positive"))


"""Rotate a reconstructed 2D image into the display orientation used by the reconstruction scripts."""
function display_reconstruction(x)
    image = Array(x)
    if ndims(image) == 3 && size(image, 3) == 1
        image = dropdims(image; dims=3)
    end
    ndims(image) == 2 || throw(ArgumentError(
        "Expected a 2D reconstruction or a singleton-z array; got size $(size(image))",
    ))
    return rotl90(image)
end


"""Save a magnitude reconstruction using an optional fixed display scale."""
function save_reconstruction_png(
    path::AbstractString,
    x;
    vmax::Union{Nothing,Real}=nothing,
    vmaxp::Real=99.9,
)
    image = abs.(display_reconstruction(x))

    fig = if isnothing(vmax)
        plt_image(image; vmaxp=vmaxp, width=10)
    else
        plt_image(image; vmin=0, vmax=vmax, width=10)
    end
    fig.savefig(path; dpi=300, transparent=false, bbox_inches="tight", pad_inches=0.0)
    return path
end


"""Save `scale * abs(abs(x) - abs(reference))` using the reference image scale."""
function save_magnitude_difference_png(
    path::AbstractString,
    x,
    reference;
    scale::Real=DIFFERENCE_DISPLAY_SCALE,
    reference_vmax::Real,
)
    difference = scale .* abs.(
        abs.(display_reconstruction(x)) .- abs.(display_reconstruction(reference)),
    )
    fig = plt_image(difference; vmin=0, vmax=reference_vmax, width=10)
    fig.savefig(path; dpi=300, transparent=false, bbox_inches="tight", pad_inches=0.0)
    return path
end


function summarize(method::String, runs, shared_rank, complex_error, mag_error,
                   kernel_residual, kernel_residual_aligned, self_residual)
    setup = getproperty.(runs, :setup_s)
    recon = getproperty.(runs, :recon_s)
    total = getproperty.(runs, :total_s)
    return (
        method=method,
        repeats=length(runs),
        setup_median_s=median(setup),
        setup_min_s=minimum(setup),
        setup_mean_s=mean(setup),
        setup_std_s=std(setup),
        recon_median_s=median(recon),
        recon_min_s=minimum(recon),
        recon_mean_s=mean(recon),
        recon_std_s=std(recon),
        total_median_s=median(total),
        total_min_s=minimum(total),
        total_mean_s=mean(total),
        total_std_s=std(total),
        shared_rank=shared_rank,
        configured_rank=LOWRANK_RANK,
        cg_iterations=CG_ITERATIONS,
        gpu_ids=join(GPU_IDS, ','),
        nfft_center_correction=NFFT_CENTER_CORRECTION,
        nfft_backend=string(AbstractNFFTs.active_backend()),
        julia_version=string(VERSION),
        complex_error_vs_kernel=complex_error,
        magnitude_nrmse_vs_kernel=mag_error,
        kernel_model_residual=kernel_residual,
        kernel_model_residual_aligned=kernel_residual_aligned,
        self_model_residual=self_residual,
    )
end


# -----------------------------------------------------------------------------
# Dataset preparation: identical to Recon_HighOrderLowRankOp.jl
# -----------------------------------------------------------------------------

isfile(DATA_FILE) || error(
    "Benchmark data file was not found: $DATA_FILE\n" *
    "Set HIGHORDER_BENCHMARK_DATA to the 2D MAT file path.",
)
isempty(GPU_IDS) && error("GPU_IDS must contain at least one CUDA device ID")

@info "Loading 2D benchmark dataset" DATA_FILE GPU_IDS N_REPEATS LOWRANK_RANK CG_ITERATIONS NFFT_CENTER_CORRECTION
raw = matread(DATA_FILE)

csm = raw["gre_csm"]
b0 = raw["gre_b0"]
mask = raw["gre_mask"]
kdata = raw["kdata"]
datatime = vec(raw["datatime"])
matrix_size = Int.(raw["matrixSize"])
fov = raw["FOV"]
k0_ecc = raw["k0_adc"]
dt_measured = raw["dt_Measured"]
kspha_measured = raw["ksphaMeasured"]
start_measured = raw["startMeasured"]
tau_measured = raw["tauMeasured"]
dt_adc = raw["dt_adc"]

kspha_measured = InterpTrajTime(
    kspha_measured,
    dt_measured,
    start_measured + tau_measured * dt_adc,
    datatime,
)

nX, nY, nZ = matrix_size
dx, dy, dz = T.(fov ./ matrix_size)
weight = samplingDensity(kspha_measured'[2:3, :], (nX, nY))
kdata = kdata ./ exp.(2π * im .* k0_ecc)'
kdata = kdata .* exp.(-2π * im .* kspha_measured[:, 1])

# Keep the coordinate and image-array transformations identical to the source
# reconstruction script so both operators see exactly the same physical model.
shift_x, shift_y = 0, -1
grid = Grid(nX, nY, nZ, dx, dy, dz; exchange_xy=false, reverse_x=false, reverse_y=false)
csm_model = Complex{T}.(permutedims(reverse(circshift(csm, (shift_x, shift_y, 0)), dims=(1)), [2, 1, 3],))
fieldmap_model = T.(permutedims(reverse(circshift(b0, (shift_x, shift_y)), dims=(1)), [2, 1]))
mask_model = Bool.(permutedims(reverse(circshift(mask, (shift_x, shift_y)), dims=(1)), [2, 1]))
kspha_model = T.(kspha_measured')  # [nTerm, nSam]

rec_params = Dict{Symbol,Any}(
    :reconSize => (nX, nY),
    :reg => L2Regularization(REGULARIZATION),
    :iterations => CG_ITERATIONS,
    :solver => CGNR,
)


# `recon_HOOp` expects its CUDA inputs on the LowRank operator's primary GPU.
# Select it before creating any CuArray inputs; otherwise CUDA.jl uses the
# process-default device (commonly GPU 0), which may not have P2P access to the
# configured primary GPU.
CUDA.device!(first(GPU_IDS))
data = CuArray(Complex{T}.(kdata))
weight = CuArray(Complex{T}.(weight))

@info "Dataset ready" matrix_size nSam=size(data, 1) nCha=size(data, 2) nVox=sum(mask_model)

# -----------------------------------------------------------------------------
# Operator constructors
# -----------------------------------------------------------------------------

function build_kernel_operator()
    return HighOrderOp_Kernel(
        grid,
        copy(kspha_model),
        T.(datatime);
        fieldmap=fieldmap_model,
        csm=csm_model,
        mask=mask_model,
        recon_terms=RECON_TERMS,
        nBlock=KERNEL_NBLOCK,
        arrayType=CuArray,
        gpus=GPU_IDS,
        verbose=false,
    )
end


function build_lowrank_operator()
    distribution = length(GPU_IDS) > 1 ? :voxel : :single
    normal_distribution = length(GPU_IDS) > 1 ? :channel : :single

    return HighOrderLowRankOp(
        grid,
        copy(kspha_model),
        T.(datatime);
        fieldmap=fieldmap_model,
        csm=csm_model,
        mask=mask_model,
        recon_terms=RECON_TERMS,
        arrayType=CuArray,
        gpus=GPU_IDS,
        L_rank=LOWRANK_RANK,
        rsvd_seed=RSVD_SEED,
        rsvd_chunk=RSVD_CHUNK,
        rsvd_oversample=RSVD_OVERSAMPLE,
        rsvd_finalize=:gram,
        rsvd_backend=:kernel,
        rsvd_distribution=distribution,
        shared_rank_max=SHARED_RANK_MAX,
        shared_basis_tol=SHARED_BASIS_TOL,
        normal_distribution=normal_distribution,
        nfft_center_correction=NFFT_CENTER_CORRECTION,
        verbose=false,
    )
end


# -----------------------------------------------------------------------------
# Warmup and timed experiments
# -----------------------------------------------------------------------------

function warmup!()
    @info "Warmup: compiling Kernel reference path"
    kernel_op = build_kernel_operator()
    try
        recon_HOOp(kernel_op, data, weight, rec_params)
    finally
        kernel_op = nothing
        GC.gc(true)
    end

    @info "Warmup: compiling LowRank path"
    lowrank_op = build_lowrank_operator()
    try
        recon_HOOp(lowrank_op, data, weight, rec_params)
    finally
        release_benchmark_backend!(lowrank_op)
        lowrank_op = nothing
        GC.gc(true)
    end
    return nothing
end


function benchmark_kernel(repeat_id::Int)
    GC.gc(true)
    free_before = gpu_free_memory_mib(GPU_IDS)
    op, setup_s = elapsed_seconds(build_kernel_operator, GPU_IDS)

    try
        # HighOrderOp_Kernel initializes all listed devices and leaves the
        # calling task on the last one. 
        CUDA.device!(first(GPU_IDS))
        x, recon_s = elapsed_seconds(
            () -> recon_HOOp(op, data, weight, rec_params),
            GPU_IDS,
        )
        x = require_reconstruction(x, "HighOrderOp_Kernel reconstruction")
        residual = weighted_residual(op, x, data, weight)
        run = (
            method="HighOrderOp_Kernel",
            repeat=repeat_id,
            setup_s=setup_s,
            recon_s=recon_s,
            total_s=setup_s + recon_s,
            shared_rank=missing,
            self_residual=residual,
            free_before_mib=free_before,
            free_after_mib=gpu_free_memory_mib(GPU_IDS),
        )
        return run, Array(x)
    finally
        op = nothing
        GC.gc(true)
    end
end


function benchmark_lowrank(repeat_id::Int)
    GC.gc(true)
    free_before = gpu_free_memory_mib(GPU_IDS)
    op, setup_s = elapsed_seconds(build_lowrank_operator, GPU_IDS)

    try
        x, recon_s = elapsed_seconds(
            () -> recon_HOOp(op, data, weight, rec_params),
            GPU_IDS,
        )
        x = require_reconstruction(x, "HighOrderLowRankOp reconstruction")
        residual = weighted_residual(op, x, data, weight)
        run = (
            method="HighOrderLowRankOp",
            repeat=repeat_id,
            setup_s=setup_s,
            recon_s=recon_s,
            total_s=setup_s + recon_s,
            shared_rank=size(op.basis, 2),
            self_residual=residual,
            free_before_mib=free_before,
            free_after_mib=gpu_free_memory_mib(GPU_IDS),
        )
        return run, Array(x)
    finally
        release_benchmark_backend!(op)
        op = nothing
        GC.gc(true)
    end
end


"""Run all timed reconstructions and the subsequent Kernel-reference metrics."""
function run_benchmark!()
    WARMUP && warmup!()

    kernel_runs = NamedTuple[]
    lowrank_runs = NamedTuple[]
    x_reference = nothing
    x_lowrank = nothing

    for repeat_id = 1:N_REPEATS
        @info "Benchmarking HighOrderOp_Kernel" repeat_id
        kernel_run, x_kernel = benchmark_kernel(repeat_id)
        push!(kernel_runs, kernel_run)
        x_reference = x_kernel

        @info "Benchmarking HighOrderLowRankOp" repeat_id
        lowrank_run, x_lr = benchmark_lowrank(repeat_id)
        push!(lowrank_runs, lowrank_run)
        x_lowrank = x_lr
    end

    # This additional Kernel operator is excluded from the timed results.
    quality_kernel = build_kernel_operator()
    kernel_reference_residual = weighted_residual(quality_kernel, x_reference, data, weight)
    kernel_model_residual_lr = weighted_residual(quality_kernel, x_lowrank, data, weight)
    scale_lr_to_kernel = alignment_scale(x_lowrank, x_reference)
    kernel_model_residual_lr_aligned = weighted_residual(quality_kernel, scale_lr_to_kernel .* x_lowrank, data, weight)
    quality_kernel = nothing
    GC.gc(true)

    complex_error = aligned_relative_error(x_lowrank, x_reference; scale=scale_lr_to_kernel)
    mag_error = magnitude_nrmse(x_lowrank, x_reference; scale=scale_lr_to_kernel)
    lowrank_self_residual = lowrank_runs[end].self_residual
    summary_rows = [
        summarize("HighOrderOp_Kernel", kernel_runs, missing, 0.0, 0.0,
 kernel_reference_residual, kernel_reference_residual,
                  kernel_runs[end].self_residual),
        summarize("HighOrderLowRankOp", lowrank_runs,
                  lowrank_runs[end].shared_rank, complex_error, mag_error,
                  kernel_model_residual_lr, kernel_model_residual_lr_aligned,
                  lowrank_self_residual),
    ]

    mkpath(OUTPUT_DIR)
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    runs_path = joinpath(OUTPUT_DIR, "2d_kernel_lowrank_runs_$(timestamp).csv")
    summary_path = joinpath(OUTPUT_DIR, "2d_kernel_lowrank_summary_$(timestamp).csv")
    write_csv(runs_path, vcat(kernel_runs, lowrank_runs))
    write_csv(summary_path, summary_rows)

    image_paths = String[]
    if SAVE_IMAGES
        kernel_magnitude = abs.(display_reconstruction(x_reference))
        common_vmax = quantile(vec(kernel_magnitude), IMAGE_VMAX_PERCENTILE / 100)

        data_name = splitext(basename(DATA_FILE))[1]
        image_prefix = joinpath(OUTPUT_DIR, "$(data_name)_$(timestamp)")

        kernel_image_path = "$(image_prefix)_kernel.png"
        lowrank_image_path = "$(image_prefix)_lowrank.png"
        lowrank_aligned_image_path = "$(image_prefix)_lowrank_aligned.png"
        difference_image_path = "$(image_prefix)_magnitude_difference.png"

        save_reconstruction_png(
            kernel_image_path,
            x_reference;
            vmax=common_vmax,
        )
        save_reconstruction_png(
            lowrank_image_path,
            x_lowrank;
            vmax=common_vmax,
        )
        save_reconstruction_png(
            lowrank_aligned_image_path,
            scale_lr_to_kernel .* x_lowrank;
            vmax=common_vmax,
        )
        save_magnitude_difference_png(
            difference_image_path,
            scale_lr_to_kernel .* x_lowrank,
            x_reference;
            reference_vmax=common_vmax,
        )

        append!(
            image_paths, 
            (
                kernel_image_path,
                lowrank_image_path,
                lowrank_aligned_image_path,
                difference_image_path,
            ),
        )
    end

    @info "Benchmark complete" runs_path summary_path image_paths
    println("\n2D Kernel vs LowRank summary")
    for row in summary_rows
        @printf("%s: setup median = %.3f s, recon median = %.3f s, total median = %.3f s\n",
                row.method, row.setup_median_s, row.recon_median_s, row.total_median_s)
    end
    @printf("LowRank complex error vs Kernel: %.3e\n", complex_error)
    @printf("LowRank magnitude NRMSE vs Kernel: %.3e\n", mag_error)
    @printf("LowRank Kernel-model residual (raw / aligned): %.3e / %.3e\n",
            kernel_model_residual_lr, kernel_model_residual_lr_aligned)
    return (runs_path=runs_path, summary_path=summary_path, image_paths=image_paths, summary=summary_rows)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_benchmark!()
end
