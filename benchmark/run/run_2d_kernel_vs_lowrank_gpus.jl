"""
Benchmark `HighOrderOp_Kernel` versus `HighOrderLowRankOp` while sweeping the
number of GPUs on the same 2D dataset.

Default experiment:

  * GPU pool:       4,5,6,7
  * GPU counts:     1,2,3,4 (prefixes of the GPU pool)
  * repeats:        5 per method and GPU count
  * LowRank rank:   25
  * rSVD oversample: 5
  * shared-basis tolerance: 0 (do not compress rank 25 further)

Each timed run constructs a fresh operator and measures:

    setup time + fixed-iteration CG reconstruction time

MAT-file loading, preprocessing, warmup, CSV writing, and quality metrics are
outside the timed regions. The benchmark alternates method order and reverses
the GPU-count order on even repeats to reduce order and thermal bias.

Run from the repository root:

    julia --project=. --threads=4 \
        benchmark/run/run_2d_kernel_vs_lowrank_gpus.jl

Useful overrides:

    HIGHORDER_BENCHMARK_GPUS=4,5,6,7 \
    HIGHORDER_BENCHMARK_GPU_COUNTS=1,2,4 \
    HIGHORDER_BENCHMARK_REPEATS=5 \
    HIGHORDER_BENCHMARK_RANK=25 \
    julia --project=. --threads=4 \
        benchmark/run/run_2d_kernel_vs_lowrank_gpus.jl

`HIGHORDER_BENCHMARK_GPU_COUNTS=n1,n2,...` selects prefixes of
`HIGHORDER_BENCHMARK_GPUS`. For example, pool `4,5,6,7` and count `2` uses
GPUs `[4,5]`. Keeping a common primary GPU ensures the persistent CUDA input
arrays remain valid for every tested configuration.
"""

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

function parse_env_int_list(name::AbstractString, default::AbstractVector{<:Integer})
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return Int.(default)
    values = parse.(Int, strip.(split(raw, ',')))
    isempty(values) && throw(ArgumentError("$name must contain at least one integer"))
    return values
end

const T = Float32
const DATA_FILE = get(
    ENV,
    "HIGHORDER_BENCHMARK_DATA",
    "/home/jyzhang/Desktop/Julia_pkg/HighOrderMRI-benchmark/demo/7T_2D_Spiral_1p0_200_r4.mat",
    # "/home/jyzhang/Desktop/Julia_pkg/HighOrderMRI-benchmark/demo/7T_2D_EPI_1p0_200_r4.mat",
)
const OUTPUT_ROOT = normpath(joinpath(@__DIR__, "..", "results"))

# GPU count n means the first n devices in GPU_POOL. Prefixes keep the primary
# CUDA device identical across all configurations, so `data` and `weight` only
# need to be uploaded once.
const GPU_POOL = parse_env_gpus("HIGHORDER_BENCHMARK_GPUS", [4, 5, 6, 7])
const GPU_COUNTS = sort(unique(parse_env_int_list(
    "HIGHORDER_BENCHMARK_GPU_COUNTS",
    collect(1:length(GPU_POOL)),
)))

const N_REPEATS = parse_env_int("HIGHORDER_BENCHMARK_REPEATS", 5)
const WARMUP = parse_env_bool("HIGHORDER_BENCHMARK_WARMUP", true)
const CONTINUE_ON_ERROR = parse_env_bool("HIGHORDER_BENCHMARK_CONTINUE_ON_ERROR", true)

const RECON_TERMS = "0111"
const KERNEL_NBLOCK = Int64(50)
const LOWRANK_RANK = parse_env_int("HIGHORDER_BENCHMARK_RANK", 25)
const RSVD_SEED = parse_env_int("HIGHORDER_BENCHMARK_RSVD_SEED", 1234)
const RSVD_CHUNK = parse_env_int("HIGHORDER_BENCHMARK_RSVD_CHUNK", 4096)
const RSVD_OVERSAMPLE = parse_env_int("HIGHORDER_BENCHMARK_RSVD_OVERSAMPLE", 5)
const SHARED_RANK_MAX = parse_env_int(
    "HIGHORDER_BENCHMARK_SHARED_RANK_MAX",
    LOWRANK_RANK,
)
const SHARED_BASIS_TOL = T(parse(
    Float64,
    get(ENV, "HIGHORDER_BENCHMARK_SHARED_BASIS_TOL", "0"),
))
const CG_ITERATIONS = parse_env_int("HIGHORDER_BENCHMARK_CG_ITERATIONS", 20)
const REGULARIZATION = T(parse(
    Float64,
    get(ENV, "HIGHORDER_BENCHMARK_REGULARIZATION", "1e-9"),
))
const NFFT_CENTER_CORRECTION = parse_env_bool(
    "HIGHORDER_BENCHMARK_NFFT_CENTER_CORRECTION",
    true,
)

isempty(GPU_POOL) && throw(ArgumentError("GPU_POOL must contain at least one device"))
all(>(0), GPU_COUNTS) || throw(ArgumentError("All GPU counts must be positive"))
maximum(GPU_COUNTS) <= length(GPU_POOL) || throw(ArgumentError(
    "Maximum GPU count $(maximum(GPU_COUNTS)) exceeds GPU pool length $(length(GPU_POOL))",
))
N_REPEATS >= 1 || throw(ArgumentError("N_REPEATS must be at least one"))
LOWRANK_RANK >= 1 || throw(ArgumentError("LOWRANK_RANK must be positive"))
SHARED_RANK_MAX >= LOWRANK_RANK || throw(ArgumentError(
    "SHARED_RANK_MAX=$SHARED_RANK_MAX must be at least LOWRANK_RANK=$LOWRANK_RANK",
))

const BASE_GPU_COUNT = minimum(GPU_COUNTS)
if BASE_GPU_COUNT != 1
    @warn "GPU count 1 is not included; speedups use $BASE_GPU_COUNT GPUs as baseline"
end

function gpu_ids_for_count(gpu_count::Integer)
    1 <= gpu_count <= length(GPU_POOL) || throw(ArgumentError(
        "gpu_count=$gpu_count is outside 1:$(length(GPU_POOL))",
    ))
    return GPU_POOL[1:gpu_count]
end


# -----------------------------------------------------------------------------
# Dataset preparation: identical physical/image convention to the 2D benchmark
# -----------------------------------------------------------------------------

isfile(DATA_FILE) || error(
    "Benchmark data file was not found: $DATA_FILE\n" *
    "Set HIGHORDER_BENCHMARK_DATA to the 2D MAT-file path.",
)

@info "Loading 2D GPU-scaling benchmark dataset" DATA_FILE GPU_POOL GPU_COUNTS N_REPEATS LOWRANK_RANK CG_ITERATIONS NFFT_CENTER_CORRECTION

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
sample_weight = samplingDensity(kspha_measured'[2:3, :], (nX, nY))
kdata = kdata ./ exp.(2π * im .* k0_ecc)'
kdata = kdata .* exp.(-2π * im .* kspha_measured[:, 1])

shift_x, shift_y = 0, -1
grid = Grid(
    nX,
    nY,
    nZ,
    dx,
    dy,
    dz;
    exchange_xy=false,
    reverse_x=false,
    reverse_y=false,
)
csm_model = Complex{T}.(
    permutedims(
        reverse(circshift(csm, (shift_x, shift_y, 0)), dims=(1)),
        [2, 1, 3],
    ),
)
fieldmap_model = T.(
    permutedims(
        reverse(circshift(b0, (shift_x, shift_y)), dims=(1)),
        [2, 1],
    ),
)
mask_model = Bool.(
    permutedims(
        reverse(circshift(mask, (shift_x, shift_y)), dims=(1)),
        [2, 1],
    ),
)
kspha_model = T.(kspha_measured')  # [nTerm, nSam]

rec_params = Dict{Symbol,Any}(
    :reconSize => (nX, nY),
    :reg => L2Regularization(REGULARIZATION),
    :iterations => CG_ITERATIONS,
    :solver => CGNR,
)

# Every GPU set is a prefix of GPU_POOL and therefore has the same primary GPU.
CUDA.device!(first(GPU_POOL))
data = CuArray(Complex{T}.(kdata))
weight = CuArray(Complex{T}.(sample_weight))

@info "Dataset ready" matrix_size nSam=size(data, 1) nCha=size(data, 2) nVox=sum(mask_model) primary_gpu=first(GPU_POOL)


# -----------------------------------------------------------------------------
# Parameterized operator constructors
# -----------------------------------------------------------------------------

function build_kernel_operator(gpu_ids::AbstractVector{<:Integer})
    return HighOrderKernelOp(
        grid,
        copy(kspha_model),
        T.(datatime);
        fieldmap=fieldmap_model,
        csm=csm_model,
        mask=mask_model,
        recon_terms=RECON_TERMS,
        nBlock=KERNEL_NBLOCK,
        arrayType=CuArray,
        gpus=collect(gpu_ids),
        verbose=false,
    )
end

function build_lowrank_operator(gpu_ids::AbstractVector{<:Integer})
    distribution = length(gpu_ids) > 1 ? :voxel : :single
    normal_distribution = length(gpu_ids) > 1 ? :channel : :single

    return HighOrderLowRankOp(
        grid,
        copy(kspha_model),
        T.(datatime);
        fieldmap=fieldmap_model,
        csm=csm_model,
        mask=mask_model,
        recon_terms=RECON_TERMS,
        arrayType=CuArray,
        gpus=collect(gpu_ids),
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

function build_operator(method::Symbol, gpu_ids::AbstractVector{<:Integer})
    method === :kernel && return build_kernel_operator(gpu_ids)
    method === :lowrank && return build_lowrank_operator(gpu_ids)
    throw(ArgumentError("Unknown benchmark method: $method"))
end

method_name(method::Symbol) = method === :kernel ? "HighOrderOp_Kernel" :
                              method === :lowrank ? "HighOrderLowRankOp" :
                              throw(ArgumentError("Unknown benchmark method: $method"))


# -----------------------------------------------------------------------------
# Timing and warmup
# -----------------------------------------------------------------------------

function error_string(err, bt)
    io = IOBuffer()
    showerror(io, CapturedException(err, bt))
    return replace(String(take!(io)), '\n' => " | ")
end

function warmup_configuration!(method::Symbol, gpu_ids::AbstractVector{<:Integer})
    op = nothing
    @info "Warmup" method=method_name(method) gpu_count=length(gpu_ids) gpu_ids=join(gpu_ids, ',')
    try
        CUDA.device!(first(gpu_ids))
        op = build_operator(method, gpu_ids)
        CUDA.device!(first(gpu_ids))
        x = recon_HOOp(op, data, weight, rec_params)
        require_reconstruction(x, "$(method_name(method)) warmup")
        synchronize_gpus!(gpu_ids)
    finally
        release_benchmark_backend!(op)
        op = nothing
        GC.gc(true)
    end
    return nothing
end

function warmup_all!()
    for gpu_count in GPU_COUNTS
        gpu_ids = gpu_ids_for_count(gpu_count)
        warmup_configuration!(:kernel, gpu_ids)
        warmup_configuration!(:lowrank, gpu_ids)
    end
    return nothing
end

function benchmark_once(
    method::Symbol,
    gpu_count::Int,
    repeat_id::Int,
    execution_order::Int,
)
    gpu_ids = gpu_ids_for_count(gpu_count)
    op = nothing
    x_host = nothing

    setup_s = missing
    recon_s = missing
    total_s = missing
    shared_rank = missing
    free_before = missing
    free_with_operator = missing
    free_after_recon = missing
    status = "ok"
    error_message = ""

    try
        GC.gc(true)
        CUDA.device!(first(gpu_ids))
        free_before = gpu_free_memory_mib(gpu_ids)
        CUDA.device!(first(gpu_ids))

        op, setup_s = elapsed_seconds(
            () -> build_operator(method, gpu_ids),
            gpu_ids,
        )

        free_with_operator = gpu_free_memory_mib(gpu_ids)
        CUDA.device!(first(gpu_ids))

        x, recon_s = elapsed_seconds(
            () -> recon_HOOp(op, data, weight, rec_params),
            gpu_ids,
        )
        x = require_reconstruction(x, "$(method_name(method)) reconstruction")
        total_s = setup_s + recon_s
        shared_rank = method === :lowrank ? size(op.basis, 2) : missing
        x_host = Array(x)
        free_after_recon = gpu_free_memory_mib(gpu_ids)
    catch err
        status = "error"
        error_message = error_string(err, catch_backtrace())
        @error "2D GPU-scaling benchmark configuration failed" method=method_name(method) gpu_count gpu_ids repeat_id error_message
        CONTINUE_ON_ERROR || rethrow()
    finally
        CUDA.device!(first(gpu_ids))
        release_benchmark_backend!(op)
        op = nothing
        GC.gc(true)
        CUDA.device!(first(GPU_POOL))
    end

    row = (
        status=status,
        error=error_message,
        method=method_name(method),
        gpu_count=gpu_count,
        gpu_ids=join(gpu_ids, ','),
        repeat=repeat_id,
        execution_order=execution_order,
        setup_s=setup_s,
        recon_s=recon_s,
        total_s=total_s,
        configured_rank=method === :lowrank ? LOWRANK_RANK : missing,
        shared_rank=shared_rank,
        rsvd_seed=method === :lowrank ? RSVD_SEED : missing,
        rsvd_oversample=method === :lowrank ? RSVD_OVERSAMPLE : missing,
        shared_basis_tol=method === :lowrank ? SHARED_BASIS_TOL : missing,
        free_before_mib=free_before,
        free_with_operator_mib=free_with_operator,
        free_after_recon_mib=free_after_recon,
        cg_iterations=CG_ITERATIONS,
        regularization=REGULARIZATION,
        nfft_center_correction=NFFT_CENTER_CORRECTION,
        nfft_backend=string(AbstractNFFTs.active_backend()),
        julia_version=string(VERSION),
    )

    return row, x_host
end


# -----------------------------------------------------------------------------
# Summary and quality metrics
# -----------------------------------------------------------------------------

safe_mean(x) = isempty(x) ? missing : mean(x)
safe_std(x) = length(x) <= 1 ? missing : std(x)
safe_median(x) = isempty(x) ? missing : median(x)
safe_minimum(x) = isempty(x) ? missing : minimum(x)
safe_maximum(x) = isempty(x) ? missing : maximum(x)

function successful_values(rows, field::Symbol)
    return Float64[
        Float64(getproperty(row, field))
        for row in rows
        if row.status == "ok" && !ismissing(getproperty(row, field))
    ]
end

function summarize_group(method::String, gpu_count::Int, rows)
    group = [
        row for row in rows
        if row.method == method && row.gpu_count == gpu_count
    ]
    setup = successful_values(group, :setup_s)
    recon = successful_values(group, :recon_s)
    total = successful_values(group, :total_s)
    shared_rank = successful_values(group, :shared_rank)

    return (
        method=method,
        gpu_count=gpu_count,
        gpu_ids=join(gpu_ids_for_count(gpu_count), ','),
        requested_repeats=length(group),
        successful_repeats=count(row -> row.status == "ok", group),
        failed_repeats=count(row -> row.status != "ok", group),
        setup_mean_s=safe_mean(setup),
        setup_std_s=safe_std(setup),
        setup_median_s=safe_median(setup),
        setup_min_s=safe_minimum(setup),
        setup_max_s=safe_maximum(setup),
        recon_mean_s=safe_mean(recon),
        recon_std_s=safe_std(recon),
        recon_median_s=safe_median(recon),
        recon_min_s=safe_minimum(recon),
        recon_max_s=safe_maximum(recon),
        total_mean_s=safe_mean(total),
        total_std_s=safe_std(total),
        total_median_s=safe_median(total),
        total_min_s=safe_minimum(total),
        total_max_s=safe_maximum(total),
        configured_rank=method == "HighOrderLowRankOp" ? LOWRANK_RANK : missing,
        shared_rank_mean=safe_mean(shared_rank),
        shared_rank_std=safe_std(shared_rank),
        shared_rank_min=safe_minimum(shared_rank),
        shared_rank_max=safe_maximum(shared_rank),
        cg_iterations=CG_ITERATIONS,
    )
end

function find_summary(rows, method::String, gpu_count::Int)
    matches = [
        row for row in rows
        if row.method == method && row.gpu_count == gpu_count
    ]
    length(matches) == 1 || error(
        "Expected one summary row for method=$method, gpu_count=$gpu_count; got $(length(matches))",
    )
    return only(matches)
end

function safe_ratio(numerator, denominator)
    (ismissing(numerator) || ismissing(denominator)) && return missing
    denominator == 0 && return missing
    return numerator / denominator
end

function add_scaling_metrics(summary_rows)
    enriched = NamedTuple[]
    for row in summary_rows
        baseline = find_summary(summary_rows, row.method, BASE_GPU_COUNT)
        gpu_ratio = row.gpu_count / BASE_GPU_COUNT

        setup_speedup = safe_ratio(baseline.setup_median_s, row.setup_median_s)
        recon_speedup = safe_ratio(baseline.recon_median_s, row.recon_median_s)
        total_speedup = safe_ratio(baseline.total_median_s, row.total_median_s)

        push!(enriched, merge(row, (
            baseline_gpu_count=BASE_GPU_COUNT,
            setup_speedup_vs_baseline=setup_speedup,
            recon_speedup_vs_baseline=recon_speedup,
            total_speedup_vs_baseline=total_speedup,
            setup_parallel_efficiency=ismissing(setup_speedup) ? missing : setup_speedup / gpu_ratio,
            recon_parallel_efficiency=ismissing(recon_speedup) ? missing : recon_speedup / gpu_ratio,
            total_parallel_efficiency=ismissing(total_speedup) ? missing : total_speedup / gpu_ratio,
        )))
    end
    return enriched
end

function compute_quality_rows(final_reconstructions, summary_rows)
    kernel_name = "HighOrderOp_Kernel"
    lowrank_name = "HighOrderLowRankOp"
    baseline_kernel = get(
        final_reconstructions,
        (kernel_name, BASE_GPU_COUNT),
        nothing,
    )

    quality_rows = NamedTuple[]
    for gpu_count in GPU_COUNTS
        x_kernel = get(final_reconstructions, (kernel_name, gpu_count), nothing)
        x_lowrank = get(final_reconstructions, (lowrank_name, gpu_count), nothing)

        lr_complex_error = missing
        lr_magnitude_error = missing
        lr_magnitude_ssim = missing
        lr_scale_real = missing
        lr_scale_imag = missing
        kernel_complex_error = missing
        kernel_magnitude_error = missing
        kernel_magnitude_ssim = missing

        if x_kernel !== nothing && x_lowrank !== nothing
            scale = complex_alignment_scale(x_lowrank, x_kernel)
            lr_scale_real = real(scale)
            lr_scale_imag = imag(scale)
            lr_complex_error = raw_complex_nrmse(x_lowrank, x_kernel)
            lr_magnitude_error = magnitude_nrmse(x_lowrank, x_kernel)
            lr_magnitude_ssim = magnitude_ssim(x_lowrank, x_kernel)
        end

        if baseline_kernel !== nothing && x_kernel !== nothing
            scale = complex_alignment_scale(x_kernel, baseline_kernel)
            kernel_complex_error = raw_complex_nrmse(x_kernel, baseline_kernel)
            kernel_magnitude_error = magnitude_nrmse(x_kernel, baseline_kernel)
            kernel_magnitude_ssim = magnitude_ssim(x_kernel, baseline_kernel)
        end

        kernel_summary = find_summary(summary_rows, kernel_name, gpu_count)
        lowrank_summary = find_summary(summary_rows, lowrank_name, gpu_count)

        push!(quality_rows, (
            gpu_count=gpu_count,
            gpu_ids=join(gpu_ids_for_count(gpu_count), ','),
            lowrank_complex_error_vs_same_gpu_kernel=lr_complex_error,
            lowrank_magnitude_nrmse_vs_same_gpu_kernel=lr_magnitude_error,
            lowrank_magnitude_ssim_vs_same_gpu_kernel=lr_magnitude_ssim,
            lowrank_alignment_scale_real=lr_scale_real,
            lowrank_alignment_scale_imag=lr_scale_imag,
            kernel_complex_error_vs_baseline_gpu=kernel_complex_error,
            kernel_magnitude_nrmse_vs_baseline_gpu=kernel_magnitude_error,
            kernel_magnitude_ssim_vs_baseline_gpu=kernel_magnitude_ssim,
            lowrank_shared_rank_mean=lowrank_summary.shared_rank_mean,
            kernel_successful_repeats=kernel_summary.successful_repeats,
            lowrank_successful_repeats=lowrank_summary.successful_repeats,
        ))
    end
    return quality_rows
end

function build_comparison_rows(summary_rows, quality_rows)
    kernel_name = "HighOrderOp_Kernel"
    lowrank_name = "HighOrderLowRankOp"
    rows = NamedTuple[]

    for gpu_count in GPU_COUNTS
        kernel = find_summary(summary_rows, kernel_name, gpu_count)
        lowrank = find_summary(summary_rows, lowrank_name, gpu_count)
        quality = only([row for row in quality_rows if row.gpu_count == gpu_count])

        push!(rows, (
            gpu_count=gpu_count,
            gpu_ids=kernel.gpu_ids,
            repeats=N_REPEATS,
            lowrank_rank=LOWRANK_RANK,
            kernel_setup_median_s=kernel.setup_median_s,
            lowrank_setup_median_s=lowrank.setup_median_s,
            kernel_recon_median_s=kernel.recon_median_s,
            lowrank_recon_median_s=lowrank.recon_median_s,
            kernel_total_median_s=kernel.total_median_s,
            lowrank_total_median_s=lowrank.total_median_s,
            lowrank_speedup_vs_kernel_setup=safe_ratio(
                kernel.setup_median_s,
                lowrank.setup_median_s,
            ),
            lowrank_speedup_vs_kernel_recon=safe_ratio(
                kernel.recon_median_s,
                lowrank.recon_median_s,
            ),
            lowrank_speedup_vs_kernel_total=safe_ratio(
                kernel.total_median_s,
                lowrank.total_median_s,
            ),
            kernel_recon_speedup_vs_baseline=kernel.recon_speedup_vs_baseline,
            lowrank_recon_speedup_vs_baseline=lowrank.recon_speedup_vs_baseline,
            kernel_total_speedup_vs_baseline=kernel.total_speedup_vs_baseline,
            lowrank_total_speedup_vs_baseline=lowrank.total_speedup_vs_baseline,
            kernel_recon_parallel_efficiency=kernel.recon_parallel_efficiency,
            lowrank_recon_parallel_efficiency=lowrank.recon_parallel_efficiency,
            kernel_total_parallel_efficiency=kernel.total_parallel_efficiency,
            lowrank_total_parallel_efficiency=lowrank.total_parallel_efficiency,
            lowrank_shared_rank_mean=quality.lowrank_shared_rank_mean,
            lowrank_complex_error_vs_kernel=quality.lowrank_complex_error_vs_same_gpu_kernel,
            lowrank_magnitude_nrmse_vs_kernel=quality.lowrank_magnitude_nrmse_vs_same_gpu_kernel,
            lowrank_magnitude_ssim_vs_kernel=quality.lowrank_magnitude_ssim_vs_same_gpu_kernel,
            kernel_complex_error_vs_baseline_gpu=quality.kernel_complex_error_vs_baseline_gpu,
            kernel_magnitude_nrmse_vs_baseline_gpu=quality.kernel_magnitude_nrmse_vs_baseline_gpu,
            kernel_magnitude_ssim_vs_baseline_gpu=quality.kernel_magnitude_ssim_vs_baseline_gpu,
        ))
    end

    return rows
end


# -----------------------------------------------------------------------------
# Main benchmark
# -----------------------------------------------------------------------------

function run_gpus_benchmark!()
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    data_name = splitext(basename(DATA_FILE))[1]
    run_dir = joinpath(
        OUTPUT_ROOT,
        "2d_kernel_lowrank_gpus_$(data_name)_$(timestamp)",
    )
    mkpath(run_dir)

    checkpoint_path = joinpath(run_dir, "runs_checkpoint.csv")
    runs_path = joinpath(run_dir, "runs.csv")
    summary_path = joinpath(run_dir, "summary.csv")
    quality_path = joinpath(run_dir, "quality.csv")
    comparison_path = joinpath(run_dir, "comparison.csv")

    WARMUP && warmup_all!()

    runs = NamedTuple[]
    final_reconstructions = Dict{Tuple{String,Int},Any}()
    execution_order = 0

    for repeat_id in 1:N_REPEATS
        counts_this_repeat = isodd(repeat_id) ? GPU_COUNTS : reverse(GPU_COUNTS)

        for gpu_count in counts_this_repeat
            # Alternate which method runs first at each point. Across five
            # repeats, neither method consistently benefits from running first.
            method_order = isodd(repeat_id + gpu_count) ?
                (:kernel, :lowrank) : (:lowrank, :kernel)

            for method in method_order
                execution_order += 1
                gpu_ids = gpu_ids_for_count(gpu_count)
                @info "Timed 2D GPU-scaling run" method=method_name(method) gpu_count gpu_ids=join(gpu_ids, ',') repeat_id execution_order total_executions=2 * length(GPU_COUNTS) * N_REPEATS

                row, x_host = benchmark_once(
                    method,
                    gpu_count,
                    repeat_id,
                    execution_order,
                )
                push!(runs, row)

                if row.status == "ok" && x_host !== nothing
                    final_reconstructions[(row.method, gpu_count)] = x_host
                end
            end
        end

        write_csv(checkpoint_path, runs)
        @info "Benchmark checkpoint written" repeat_id checkpoint_path
    end

    raw_summary_rows = NamedTuple[]
    for gpu_count in GPU_COUNTS
        push!(raw_summary_rows, summarize_group(
            "HighOrderOp_Kernel",
            gpu_count,
            runs,
        ))
        push!(raw_summary_rows, summarize_group(
            "HighOrderLowRankOp",
            gpu_count,
            runs,
        ))
    end
    summary_rows = add_scaling_metrics(raw_summary_rows)
    quality_rows = compute_quality_rows(final_reconstructions, summary_rows)
    comparison_rows = build_comparison_rows(summary_rows, quality_rows)

    write_csv(runs_path, runs)
    write_csv(summary_path, summary_rows)
    write_csv(quality_path, quality_rows)
    write_csv(comparison_path, comparison_rows)

    println("\n2D Kernel vs LowRank GPU-scaling benchmark complete")
    println("Run directory: $run_dir")
    println("Repeats per method/configuration: $N_REPEATS")
    println("LowRank rank: $LOWRANK_RANK")
    println()
    @printf(
        "%-6s %-14s %-14s %-14s %-14s %-10s\n",
        "GPUs",
        "Kernel recon",
        "LowRank recon",
        "LR/K speedup",
        "LR NRMSE",
        "LR SSIM",
    )
    for row in comparison_rows
        metrics = (
            row.kernel_recon_median_s,
            row.lowrank_recon_median_s,
            row.lowrank_speedup_vs_kernel_recon,
            row.lowrank_magnitude_nrmse_vs_kernel,
            row.lowrank_magnitude_ssim_vs_kernel,
        )
        if any(ismissing, metrics)
            println(
                rpad(string(row.gpu_count), 6),
                rpad(string(row.kernel_recon_median_s), 14),
                rpad(string(row.lowrank_recon_median_s), 14),
                rpad(string(row.lowrank_speedup_vs_kernel_recon), 14),
                string(row.lowrank_magnitude_nrmse_vs_kernel),
                string(row.lowrank_magnitude_ssim_vs_kernel),
            )
        else
            @printf(
                "%-6d %-14.3f %-14.3f %-14.3f %-14.3e %-10.5f\n",
                row.gpu_count,
                row.kernel_recon_median_s,
                row.lowrank_recon_median_s,
                row.lowrank_speedup_vs_kernel_recon,
                row.lowrank_magnitude_nrmse_vs_kernel,
                row.lowrank_magnitude_ssim_vs_kernel,
            )
        end
    end

    @info "GPU-scaling benchmark outputs written" run_dir runs_path summary_path quality_path comparison_path

    return (
        run_dir=run_dir,
        runs_path=runs_path,
        summary_path=summary_path,
        quality_path=quality_path,
        comparison_path=comparison_path,
        runs=runs,
        summary=summary_rows,
        quality=quality_rows,
        comparison=comparison_rows,
    )
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_gpus_benchmark!()
end
