"""
Benchmark 2D `HighOrderKernelOp` and `HighOrderLowRankOp` reconstruction
quality and runtime versus global coil-compression rank.

The physical 32-channel reconstruction is computed first and remains the
fixed reference. One full-data SVD basis is fitted once and truncated to the
same nested virtual-coil basis at 4, 8, 12, 16, 20, 24, and 28 channels. The same
transform is applied to k-space data and CSMs for both operators. Every
operator/channel combination is reconstructed independently ten times.

Run from the repository root:

    /home/jyzhang/julia/julia-1.12.7/bin/julia --project=benchmark --threads=4 \
        benchmark/run/sweep_2d_coil_compression.jl

Useful overrides:

    HIGHORDER_2D_CC_DATA=/path/to/2d.mat
    HIGHORDER_2D_CC_GPUS=0,1,2,3,4,5,6,7
    HIGHORDER_2D_CC_COIL_COUNTS=4,8,12,16,20,24,28,32
    HIGHORDER_2D_CC_REPEATS=10
    HIGHORDER_2D_CC_CG_ITERATIONS=20
    HIGHORDER_2D_CC_LOWRANK_RANK=25
    HIGHORDER_2D_CC_SHARED_BASIS_TOL=1e-2
    HIGHORDER_2D_CC_WARMUP=true
    HIGHORDER_2D_CC_RUN_DIR=/path/to/resume_or_output_directory
    HIGHORDER_2D_CC_SAVE_IMAGES=true

The MAT dataset used by default contains no noise-only acquisition. This
benchmark therefore measures conventional global SVD coil compression with
identity noise covariance; it does not claim noise-whitened compression.
"""

using CUDA
using AbstractNFFTs
using NFFT
using Dates
using LinearAlgebra
using MAT
using Printf
using PyPlot
using RegularizedLeastSquares
using Serialization
using Statistics
using HighOrderMRI

include(joinpath(@__DIR__, "..", "common", "benchmark_utils.jl"))


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

const CC2D_T = Float32
const CC2D_DATA_FILE = normpath(get(
    ENV,
    "HIGHORDER_2D_CC_DATA",
    joinpath(@__DIR__, "..", "..", "demo", "7T_2D_Spiral_1p0_200_r4.mat"),
))
const CC2D_RESULTS_ROOT = normpath(joinpath(@__DIR__, "..", "results"))
const CC2D_TIMESTAMP = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
const CC2D_DATASET_TAG = splitext(basename(CC2D_DATA_FILE))[1]
const CC2D_RUN_DIR = normpath(get(
    ENV,
    "HIGHORDER_2D_CC_RUN_DIR",
    joinpath(
        CC2D_RESULTS_ROOT,
        "2d_coil_compression_$(CC2D_DATASET_TAG)_$(CC2D_TIMESTAMP)",
    ),
))

function parse_env_int_list(name::AbstractString, default::AbstractString)
    raw = strip(get(ENV, name, default))
    isempty(raw) && return Int[]
    return parse.(Int, strip.(split(raw, ',')))
end

const CC2D_GPU_IDS = parse_env_gpus(
    "HIGHORDER_2D_CC_GPUS",
    [0, 1, 2, 3, 4, 5, 6, 7],
)
const CC2D_ARRAY_TYPE = CuArray
const CC2D_METHODS = (:kernel, :lowrank)
const CC2D_COIL_COUNTS = parse_env_int_list(
    "HIGHORDER_2D_CC_COIL_COUNTS",
    "4,8,12,16,20,24,28,32",
)
const CC2D_REPEATS = parse_env_int("HIGHORDER_2D_CC_REPEATS", 10)
const CC2D_CG_ITERATIONS = parse_env_int("HIGHORDER_2D_CC_CG_ITERATIONS", 20)
const CC2D_NBLOCK = parse_env_int(
    "HIGHORDER_2D_CC_NBLOCK",
    parse_env_int("HIGHORDER_2D_CC_KERNEL_NBLOCK", 50),
)
const CC2D_LOWRANK_RANK = parse_env_int(
    "HIGHORDER_2D_CC_LOWRANK_RANK",
    25,
)
const CC2D_RSVD_SEED = parse_env_int("HIGHORDER_2D_CC_RSVD_SEED", 1234)
const CC2D_RSVD_CHUNK = parse_env_int("HIGHORDER_2D_CC_RSVD_CHUNK", 4096)
const CC2D_RSVD_OVERSAMPLE = parse_env_int(
    "HIGHORDER_2D_CC_RSVD_OVERSAMPLE",
    5,
)
const CC2D_SHARED_RANK_MAX = parse_env_int(
    "HIGHORDER_2D_CC_SHARED_RANK_MAX",
    32,
)
const CC2D_SHARED_BASIS_TOL = CC2D_T(parse(
    Float64,
    get(ENV, "HIGHORDER_2D_CC_SHARED_BASIS_TOL", "1e-2"),
))
const CC2D_REGULARIZATION = CC2D_T(parse(
    Float64,
    get(ENV, "HIGHORDER_2D_CC_REGULARIZATION", "1e-9"),
))
const CC2D_RECON_TERMS = get(ENV, "HIGHORDER_2D_CC_RECON_TERMS", "0111")
const CC2D_WARMUP = parse_env_bool("HIGHORDER_2D_CC_WARMUP", true)
const CC2D_WARMUP_COILS = parse_env_int("HIGHORDER_2D_CC_WARMUP_COILS", 4)
const CC2D_SAVE_IMAGES = parse_env_bool("HIGHORDER_2D_CC_SAVE_IMAGES", true)
const CC2D_IMAGE_VMAX_PERCENTILE = parse(
    Float64,
    get(ENV, "HIGHORDER_2D_CC_IMAGE_VMAX_PERCENTILE", "99.9"),
)
const CC2D_DIFFERENCE_SCALE = parse(
    Float64,
    get(ENV, "HIGHORDER_2D_CC_DIFFERENCE_SCALE", "10"),
)
const CC2D_CHECKPOINT_FILE = joinpath(CC2D_RUN_DIR, "runs_checkpoint.jls")
const CC2D_REFERENCE_FILE = joinpath(CC2D_RUN_DIR, "reference_32ch.jls")

isempty(CC2D_GPU_IDS) && throw(ArgumentError("HIGHORDER_2D_CC_GPUS must not be empty"))
isempty(CC2D_COIL_COUNTS) && throw(ArgumentError(
    "HIGHORDER_2D_CC_COIL_COUNTS must contain at least one channel count",
))
length(unique(CC2D_COIL_COUNTS)) == length(CC2D_COIL_COUNTS) ||
    throw(ArgumentError("HIGHORDER_2D_CC_COIL_COUNTS contains duplicates"))
all(>(0), CC2D_COIL_COUNTS) || throw(ArgumentError(
    "Every requested virtual-coil count must be positive",
))
CC2D_REPEATS >= 1 || throw(ArgumentError("HIGHORDER_2D_CC_REPEATS must be positive"))
CC2D_CG_ITERATIONS >= 1 || throw(ArgumentError(
    "HIGHORDER_2D_CC_CG_ITERATIONS must be positive",
))
CC2D_NBLOCK >= 1 || throw(ArgumentError(
    "HIGHORDER_2D_CC_NBLOCK must be positive",
))
CC2D_LOWRANK_RANK >= 1 || throw(ArgumentError(
    "HIGHORDER_2D_CC_LOWRANK_RANK must be positive",
))
CC2D_RSVD_CHUNK >= 1 || throw(ArgumentError(
    "HIGHORDER_2D_CC_RSVD_CHUNK must be positive",
))
CC2D_RSVD_OVERSAMPLE >= 0 || throw(ArgumentError(
    "HIGHORDER_2D_CC_RSVD_OVERSAMPLE must be nonnegative",
))
CC2D_SHARED_RANK_MAX >= 1 || throw(ArgumentError(
    "HIGHORDER_2D_CC_SHARED_RANK_MAX must be positive",
))
CC2D_SHARED_BASIS_TOL >= 0 || throw(ArgumentError(
    "HIGHORDER_2D_CC_SHARED_BASIS_TOL must be nonnegative",
))
CC2D_DIFFERENCE_SCALE > 0 || throw(ArgumentError(
    "HIGHORDER_2D_CC_DIFFERENCE_SCALE must be positive",
))


# -----------------------------------------------------------------------------
# Dataset and coil-basis preparation
# -----------------------------------------------------------------------------

function require_mat_keys(raw::AbstractDict, keys)
    missing_keys = filter(key -> !haskey(raw, key), keys)
    isempty(missing_keys) || throw(ArgumentError(
        "2D MAT file is missing keys: $(join(missing_keys, ", "))",
    ))
    return nothing
end

function mask_bounding_box_2d(mask::AbstractMatrix{Bool})
    indices = findall(mask)
    isempty(indices) && throw(ArgumentError("Reconstruction mask must not be empty"))
    rows = extrema(index -> index[1], indices)
    columns = extrema(index -> index[2], indices)
    return (rows[1]:rows[2], columns[1]:columns[2])
end

function load_2d_coil_dataset(path::AbstractString)
    isfile(path) || error(
        "2D benchmark data file was not found: $path\n" *
        "Set HIGHORDER_2D_CC_DATA to the MAT file path.",
    )
    raw = matread(path)
    require_mat_keys(raw, (
        "gre_csm",
        "gre_b0",
        "gre_mask",
        "kdata",
        "datatime",
        "matrixSize",
        "FOV",
        "k0_adc",
        "dt_Measured",
        "ksphaMeasured",
        "startMeasured",
        "tauMeasured",
        "dt_adc",
    ))

    csm = raw["gre_csm"]
    fieldmap = raw["gre_b0"]
    mask = raw["gre_mask"]
    kdata = raw["kdata"]
    datatime = vec(raw["datatime"])
    matrix_size = Tuple(Int.(vec(raw["matrixSize"])))
    fov = Tuple(vec(raw["FOV"]))
    k0_ecc = raw["k0_adc"]
    dt_measured = raw["dt_Measured"]
    kspha_measured = raw["ksphaMeasured"]
    start_measured = raw["startMeasured"]
    tau_measured = raw["tauMeasured"]
    dt_adc = raw["dt_adc"]

    length(matrix_size) == 3 || throw(DimensionMismatch(
        "matrixSize must contain three entries; got $matrix_size",
    ))
    length(fov) == 3 || throw(DimensionMismatch("FOV must contain three entries"))
    nX, nY, nZ = matrix_size
    nZ == 1 || throw(ArgumentError(
        "This benchmark is for 2D data with nZ=1; got matrixSize=$matrix_size",
    ))
    size(kdata, 1) == length(datatime) || throw(DimensionMismatch(
        "kdata and datatime sample counts do not match",
    ))
    size(kdata, 2) == size(csm, 3) || throw(DimensionMismatch(
        "kdata and CSM channel counts do not match",
    ))

    kspha_interpolated = InterpTrajTime(
        kspha_measured,
        dt_measured,
        start_measured + tau_measured * dt_adc,
        datatime,
    )
    weight_host = samplingDensity(
        kspha_interpolated'[2:3, :],
        (nX, nY),
    )
    kdata = kdata ./ exp.(2π * im .* k0_ecc)'
    kdata = kdata .* exp.(-2π * im .* kspha_interpolated[:, 1])

    shift_x, shift_y = 0, -1
    grid = Grid(
        nX,
        nY,
        nZ,
        CC2D_T(fov[1] / nX),
        CC2D_T(fov[2] / nY),
        CC2D_T(fov[3] / nZ);
        exchange_xy=false,
        reverse_x=false,
        reverse_y=false,
    )
    csm_model = Complex{CC2D_T}.(permutedims(
        reverse(circshift(csm, (shift_x, shift_y, 0)); dims=1),
        (2, 1, 3),
    ))
    fieldmap_model = CC2D_T.(permutedims(
        reverse(circshift(fieldmap, (shift_x, shift_y)); dims=1),
        (2, 1),
    ))
    mask_model = Bool.(permutedims(
        reverse(circshift(mask, (shift_x, shift_y)); dims=1),
        (2, 1),
    ))

    return (
        grid=grid,
        matrix_size=(nX, nY),
        data_host=Complex{CC2D_T}.(kdata),
        weight_host=Complex{CC2D_T}.(weight_host),
        datatime=CC2D_T.(datatime),
        kspha=CC2D_T.(kspha_interpolated'),
        csm=csm_model,
        fieldmap=fieldmap_model,
        mask=mask_model,
        roi=mask_bounding_box_2d(mask_model),
        input_coils=size(kdata, 2),
        n_sample=size(kdata, 1),
    )
end

function elapsed_host_seconds(f::F) where {F}
    GC.gc(true)
    start = time_ns()
    value = f()
    return value, (time_ns() - start) * 1e-9
end

function truncate_coil_transform_2d(
    full_transform::CoilCompressionTransform{T},
    n_virtual_coils::Integer,
) where {T<:AbstractFloat}
    input_coils = size(full_transform, 1)
    1 <= n_virtual_coils <= input_coils || throw(ArgumentError(
        "n_virtual_coils=$n_virtual_coils must be in 1:$input_coils",
    ))
    singular_values = copy(full_transform.singular_values)
    total_energy = sum(abs2, singular_values)
    total_energy > zero(T) || throw(ArgumentError(
        "The fitted coil spectrum has zero energy",
    ))
    retained_energy = sum(
        abs2,
        view(singular_values, 1:n_virtual_coils),
    ) / total_energy
    compression_matrix = Matrix{Complex{T}}(
        view(full_transform.compression_matrix, :, 1:n_virtual_coils),
    )
    noise_covariance = isnothing(full_transform.noise_covariance) ?
        nothing : copy(full_transform.noise_covariance)
    return CoilCompressionTransform(
        compression_matrix,
        singular_values,
        T(retained_energy),
        noise_covariance,
        full_transform.prewhitening_scale_factor,
    )
end

function write_coil_spectrum_2d(full_transform)
    singular_values = full_transform.singular_values
    squared = abs2.(singular_values)
    total = sum(squared)
    cumulative = cumsum(squared) ./ total
    rows = NamedTuple[
        (
            coil_component=index,
            singular_value=value,
            energy_fraction=squared[index] / total,
            cumulative_energy=cumulative[index],
        )
        for (index, value) in enumerate(singular_values)
    ]
    write_csv(joinpath(CC2D_RUN_DIR, "coil_spectrum.csv"), rows)
    return nothing
end

function prepare_virtual_coils(dataset, full_transform, n_virtual_coils::Int)
    if n_virtual_coils == dataset.input_coils
        transform = full_transform
        data_host = dataset.data_host
        csm = dataset.csm
        data_compression_s = 0.0
        csm_compression_s = 0.0
    else
        transform = truncate_coil_transform_2d(full_transform, n_virtual_coils)
        data_host, data_compression_s = elapsed_host_seconds(
            () -> apply_coil_compression(
                dataset.data_host,
                transform;
                coil_dim=2,
            ),
        )
        csm, csm_compression_s = elapsed_host_seconds(
            () -> apply_coil_compression(
                dataset.csm,
                transform;
                coil_dim=3,
            ),
        )
    end

    return (
        data=data_host,
        csm=csm,
        transform=transform,
        data_compression_s=data_compression_s,
        csm_compression_s=csm_compression_s,
        compression_apply_s=data_compression_s + csm_compression_s,
    )
end


# -----------------------------------------------------------------------------
# Operator reconstruction and quality metrics
# -----------------------------------------------------------------------------

function operator_name_2d(method::Symbol)
    method === :kernel && return "HighOrderKernelOp"
    method === :lowrank && return "HighOrderLowRankOp"
    throw(ArgumentError("Unknown 2D coil-compression method: $method"))
end

function operator_tag_2d(method::Symbol)
    method === :kernel && return "kernel"
    method === :lowrank && return "lowrank"
    throw(ArgumentError("Unknown 2D coil-compression method: $method"))
end

operator_gpu_ids_2d(::Symbol) = CC2D_GPU_IDS

function build_kernel_operator_2d(dataset, csm)
    return HighOrderKernelOp(
        dataset.grid,
        CC2D_T.(dataset.kspha),
        CC2D_T.(dataset.datatime);
        fieldmap=CC2D_T.(dataset.fieldmap),
        csm=Complex{CC2D_T}.(csm),
        mask=dataset.mask,
        recon_terms=CC2D_RECON_TERMS,
        nBlock=Int64(CC2D_NBLOCK),
        arrayType=CC2D_ARRAY_TYPE,
        gpus=CC2D_GPU_IDS,
        verbose=false,
    )
end

function build_lowrank_operator_2d(dataset, csm)
    return HighOrderLowRankOp(
        dataset.grid,
        CC2D_T.(dataset.kspha),
        CC2D_T.(dataset.datatime);
        recon_terms=CC2D_RECON_TERMS,
        csm=Complex{CC2D_T}.(csm),
        fieldmap=CC2D_T.(dataset.fieldmap),
        mask=dataset.mask,
        arrayType=CC2D_ARRAY_TYPE,
        gpus=CC2D_GPU_IDS,
        L_rank=CC2D_LOWRANK_RANK,
        rsvd_seed=CC2D_RSVD_SEED,
        rsvd_chunk=CC2D_RSVD_CHUNK,
        rsvd_oversample=CC2D_RSVD_OVERSAMPLE,
        rsvd_finalize=:gram,
        rsvd_backend=:kernel,
        rsvd_distribution=:voxel,
        shared_rank_max=CC2D_SHARED_RANK_MAX,
        shared_basis_tol=CC2D_SHARED_BASIS_TOL,
        normal_distribution=:channel,
        verbose=false,
    )
end

function build_operator_2d(method::Symbol, dataset, csm)
    method === :kernel && return build_kernel_operator_2d(dataset, csm)
    method === :lowrank && return build_lowrank_operator_2d(dataset, csm)
    throw(ArgumentError("Unknown 2D coil-compression method: $method"))
end

function reconstruction_parameters(iterations::Integer=CC2D_CG_ITERATIONS)
    return Dict{Symbol,Any}(
        :reconSize => nothing,
        :reg => L2Regularization(CC2D_REGULARIZATION),
        :iterations => Int(iterations),
        :solver => CGNR,
    )
end

function prepare_reconstruction_inputs_2d(method::Symbol, prepared, weight)
    if method === :kernel
        inputs, input_prepare_s = elapsed_host_seconds(
            () -> (Complex{CC2D_T}.(prepared.data), Complex{CC2D_T}.(weight)),
        )
        return (
            kdata=inputs[1],
            weight=inputs[2],
            input_prepare_s=input_prepare_s,
        )
    end

    gpu_ids = operator_gpu_ids_2d(method)
    CUDA.device!(first(gpu_ids))
    inputs, input_prepare_s = elapsed_seconds(
        () -> (
            CC2D_ARRAY_TYPE{Complex{CC2D_T}}(prepared.data),
            CC2D_ARRAY_TYPE{Complex{CC2D_T}}(weight),
        ),
        gpu_ids,
    )
    return (
        kdata=inputs[1],
        weight=inputs[2],
        input_prepare_s=input_prepare_s,
    )
end

function reconstruct_operator_once(
    method::Symbol,
    dataset,
    prepared,
    weight,
    repeat_id::Int,
)
    setup_s = missing
    recon_s = missing
    input_prepare_s = missing
    shared_rank = missing
    x_host = nothing
    op = nothing
    inputs = nothing
    status = "ok"
    error_message = ""
    gpu_ids = operator_gpu_ids_2d(method)
    free_before_mib = gpu_free_memory_mib(gpu_ids)

    try
        GC.gc(true)
        CUDA.device!(first(gpu_ids))
        op, setup_s = elapsed_seconds(
            () -> build_operator_2d(method, dataset, prepared.csm),
            gpu_ids,
        )
        shared_rank = method === :lowrank ? size(op.basis, 2) : missing
        try
            CUDA.device!(first(gpu_ids))
            rec_params = reconstruction_parameters()
            rec_params[:reconSize] = dataset.matrix_size
            inputs = prepare_reconstruction_inputs_2d(method, prepared, weight)
            input_prepare_s = inputs.input_prepare_s
            x, recon_s = elapsed_seconds(
                () -> recon_HOOp(
                    op,
                    inputs.kdata,
                    inputs.weight,
                    rec_params,
                ),
                gpu_ids,
            )
            x_host = reshape(
                Complex{CC2D_T}.(require_reconstruction(
                    x,
                    "$(operator_name_2d(method)) coil-compression reconstruction",
                )),
                dataset.matrix_size,
            )
            all(isfinite, x_host) || throw(DomainError(
                nothing,
                "Reconstruction contains non-finite values",
            ))
        finally
            method === :lowrank && !isnothing(op) && close(op)
            inputs = nothing
            op = nothing
            GC.gc(true)
        end
    catch err
        status = "error"
        error_message = replace(
            sprint(showerror, err, catch_backtrace()),
            '\n' => " | ",
            '\r' => "",
        )
        @error(
            "2D coil-compression reconstruction failed",
            operator=operator_name_2d(method),
            repeat_id,
            error_message,
        )
    end

    return (
        x=x_host,
        status=status,
        error=error_message,
        setup_s=setup_s,
        recon_s=recon_s,
        input_prepare_s=input_prepare_s,
        shared_rank=shared_rank,
        operator_total_s=benchmark_stage_total(setup_s, recon_s),
        free_before_mib=free_before_mib,
        free_after_mib=gpu_free_memory_mib(gpu_ids),
    )
end

function reconstruction_metrics_2d(x, reference, dataset)
    mask_values = vec(dataset.mask)
    x_masked = vec(x)[mask_values]
    reference_masked = vec(reference)[mask_values]
    return (
        raw_complex_nrmse_masked=raw_complex_nrmse(x_masked, reference_masked),
        magnitude_nrmse_masked=magnitude_nrmse(x_masked, reference_masked),
        magnitude_ssim_roi=magnitude_ssim(
            view(x, dataset.roi...),
            view(reference, dataset.roi...),
        ),
        raw_complex_nrmse_full=raw_complex_nrmse(x, reference),
        magnitude_nrmse_full=magnitude_nrmse(x, reference),
        magnitude_ssim_full=magnitude_ssim(x, reference),
    )
end

function empty_metrics_2d()
    return (
        raw_complex_nrmse_masked=missing,
        magnitude_nrmse_masked=missing,
        magnitude_ssim_roi=missing,
        raw_complex_nrmse_full=missing,
        magnitude_nrmse_full=missing,
        magnitude_ssim_full=missing,
    )
end

function display_reconstruction_2d(x)
    ndims(x) == 2 || throw(DimensionMismatch(
        "Expected a two-dimensional reconstruction; got size $(size(x))",
    ))
    return rotl90(Array(x))
end

function save_reconstruction_artifacts_2d(
    method::Symbol,
    n_virtual_coils::Int,
    x,
    reference,
    reference_vmax::Real,
)
    CC2D_SAVE_IMAGES || return nothing
    image_dir = joinpath(CC2D_RUN_DIR, "reconstruction")
    difference_dir = joinpath(CC2D_RUN_DIR, "difference")
    mkpath(image_dir)
    mkpath(difference_dir)

    reconstruction = abs.(display_reconstruction_2d(x))
    difference = benchmark_magnitude_difference(
        reconstruction, abs.(display_reconstruction_2d(reference));
        scale=CC2D_DIFFERENCE_SCALE)
    tag = @sprintf("%s_coil%02d", operator_tag_2d(method), n_virtual_coils)

    fig = plt_image(reconstruction; vmin=0, vmax=reference_vmax, width=10)
    fig.savefig(
        joinpath(image_dir, "$(tag)_magnitude.png");
        dpi=300,
        transparent=false,
        bbox_inches="tight",
        pad_inches=0,
    )
    PyPlot.close(fig)

    fig = plt_image(difference; vmin=0, vmax=reference_vmax, width=10)
    fig.savefig(
        joinpath(
            difference_dir,
            "$(tag)_magnitude_absdiff_x$(CC2D_DIFFERENCE_SCALE).png",
        );
        dpi=300,
        transparent=false,
        bbox_inches="tight",
        pad_inches=0,
    )
    PyPlot.close(fig)
    return nothing
end


# -----------------------------------------------------------------------------
# Checkpoint and aggregation
# -----------------------------------------------------------------------------

function load_runs_checkpoint_2d()
    isfile(CC2D_CHECKPOINT_FILE) || return NamedTuple[]
    rows = deserialize(CC2D_CHECKPOINT_FILE)
    rows isa AbstractVector || error(
        "Invalid checkpoint file: $CC2D_CHECKPOINT_FILE",
    )
    return NamedTuple[row for row in rows]
end

function save_runs_checkpoint_2d(rows::Vector{NamedTuple})
    mkpath(CC2D_RUN_DIR)
    temporary = CC2D_CHECKPOINT_FILE * ".tmp"
    open(temporary, "w") do io
        serialize(io, rows)
    end
    mv(temporary, CC2D_CHECKPOINT_FILE; force=true)
    write_csv(joinpath(CC2D_RUN_DIR, "runs.csv"), rows)
    return nothing
end

reconstruction_checkpoint_2d(method::Symbol, n_virtual_coils::Integer) = joinpath(
    CC2D_RUN_DIR,
    @sprintf(
        "reconstruction_%s_%02dch.jls",
        operator_tag_2d(method),
        n_virtual_coils,
    ),
)

function save_reconstruction_checkpoint_2d(
    method::Symbol,
    n_virtual_coils::Integer,
    x,
)
    open(reconstruction_checkpoint_2d(method, n_virtual_coils), "w") do io
        serialize(io, x)
    end
    return nothing
end

function load_reference_2d()
    isfile(CC2D_REFERENCE_FILE) || return nothing
    return deserialize(CC2D_REFERENCE_FILE)
end

function save_reference_2d(reference)
    open(CC2D_REFERENCE_FILE, "w") do io
        serialize(io, reference)
    end
    return nothing
end

safe_mean(values) = mean(Float64.(values))
safe_median(values) = median(Float64.(values))
safe_minimum(values) = minimum(Float64.(values))
safe_std(values) = length(values) > 1 ? std(Float64.(values)) : 0.0

function summarize_runs_2d(rows, input_coils::Int)
    successful = [row for row in rows if row.status == "ok"]
    summaries = NamedTuple[]
    for method in CC2D_METHODS
        operator = operator_name_2d(method)
        method_rows = [row for row in successful if row.operator == operator]
        for n_virtual_coils in sort(unique(Int[
            row.virtual_coils for row in method_rows
        ]))
            group = [
                row for row in method_rows
                if row.virtual_coils == n_virtual_coils
            ]
            isempty(group) && continue
            first_row = first(group)
            setup = getproperty.(group, :setup_s)
            recon = getproperty.(group, :recon_s)
            input_prepare = getproperty.(group, :input_prepare_s)
            operator_total = getproperty.(group, :operator_total_s)
            end_to_end = getproperty.(group, :end_to_end_s)
            including_fit = getproperty.(group, :total_including_shared_fit_s)
            raw_complex = getproperty.(group, :raw_complex_nrmse_masked)
            magnitude = getproperty.(group, :magnitude_nrmse_masked)
            ssim = getproperty.(group, :magnitude_ssim_roi)

            push!(summaries, (
                status="ok",
                operator=operator,
                input_coils=input_coils,
                virtual_coils=n_virtual_coils,
                compression_ratio=input_coils / n_virtual_coils,
                coil_basis=first_row.coil_basis,
                retained_energy=first_row.retained_energy,
                discarded_energy=1 - first_row.retained_energy,
                configured_lowrank_rank=first_row.configured_lowrank_rank,
                shared_rank=first_row.shared_rank,
                successful_repeats=length(group),
                requested_repeats=CC2D_REPEATS,
                raw_complex_nrmse_mean=safe_mean(raw_complex),
                raw_complex_nrmse_std=safe_std(raw_complex),
                magnitude_nrmse_mean=safe_mean(magnitude),
                magnitude_nrmse_std=safe_std(magnitude),
                magnitude_ssim_mean=safe_mean(ssim),
                magnitude_ssim_std=safe_std(ssim),
                setup_median_s=safe_median(setup),
                setup_min_s=safe_minimum(setup),
                setup_mean_s=safe_mean(setup),
                setup_std_s=safe_std(setup),
                recon_median_s=safe_median(recon),
                recon_min_s=safe_minimum(recon),
                recon_mean_s=safe_mean(recon),
                recon_std_s=safe_std(recon),
                input_prepare_median_s=safe_median(input_prepare),
                operator_total_median_s=safe_median(operator_total),
                operator_total_mean_s=safe_mean(operator_total),
                operator_total_std_s=safe_std(operator_total),
                compression_apply_s=first_row.compression_apply_s,
                end_to_end_median_s=safe_median(end_to_end),
                end_to_end_mean_s=safe_mean(end_to_end),
                end_to_end_std_s=safe_std(end_to_end),
                total_including_shared_fit_median_s=safe_median(including_fit),
                timing_kind="fresh_operator_with_separately_measured_coil_fit",
            ))
        end
    end

    with_speedups = NamedTuple[]
    for method in CC2D_METHODS
        operator = operator_name_2d(method)
        method_summaries = [row for row in summaries if row.operator == operator]
        baseline_index = findfirst(
            row -> row.virtual_coils == input_coils,
            method_summaries,
        )
        isnothing(baseline_index) && error(
            "No successful $input_coils-channel $operator reconstruction is available",
        )
        baseline = method_summaries[baseline_index]
        append!(with_speedups, [merge(row, (
            cg_speedup_vs_32=baseline.recon_median_s / row.recon_median_s,
            operator_total_speedup_vs_32=
                baseline.operator_total_median_s / row.operator_total_median_s,
            end_to_end_speedup_vs_32=
                baseline.end_to_end_median_s / row.end_to_end_median_s,
            total_including_shared_fit_speedup_vs_32=
                baseline.total_including_shared_fit_median_s /
                    row.total_including_shared_fit_median_s,
        )) for row in method_summaries])
    end
    summaries = with_speedups
    write_csv(joinpath(CC2D_RUN_DIR, "summary.csv"), summaries)
    return summaries
end


# -----------------------------------------------------------------------------
# Warmup and main sweep
# -----------------------------------------------------------------------------

function warmup_coil_compression_2d!(dataset, full_transform, weight)
    1 <= CC2D_WARMUP_COILS <= dataset.input_coils || throw(ArgumentError(
        "HIGHORDER_2D_CC_WARMUP_COILS must be in 1:$(dataset.input_coils)",
    ))
    prepared = prepare_virtual_coils(
        dataset,
        full_transform,
        CC2D_WARMUP_COILS,
    )
    for method in CC2D_METHODS
        @info(
            "Warming up 2D coil-compression path",
            operator=operator_name_2d(method),
            virtual_coils=CC2D_WARMUP_COILS,
            cg_iterations=1,
        )
        gpu_ids = operator_gpu_ids_2d(method)
        CUDA.device!(first(gpu_ids))
        op = build_operator_2d(method, dataset, prepared.csm)
        inputs = nothing
        try
            parameters = reconstruction_parameters(1)
            parameters[:reconSize] = dataset.matrix_size
            inputs = prepare_reconstruction_inputs_2d(method, prepared, weight)
            x = recon_HOOp(
                op,
                inputs.kdata,
                inputs.weight,
                parameters,
            )
            require_reconstruction(
                x,
                "$(operator_name_2d(method)) coil-compression warmup",
            )
            synchronize_gpus!(gpu_ids)
        finally
            method === :lowrank && !isnothing(op) && close(op)
            inputs = nothing
            op = nothing
            GC.gc(true)
        end
    end
    prepared = nothing
    GC.gc(true)
    return nothing
end

function write_configuration_2d(
    dataset,
    full_transform,
    coil_fit_s,
    transform_orthogonality_error,
)
    requested = CC2D_COIL_COUNTS
    row = (
        data_file=CC2D_DATA_FILE,
        run_directory=CC2D_RUN_DIR,
        operators=join(operator_name_2d.(CC2D_METHODS), ','),
        reference="physical_32_channel_HighOrderKernelOp_reconstruction",
        matrix_size=join(dataset.matrix_size, 'x'),
        n_sample=dataset.n_sample,
        input_coils=dataset.input_coils,
        coil_sweep=join(requested, ','),
        coil_compression_method="global_full_data_SVD_identity_noise_covariance",
        noise_covariance_used=false,
        prewhitening_scale_factor=full_transform.prewhitening_scale_factor,
        calibration_observations=size(dataset.data_host, 1),
        coil_fit_s=coil_fit_s,
        transform_orthogonality_error=transform_orthogonality_error,
        gpu_ids=join(CC2D_GPU_IDS, ','),
        nfft_backend=string(AbstractNFFTs.active_backend()),
        kernel_input_storage="Array",
        lowrank_input_storage="CuArray",
        kernel_and_lowrank_use_mask=true,
        repeats=CC2D_REPEATS,
        cg_iterations=CC2D_CG_ITERATIONS,
        regularization=CC2D_REGULARIZATION,
        recon_terms=CC2D_RECON_TERMS,
        nblock=CC2D_NBLOCK,
        lowrank_rank=CC2D_LOWRANK_RANK,
        rsvd_seed=CC2D_RSVD_SEED,
        rsvd_chunk=CC2D_RSVD_CHUNK,
        rsvd_oversample=CC2D_RSVD_OVERSAMPLE,
        rsvd_finalize="gram",
        rsvd_backend="kernel",
        rsvd_distribution="voxel",
        shared_rank_max=CC2D_SHARED_RANK_MAX,
        shared_basis_tol=CC2D_SHARED_BASIS_TOL,
        normal_distribution="channel",
        warmup=CC2D_WARMUP,
        warmup_coils=CC2D_WARMUP_COILS,
        timing_definition=
            "compression_apply + input_prepare + fresh_operator_setup + fixed_iteration_CG",
        julia_version=string(VERSION),
    )
    write_csv(joinpath(CC2D_RUN_DIR, "configuration.csv"), [row])
    return nothing
end

function run_2d_coil_compression_sweep!()
    mkpath(CC2D_RUN_DIR)
    AbstractNFFTs.set_active_backend!(NFFT.backend())
    contract = benchmark_run_contract((data=CC2D_DATA_FILE,), (
        experiment="2d_coil_compression", precision=string(CC2D_T),
        gpus=CC2D_GPU_IDS, methods=collect(string.(CC2D_METHODS)), coils=CC2D_COIL_COUNTS,
        repeats=CC2D_REPEATS, cg_iterations=CC2D_CG_ITERATIONS,
        regularization=CC2D_REGULARIZATION, recon_terms=CC2D_RECON_TERMS,
        nblock=CC2D_NBLOCK, local_rank=CC2D_LOWRANK_RANK, rsvd_seed=CC2D_RSVD_SEED,
        rsvd_chunk=CC2D_RSVD_CHUNK, rsvd_oversample=CC2D_RSVD_OVERSAMPLE,
        shared_rank_max=CC2D_SHARED_RANK_MAX, shared_basis_tol=CC2D_SHARED_BASIS_TOL,
        nfft_backend=string(AbstractNFFTs.active_backend()),
        warmup=CC2D_WARMUP, warmup_coils=CC2D_WARMUP_COILS,
    ); source_root=normpath(joinpath(@__DIR__, "..", "..")), scripts=[@__FILE__])
    validate_benchmark_run!(CC2D_RUN_DIR, contract)
    CUDA.functional() || error("CUDA is not functional on this host")
    available_devices = length(CUDA.devices())
    all(gpu -> 0 <= gpu < available_devices, CC2D_GPU_IDS) || throw(ArgumentError(
        "Requested GPU IDs $(CC2D_GPU_IDS), but available zero-based IDs are " *
        "0:$(available_devices - 1)",
    ))

    @info(
        "Loading 2D two-operator coil-compression benchmark",
        data_file=CC2D_DATA_FILE,
        run_directory=CC2D_RUN_DIR,
        gpu_ids=CC2D_GPU_IDS,
        operators=operator_name_2d.(CC2D_METHODS),
        coil_counts=CC2D_COIL_COUNTS,
    )
    dataset = load_2d_coil_dataset(CC2D_DATA_FILE)
    dataset.input_coils == 32 || throw(ArgumentError(
        "This requested sweep requires 32 input channels; dataset has " *
        "$(dataset.input_coils)",
    ))
    all(count -> count <= dataset.input_coils, CC2D_COIL_COUNTS) ||
        throw(ArgumentError(
            "Every channel count must be at most $(dataset.input_coils)",
        ))
    dataset.input_coils in CC2D_COIL_COUNTS || throw(ArgumentError(
        "HIGHORDER_2D_CC_COIL_COUNTS must include the uncompressed " *
        "$(dataset.input_coils)-channel baseline",
    ))

    full_transform, coil_fit_s = elapsed_host_seconds(
        () -> fit_coil_compression(
            dataset.data_host;
            coil_dim=2,
            n_virtual_coils=dataset.input_coils,
        ),
    )
    write_coil_spectrum_2d(full_transform)
    transform_orthogonality_error = norm(
        full_transform.compression_matrix' * full_transform.compression_matrix -
        Matrix{Complex{CC2D_T}}(I, dataset.input_coils, dataset.input_coils),
    ) / sqrt(dataset.input_coils)
    write_configuration_2d(
        dataset,
        full_transform,
        coil_fit_s,
        transform_orthogonality_error,
    )

    CUDA.device!(first(CC2D_GPU_IDS))
    weight = Complex{CC2D_T}.(dataset.weight_host)
    CC2D_WARMUP && warmup_coil_compression_2d!(
        dataset,
        full_transform,
        weight,
    )

    rows = load_runs_checkpoint_2d()
    isempty(rows) || all(row -> hasproperty(row, :operator), rows) || error(
        "The checkpoint predates the multi-operator benchmark. " *
        "Use a new HIGHORDER_2D_CC_RUN_DIR instead of mixing CSV schemas.",
    )
    completed = Set(
        (String(row.operator), Int(row.virtual_coils), Int(row.repeat))
        for row in rows if row.status == "ok"
    )
    reference = load_reference_2d()
    coil_counts = CC2D_COIL_COUNTS

    for method in CC2D_METHODS
        operator = operator_name_2d(method)
        execution_coil_counts = method === :kernel ?
            [dataset.input_coils; filter(!=(dataset.input_coils), coil_counts)] :
            coil_counts
        for n_virtual_coils in execution_coil_counts
            pending_repeats = [
                repeat_id for repeat_id in 1:CC2D_REPEATS
                if !((operator, n_virtual_coils, repeat_id) in completed)
            ]
            if isempty(pending_repeats)
                @info(
                    "Skipping checkpointed operator/channel combination",
                    operator,
                    n_virtual_coils,
                )
                continue
            end

            prepared = prepare_virtual_coils(
                dataset,
                full_transform,
                n_virtual_coils,
            )
            retained_energy = n_virtual_coils == dataset.input_coils ?
                1.0 : Float64(prepared.transform.retained_energy)

            for repeat_id in pending_repeats
                @info(
                    "Benchmarking 2D coil compression",
                    operator,
                    n_virtual_coils,
                    repeat_id,
                    repeats=CC2D_REPEATS,
                )
                result = reconstruct_operator_once(
                    method,
                    dataset,
                    prepared,
                    weight,
                    repeat_id,
                )
                metrics = empty_metrics_2d()

                if result.status == "ok"
                    x_host = result.x
                    if method === :kernel &&
                       n_virtual_coils == dataset.input_coils &&
                       isnothing(reference)
                        reference = copy(x_host)
                        save_reference_2d(reference)
                    end
                    isnothing(reference) && error(
                        "The 32-channel Kernel reference must succeed before " *
                        "other reconstruction metrics can be computed",
                    )
                    metrics = reconstruction_metrics_2d(
                        x_host,
                        reference,
                        dataset,
                    )
                    save_reconstruction_checkpoint_2d(
                        method,
                        n_virtual_coils,
                        x_host,
                    )
                end

                end_to_end_s = if result.status == "ok"
                    benchmark_stage_total(prepared.compression_apply_s,
                        result.input_prepare_s, result.operator_total_s)
                else
                    missing
                end
                row = merge((
                    status=result.status,
                    error=result.error,
                    operator=operator,
                    input_coils=dataset.input_coils,
                    virtual_coils=n_virtual_coils,
                    repeat=repeat_id,
                    compression_ratio=dataset.input_coils / n_virtual_coils,
                    coil_basis=n_virtual_coils == dataset.input_coils ?
                        "physical_channels" : "global_full_data_SVD",
                    retained_energy=retained_energy,
                    discarded_energy=1 - retained_energy,
                    configured_lowrank_rank=method === :lowrank ?
                        CC2D_LOWRANK_RANK : missing,
                    shared_rank=result.shared_rank,
                    coil_fit_s=coil_fit_s,
                    data_compression_s=prepared.data_compression_s,
                    csm_compression_s=prepared.csm_compression_s,
                    compression_apply_s=prepared.compression_apply_s,
                    input_prepare_s=result.input_prepare_s,
                    setup_s=result.setup_s,
                    recon_s=result.recon_s,
                    operator_total_s=result.operator_total_s,
                    end_to_end_s=end_to_end_s,
                    total_including_shared_fit_s=benchmark_stage_total(coil_fit_s, end_to_end_s),
                    timing_kind="fresh_operator_with_separately_measured_coil_fit",
                    free_before_mib=result.free_before_mib,
                    free_after_mib=result.free_after_mib,
                    cg_iterations=CC2D_CG_ITERATIONS,
                    regularization=CC2D_REGULARIZATION,
                    recon_terms=CC2D_RECON_TERMS,
                    gpu_ids=join(operator_gpu_ids_2d(method), ','),
                    nfft_backend=string(AbstractNFFTs.active_backend()),
                    input_storage=method === :kernel ? "Array" : "CuArray",
                    rsvd_seed=method === :lowrank ? CC2D_RSVD_SEED : missing,
                    rsvd_chunk=method === :lowrank ? CC2D_RSVD_CHUNK : missing,
                    rsvd_oversample=method === :lowrank ?
                        CC2D_RSVD_OVERSAMPLE : missing,
                    shared_rank_max=method === :lowrank ?
                        CC2D_SHARED_RANK_MAX : missing,
                    shared_basis_tol=method === :lowrank ?
                        CC2D_SHARED_BASIS_TOL : missing,
                    julia_version=string(VERSION),
                ), metrics)
                push!(rows, row)
                result.status == "ok" && push!(
                    completed,
                    (operator, n_virtual_coils, repeat_id),
                )
                save_runs_checkpoint_2d(rows)
            end

            prepared = nothing
            GC.gc(true)
        end
    end

    summaries = summarize_runs_2d(rows, dataset.input_coils)
    reference = load_reference_2d()
    isnothing(reference) && error("32-channel reference file was not produced")
    reference_vmax = quantile(
        vec(abs.(display_reconstruction_2d(reference))),
        CC2D_IMAGE_VMAX_PERCENTILE / 100,
    )
    for method in CC2D_METHODS
        for n_virtual_coils in coil_counts
            checkpoint = reconstruction_checkpoint_2d(method, n_virtual_coils)
            isfile(checkpoint) || continue
            save_reconstruction_artifacts_2d(
                method,
                n_virtual_coils,
                deserialize(checkpoint),
                reference,
                reference_vmax,
            )
        end
    end

    for gpu_id in CC2D_GPU_IDS
        CUDA.device!(gpu_id)
        CUDA.reclaim()
    end
    CUDA.device!(first(CC2D_GPU_IDS))

    println("\n2D Kernel/LowRank coil-compression sweep complete")
    println("Run directory: $CC2D_RUN_DIR")
    println("Runs: $(joinpath(CC2D_RUN_DIR, "runs.csv"))")
    println("Summary: $(joinpath(CC2D_RUN_DIR, "summary.csv"))")
    for row in summaries
        @printf(
            "%-18s %2d channels: magnitude NRMSE = %.4e, SSIM = %.6f, setup/recon = %.3f/%.3f s, CG speedup = %.2fx\n",
            row.operator,
            row.virtual_coils,
            row.magnitude_nrmse_mean,
            row.magnitude_ssim_mean,
            row.setup_median_s,
            row.recon_median_s,
            row.cg_speedup_vs_32,
        )
    end
    return summaries
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && run_2d_coil_compression_sweep!()
