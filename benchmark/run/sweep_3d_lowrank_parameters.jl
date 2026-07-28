"""
Quality and first-run timing sweep for the 3D `HighOrderLowRankOp`.

The saved `HighOrderOp_Kernel` reconstruction is loaded once as the reference;
this script never constructs or runs `HighOrderOp_Kernel`.

The default experiment consists of:

* one untimed full warmup at `L_rank=2`, `shared_basis_tol=1e-2`;
* a timed rank sweep at `shared_basis_tol=1e-2` for
  `L_rank = [2, 4, 6, 8, 10, 15, 20, 25]`;
* a timed tolerance sweep at `L_rank=6` for
  `shared_basis_tol = [5e-2, 2e-2, 1e-2, 5e-3]`.

The `(L_rank=15, shared_basis_tol=1e-2)` result is shared by both sweeps, so
there are eleven unique timed configurations. Every successful configuration
exports magnitude/phase NIfTI volumes, a magnitude mosaic, and a magnitude
absolute-difference mosaic/NIfTI. All mosaics use the 99.9th percentile of the
Kernel-reference magnitude as one fixed `vmax`.

Run from the repository root:

    julia --project=. --threads=5 benchmark/run/sweep_3d_lowrank_parameters.jl

Useful overrides:

    HIGHORDER_3D_GPUS=2,3,4,5,6
    HIGHORDER_3D_RUN_DIR=/path/to/existing/or/new/run_directory
    HIGHORDER_3D_WARMUP=false
    HIGHORDER_3D_RETRY_ERRORS=true
    HIGHORDER_3D_RERUN_CONFIGS=L06_tol5e-2_seed1234
    HIGHORDER_3D_VERBOSE=true

MAT-file I/O, preprocessing, metric evaluation, checkpointing, plotting, and
NIfTI export are outside the timed regions. `operator_setup_s` measures only
`HighOrderLowRankOp(...)`. `cg_recon_s` measures the complete
`recon_HOOp(...)` call, including its solver and lazy normal-backend setup.
"""

using AbstractNFFTs
using CUDA
using CSV
using Dates
using HighOrderMRI
using LinearAlgebra
using MAT
using MRIGeometry
using NIfTI
using NonuniformFFTs
using Printf
using PyPlot
using RegularizedLeastSquares
using Serialization
using Statistics

include(joinpath(@__DIR__, "..", "common", "benchmark_utils.jl"))


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

function parse_env_float(name::AbstractString, default::Real)
    return parse(Float64, get(ENV, name, string(default)))
end

function parse_env_string_list(name::AbstractString, default::AbstractString)
    raw = strip(get(ENV, name, default))
    isempty(raw) && return String[]
    return String[strip(value) for value in split(raw, ',') if !isempty(strip(value))]
end

const T = Float32
const DEFAULT_DATA_FILE = "/home/jyzhang/Desktop/Julia_pkg/HighOrderMRI_Benchmark_Data/" *
    "Data_meas_MID00109_FID08979_Spiral_1p0x1p0x1p0_220x220x160_3x2_int16_3p8ms_tr50_fa15_te10p0.mat"
const DEFAULT_REFERENCE_FILE = "/home/jyzhang/Desktop/Julia_pkg/HighOrderMRI_Benchmark_Data/" *
    "20260728_HighOrderOp_Kernel.mat"

const DATA_FILE = get(ENV, "HIGHORDER_3D_DATA", DEFAULT_DATA_FILE)
const REFERENCE_FILE = get(ENV, "HIGHORDER_3D_REFERENCE", DEFAULT_REFERENCE_FILE)
const RESULTS_ROOT = normpath(joinpath(@__DIR__, "..", "results"))
const RUN_TIMESTAMP = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
const DEFAULT_RUN_DIR = joinpath(RESULTS_ROOT, "3d_lowrank_sweep_$(RUN_TIMESTAMP)")
const RUN_DIR = abspath(get(ENV, "HIGHORDER_3D_RUN_DIR", DEFAULT_RUN_DIR))

const GPU_IDS = parse_env_gpus("HIGHORDER_3D_GPUS", [2, 3, 4, 5, 6])
const WARMUP = parse_env_bool("HIGHORDER_3D_WARMUP", true)
const RETRY_ERRORS = parse_env_bool("HIGHORDER_3D_RETRY_ERRORS", false)
const RERUN_CONFIG_IDS = parse_env_string_list("HIGHORDER_3D_RERUN_CONFIGS", "")
const VERBOSE = parse_env_bool("HIGHORDER_3D_VERBOSE", false)

const RECON_TERMS = "1111"
const CG_ITERATIONS = parse_env_int("HIGHORDER_3D_CG_ITERATIONS", 20)
const REGULARIZATION = T(parse_env_float("HIGHORDER_3D_REGULARIZATION", 1e-9))
const RSVD_SEED = parse_env_int("HIGHORDER_3D_RSVD_SEED", 1234)
const RSVD_CHUNK = parse_env_int("HIGHORDER_3D_RSVD_CHUNK", 4096)
const RSVD_OVERSAMPLE = parse_env_int("HIGHORDER_3D_RSVD_OVERSAMPLE", 5)
const SHARED_RANK_MAX = parse_env_int("HIGHORDER_3D_SHARED_RANK_MAX", 32)
const NFFT_CENTER_CORRECTION = parse_env_bool(
    "HIGHORDER_3D_NFFT_CENTER_CORRECTION",
    true,
)

const WARMUP_RANK = 2
const RANK_SWEEP_TOL = 1e-2
const TOL_SWEEP_RANK = 6
const RANKS = [2, 4, 6, 8, 10, 15, 20, 25]
const BASIS_TOLS = [5e-2, 2e-2, 1e-2, 5e-3]

const REFERENCE_VMAX_PERCENTILE = 99.9
const DIFFERENCE_SCALE = parse_env_float("HIGHORDER_3D_DIFFERENCE_SCALE", 10)
const MOSAIC_DIM = 3
const MOSAIC_NROW = 10
const MOSAIC_NCOL = 16
const PNG_DPI = parse_env_int("HIGHORDER_3D_PNG_DPI", 300)

isempty(GPU_IDS) && throw(ArgumentError("HIGHORDER_3D_GPUS must contain at least one GPU"))
all(>(0), RANKS) || throw(ArgumentError("All rank-sweep values must be positive"))
all(>(0), BASIS_TOLS) || throw(ArgumentError("All basis tolerances must be positive"))
SHARED_RANK_MAX > 0 || throw(ArgumentError("SHARED_RANK_MAX must be positive"))
CG_ITERATIONS > 0 || throw(ArgumentError("CG_ITERATIONS must be positive"))
RSVD_OVERSAMPLE >= 0 || throw(ArgumentError("RSVD_OVERSAMPLE must be non-negative"))
PNG_DPI > 0 || throw(ArgumentError("PNG_DPI must be positive"))
DIFFERENCE_SCALE > 0 || throw(ArgumentError("DIFFERENCE_SCALE must be positive"))


# -----------------------------------------------------------------------------
# Configuration matrix
# -----------------------------------------------------------------------------

function tolerance_tag(tol::Real)
    text = @sprintf("%.0e", Float64(tol))
    text = replace(text, "e-0" => "e-", "e+0" => "e", "e+" => "e")
    return text
end

configuration_id(rank::Integer, tol::Real) =
    @sprintf("L%02d_tol%s_seed%d", rank, tolerance_tag(tol), RSVD_SEED)

function difference_scale_tag(scale::Real)
    isinteger(scale) && return string(Int(scale))
    return replace(@sprintf("%.3g", Float64(scale)), "." => "p")
end

function experiment_configurations()
    configs = NamedTuple[]

    for rank in RANKS
        push!(configs, (
            config_id=configuration_id(rank, RANK_SWEEP_TOL),
            L_rank=rank,
            shared_basis_tol=Float64(RANK_SWEEP_TOL),
            in_rank_sweep=true,
            in_tolerance_sweep=(rank == TOL_SWEEP_RANK),
        ))
    end

    for tol in BASIS_TOLS
        isapprox(tol, RANK_SWEEP_TOL; rtol=0, atol=eps(Float64)) && continue
        push!(configs, (
            config_id=configuration_id(TOL_SWEEP_RANK, tol),
            L_rank=TOL_SWEEP_RANK,
            shared_basis_tol=Float64(tol),
            in_rank_sweep=false,
            in_tolerance_sweep=true,
        ))
    end

    ids = getproperty.(configs, :config_id)
    length(unique(ids)) == length(ids) || error("Experiment configuration IDs are not unique")
    return configs
end

function refresh_sweep_membership(row)
    row_rank = Int(row.L_rank)
    row_tol = Float64(row.shared_basis_tol)
    in_rank_sweep =
        row_rank in RANKS &&
        isapprox(row_tol, RANK_SWEEP_TOL; rtol=0, atol=eps(Float64))
    in_tolerance_sweep =
        row_rank == TOL_SWEEP_RANK &&
        any(tol -> isapprox(row_tol, tol; rtol=0, atol=eps(Float64)), BASIS_TOLS)

    return merge(row, (
        in_rank_sweep=in_rank_sweep,
        in_tolerance_sweep=in_tolerance_sweep,
    ))
end


# -----------------------------------------------------------------------------
# MAT loading and 3D preprocessing
# -----------------------------------------------------------------------------

function error_string(err, bt)
    io = IOBuffer()
    showerror(io, CapturedException(err, bt))
    return replace(String(take!(io)), '\n' => " | ")
end

mat_scalar_int(value) = Int(value isa AbstractArray ? only(value) : value)

function read_mat_fields(path::AbstractString, names)
    file = matopen(path)
    try
        return Dict{String,Any}(name => read(file, name) for name in names)
    finally
        close(file)
    end
end

function load_reference(path::AbstractString)
    file = matopen(path)
    try
        return read(file, "x")
    finally
        close(file)
    end
end

function load_3d_dataset(path::AbstractString)
    names = (
        "b0", "mask", "csm", "kdata", "datatime", "kspha", "weight",
        "nX", "nY", "nZ", "nSlice", "nRep", "nEcho",
        "nSample_adc", "nDynamic", "nCha", "geo", "gridding",
    )
    raw = read_mat_fields(path, names)

    nX = mat_scalar_int(raw["nX"])
    nY = mat_scalar_int(raw["nY"])
    nZ = mat_scalar_int(raw["nZ"])
    nSlice = mat_scalar_int(raw["nSlice"])
    nRep = mat_scalar_int(raw["nRep"])
    nEcho = mat_scalar_int(raw["nEcho"])
    nSample = mat_scalar_int(raw["nSample_adc"])
    nDynamic = mat_scalar_int(raw["nDynamic"])
    nCha = mat_scalar_int(raw["nCha"])

    nEcho == 1 || throw(ArgumentError(
        "The benchmark currently selects echo 1 and expects nEcho=1; got $nEcho",
    ))

    geo = Geometry(raw["geo"]; T=T)
    grid = Grid(raw["gridding"]; T=T)
    expected_size = (nX, nY, nZ)
    grid.matrixSize == expected_size || throw(DimensionMismatch(
        "Grid size $(grid.matrixSize) does not match MAT metadata $expected_size",
    ))
    Tuple(Int.(geo.MatrixSize)) == expected_size || throw(DimensionMismatch(
        "Geometry size $(Tuple(geo.MatrixSize)) does not match MAT metadata $expected_size",
    ))

    csm = Complex{T}.(raw["csm"])
    fieldmap = T.(raw["b0"])
    mask = Bool.(raw["mask"])
    expected_csm_size = (nX, nY, nZ, nCha)
    size(csm) == expected_csm_size || throw(DimensionMismatch(
        "Expected csm size $expected_csm_size; got $(size(csm))",
    ))
    size(fieldmap) == expected_size || throw(DimensionMismatch(
        "Expected fieldmap size $expected_size; got $(size(fieldmap))",
    ))
    size(mask) == expected_size || throw(DimensionMismatch(
        "Expected mask size $expected_size; got $(size(mask))",
    ))

    kspha = reshape(raw["kspha"], nEcho, nSample, nDynamic, 16)
    kspha_model = T.(permutedims(@view(kspha[1, :, :, :]), (3, 1, 2)))
    times = T.(reshape(raw["datatime"], nSample, nDynamic))

    # First-order trajectories are stored in DCS. The CSM, field map, mask,
    # Grid, and saved Kernel reconstruction use RPS image coordinates.
    R_DCS2RPS = T.(transpose(geo.R_RPS_DCS))
    k_dcs = @view kspha_model[2:4, :, :]
    k_rps = reshape(
        R_DCS2RPS * reshape(k_dcs, 3, :),
        3,
        nSample,
        nDynamic,
    )
    k_dcs .= k_rps

    kdata = raw["kdata"]
    data_host = reshape(
        @view(kdata[1, 1, 1, :, :, :]),
        nSample * nDynamic,
        nCha,
    )
    weights = reshape(raw["weight"], nEcho, nSample, nDynamic)
    weight_host = vec(@view weights[1, :, :])

    CUDA.device!(first(GPU_IDS))
    data = CuArray(Complex{T}.(data_host))
    weight = CuArray(Complex{T}.(weight_host))

    rec_params = Dict{Symbol,Any}(
        :reconSize => expected_size,
        :reg => L2Regularization(REGULARIZATION),
        :iterations => CG_ITERATIONS,
        :solver => CGNR,
    )

    @info(
        "3D benchmark dataset ready",
        matrix_size=expected_size,
        nSample,
        nDynamic,
        nCha,
        nVox=sum(mask),
        gpu_ids=GPU_IDS,
    )

    dataset = (
        grid=grid,
        geo=geo,
        csm=csm,
        fieldmap=fieldmap,
        mask=mask,
        kspha=kspha_model,
        times=times,
        data=data,
        weight=weight,
        rec_params=rec_params,
        matrix_size=expected_size,
        nSample=nSample,
        nDynamic=nDynamic,
        nCha=nCha,
    )

    # Drop the original MAT arrays and the host views that retain the 2.6 GB
    # input container before the first large LowRank operator is constructed.
    raw = nothing
    kdata = nothing
    data_host = nothing
    weights = nothing
    weight_host = nothing
    kspha = nothing
    GC.gc(true)

    return dataset
end


# -----------------------------------------------------------------------------
# Metrics and artifact export
# -----------------------------------------------------------------------------

function mask_bounding_box(mask::AbstractArray{Bool,3})
    lower = collect(size(mask))
    upper = ones(Int, 3)
    found = false

    for index in CartesianIndices(mask)
        mask[index] || continue
        found = true
        for dim in 1:3
            coordinate = index[dim]
            lower[dim] = min(lower[dim], coordinate)
            upper[dim] = max(upper[dim], coordinate)
        end
    end

    found || throw(ArgumentError("Reconstruction mask is empty"))
    return ntuple(dim -> lower[dim]:upper[dim], 3)
end

function save_magnitude_png(
    path::AbstractString,
    magnitude::AbstractArray{<:Real,3},
    reference_vmax::Real,
)
    mkpath(dirname(path))
    fig = plt_image(
        magnitude;
        dim=MOSAIC_DIM,
        nRow=MOSAIC_NROW,
        nCol=MOSAIC_NCOL,
        vmin=0,
        vmax=reference_vmax,
    )
    try
        fig.savefig(
            path;
            dpi=PNG_DPI,
            transparent=false,
            bbox_inches="tight",
            pad_inches=0.0,
        )
    finally
        PyPlot.close(fig)
    end
    return path
end

function export_reference!(
    x_reference,
    geo,
    reference_vmax::Real,
)
    reference_dir = joinpath(RUN_DIR, "reference")
    mkpath(reference_dir)
    name = "HighOrderOp_Kernel_reference"
    png_path = joinpath(reference_dir, "$(name)_mag.png")
    mag_path = joinpath(reference_dir, "$(name)_mag.nii.gz")
    pha_path = joinpath(reference_dir, "$(name)_pha.nii.gz")

    if !(isfile(png_path) && isfile(mag_path) && isfile(pha_path))
        save_magnitude_png(png_path, abs.(x_reference), reference_vmax)
        export_nifti(x_reference, geo, reference_dir, name)
    end

    return (
        reference_png=relpath(png_path, RUN_DIR),
        reference_mag_nifti=relpath(mag_path, RUN_DIR),
        reference_phase_nifti=relpath(pha_path, RUN_DIR),
    )
end

function export_configuration_artifacts!(
    config_id::AbstractString,
    x,
    x_reference,
    geo,
    reference_vmax::Real,
)
    reconstruction_dir = joinpath(RUN_DIR, "reconstruction")
    difference_dir = joinpath(RUN_DIR, "difference")
    mkpath(reconstruction_dir)
    mkpath(difference_dir)

    reconstruction_png = joinpath(reconstruction_dir, "$(config_id)_mag.png")
    reconstruction_mag = joinpath(reconstruction_dir, "$(config_id)_mag.nii.gz")
    reconstruction_phase = joinpath(reconstruction_dir, "$(config_id)_pha.nii.gz")

    difference = DIFFERENCE_SCALE .* abs.(abs.(x) .- abs.(x_reference))
    difference_name =
        "$(config_id)_magnitude_absdiff_x$(difference_scale_tag(DIFFERENCE_SCALE))"
    difference_png = joinpath(difference_dir, "$(difference_name).png")
    difference_nifti = joinpath(difference_dir, "$(difference_name).nii.gz")

    save_magnitude_png(reconstruction_png, abs.(x), reference_vmax)
    export_nifti(x, geo, reconstruction_dir, config_id)

    save_magnitude_png(difference_png, difference, reference_vmax)
    export_nifti(difference, geo, difference_dir, difference_name)

    all(isfile, (
        reconstruction_png,
        reconstruction_mag,
        reconstruction_phase,
        difference_png,
        difference_nifti,
    )) || error("One or more artifacts were not created for $config_id")

    return (
        reconstruction_png=relpath(reconstruction_png, RUN_DIR),
        reconstruction_mag_nifti=relpath(reconstruction_mag, RUN_DIR),
        reconstruction_phase_nifti=relpath(reconstruction_phase, RUN_DIR),
        difference_png=relpath(difference_png, RUN_DIR),
        difference_nifti=relpath(difference_nifti, RUN_DIR),
    )
end


# -----------------------------------------------------------------------------
# Operator construction, warmup, and one timed configuration
# -----------------------------------------------------------------------------

function build_lowrank_operator(dataset, rank::Int, tol::Real)
    distribution = length(GPU_IDS) > 1 ? :voxel : :single
    normal_distribution = length(GPU_IDS) > 1 ? :channel : :single

    return HighOrderLowRankOp(
        dataset.grid,
        copy(dataset.kspha),
        dataset.times;
        fieldmap=dataset.fieldmap,
        csm=dataset.csm,
        mask=dataset.mask,
        recon_terms=RECON_TERMS,
        arrayType=CuArray,
        gpus=GPU_IDS,
        L_rank=rank,
        rsvd_seed=RSVD_SEED,
        rsvd_chunk=RSVD_CHUNK,
        rsvd_oversample=RSVD_OVERSAMPLE,
        rsvd_finalize=:gram,
        rsvd_backend=:kernel,
        rsvd_distribution=distribution,
        shared_rank_max=SHARED_RANK_MAX,
        shared_basis_tol=T(tol),
        normal_distribution=normal_distribution,
        nfft_center_correction=NFFT_CENTER_CORRECTION,
        verbose=VERBOSE,
    )
end

function warmup!(dataset)
    @info(
        "3D full warmup",
        L_rank=WARMUP_RANK,
        shared_basis_tol=RANK_SWEEP_TOL,
        shared_rank_max=SHARED_RANK_MAX,
    )
    op = nothing
    try
        op = build_lowrank_operator(dataset, WARMUP_RANK, RANK_SWEEP_TOL)
        CUDA.device!(first(GPU_IDS))
        x = recon_HOOp(op, dataset.data, dataset.weight, dataset.rec_params)
        require_reconstruction(x, "3D LowRank warmup")
    finally
        release_benchmark_backend!(op)
        op = nothing
        GC.gc(true)
    end
    return nothing
end

function benchmark_configuration(
    config,
    config_index::Int,
    total_configs::Int,
    dataset,
    x_reference,
    reference_vmax::Real,
    roi,
)
    op = nothing
    operator_setup_s = missing
    cg_recon_s = missing
    total_s = missing
    shared_rank = missing
    free_before_mib = missing
    free_with_operator_mib = missing
    free_after_recon_mib = missing

    complex_nrmse_masked = missing
    magnitude_nrmse_masked = missing
    magnitude_ssim_roi = missing
    complex_nrmse_full = missing
    magnitude_nrmse_full = missing
    magnitude_ssim_full = missing

    reconstruction_png = ""
    reconstruction_mag_nifti = ""
    reconstruction_phase_nifti = ""
    difference_png = ""
    difference_nifti = ""

    status = "ok"
    error_message = ""

    try
        @info(
            "3D LowRank configuration",
            config_index,
            total_configs,
            config_id=config.config_id,
            L_rank=config.L_rank,
            shared_basis_tol=config.shared_basis_tol,
            shared_rank_max=SHARED_RANK_MAX,
        )

        GC.gc(true)
        free_before_mib = gpu_free_memory_mib(GPU_IDS)

        op, operator_setup_s = elapsed_seconds(
            () -> build_lowrank_operator(
                dataset,
                config.L_rank,
                config.shared_basis_tol,
            ),
            GPU_IDS,
        )
        shared_rank = size(op.basis, 2)
        free_with_operator_mib = gpu_free_memory_mib(GPU_IDS)
        CUDA.device!(first(GPU_IDS))

        x, cg_recon_s = elapsed_seconds(
            () -> recon_HOOp(op, dataset.data, dataset.weight, dataset.rec_params),
            GPU_IDS,
        )
        x = require_reconstruction(x, "3D HighOrderLowRankOp")
        total_s = operator_setup_s + cg_recon_s
        free_after_recon_mib = gpu_free_memory_mib(GPU_IDS)

        x_host = reshape(Array(x), dataset.matrix_size)
        size(x_host) == size(x_reference) || throw(DimensionMismatch(
            "Reconstruction $(size(x_host)) and reference $(size(x_reference)) differ",
        ))
        all(isfinite, x_host) || throw(DomainError(
            nothing,
            "Reconstruction contains non-finite values",
        ))

        mask_values = vec(dataset.mask)
        x_masked = vec(x_host)[mask_values]
        reference_masked = vec(x_reference)[mask_values]

        complex_nrmse_masked = raw_complex_nrmse(x_masked, reference_masked)
        magnitude_nrmse_masked = magnitude_nrmse(x_masked, reference_masked)
        magnitude_ssim_roi = magnitude_ssim(
            view(x_host, roi...),
            view(x_reference, roi...),
        )

        complex_nrmse_full = raw_complex_nrmse(x_host, x_reference)
        magnitude_nrmse_full = magnitude_nrmse(x_host, x_reference)
        magnitude_ssim_full = magnitude_ssim(x_host, x_reference)

        artifacts = export_configuration_artifacts!(
            config.config_id,
            x_host,
            x_reference,
            dataset.geo,
            reference_vmax,
        )
        reconstruction_png = artifacts.reconstruction_png
        reconstruction_mag_nifti = artifacts.reconstruction_mag_nifti
        reconstruction_phase_nifti = artifacts.reconstruction_phase_nifti
        difference_png = artifacts.difference_png
        difference_nifti = artifacts.difference_nifti
    catch err
        status = "error"
        error_message = error_string(err, catch_backtrace())
        @error(
            "3D LowRank configuration failed",
            config_id=config.config_id,
            L_rank=config.L_rank,
            shared_basis_tol=config.shared_basis_tol,
            error_message,
        )
    finally
        release_benchmark_backend!(op)
        op = nothing
        GC.gc(true)
    end

    return (
        config_id=config.config_id,
        status=status,
        error=error_message,
        in_rank_sweep=config.in_rank_sweep,
        in_tolerance_sweep=config.in_tolerance_sweep,
        L_rank=config.L_rank,
        shared_basis_tol=config.shared_basis_tol,
        shared_rank_max=SHARED_RANK_MAX,
        shared_rank=shared_rank,
        rsvd_seed=RSVD_SEED,
        rsvd_chunk=RSVD_CHUNK,
        rsvd_oversample=RSVD_OVERSAMPLE,
        rsvd_finalize="gram",
        rsvd_backend="kernel",
        rsvd_distribution=length(GPU_IDS) > 1 ? "voxel" : "single",
        normal_distribution=length(GPU_IDS) > 1 ? "channel" : "single",
        operator_setup_s=operator_setup_s,
        cg_recon_s=cg_recon_s,
        total_s=total_s,
        complex_nrmse_masked=complex_nrmse_masked,
        magnitude_nrmse_masked=magnitude_nrmse_masked,
        magnitude_ssim_roi=magnitude_ssim_roi,
        complex_nrmse_full=complex_nrmse_full,
        magnitude_nrmse_full=magnitude_nrmse_full,
        magnitude_ssim_full=magnitude_ssim_full,
        reference_vmax=reference_vmax,
        reference_vmax_percentile=REFERENCE_VMAX_PERCENTILE,
        reconstruction_png=reconstruction_png,
        reconstruction_mag_nifti=reconstruction_mag_nifti,
        reconstruction_phase_nifti=reconstruction_phase_nifti,
        difference_png=difference_png,
        difference_nifti=difference_nifti,
        difference_scale=DIFFERENCE_SCALE,
        free_before_mib=free_before_mib,
        free_with_operator_mib=free_with_operator_mib,
        free_after_recon_mib=free_after_recon_mib,
        cg_iterations=CG_ITERATIONS,
        regularization=REGULARIZATION,
        recon_terms=RECON_TERMS,
        gpu_ids=join(GPU_IDS, ','),
        nfft_center_correction=NFFT_CENTER_CORRECTION,
        nfft_backend=string(AbstractNFFTs.active_backend()),
        timing_kind="first_run_after_rank2_warmup",
        julia_version=string(VERSION),
    )
end


# -----------------------------------------------------------------------------
# Checkpointing and tabular outputs
# -----------------------------------------------------------------------------

function checkpoint_path()
    return joinpath(RUN_DIR, "runs_checkpoint.jls")
end

function migrate_difference_artifact(row, geo, reference_vmax::Real)
    if row.status != "ok"
        return hasproperty(row, :difference_scale) ?
            row : merge(row, (difference_scale=DIFFERENCE_SCALE,))
    end
    previous_scale = hasproperty(row, :difference_scale) ?
        Float64(row.difference_scale) : 1.0
    isapprox(previous_scale, DIFFERENCE_SCALE; rtol=0, atol=eps(Float64)) &&
        return row

    previous_path = joinpath(RUN_DIR, String(row.difference_nifti))
    if !isfile(previous_path)
        @warn(
            "Cannot migrate an older difference artifact; configuration will be rerun",
            config_id=row.config_id,
            previous_path,
        )
        return nothing
    end

    previous_scale > 0 || error(
        "Invalid previous difference scale $previous_scale for $(row.config_id)",
    )
    previous_difference = Array(NIfTI.niread(previous_path).raw)
    difference = (DIFFERENCE_SCALE / previous_scale) .* previous_difference

    difference_dir = joinpath(RUN_DIR, "difference")
    difference_name =
        "$(row.config_id)_magnitude_absdiff_x$(difference_scale_tag(DIFFERENCE_SCALE))"
    difference_png = joinpath(difference_dir, "$(difference_name).png")
    difference_nifti = joinpath(difference_dir, "$(difference_name).nii.gz")
    save_magnitude_png(difference_png, difference, reference_vmax)
    export_nifti(difference, geo, difference_dir, difference_name)

    @info(
        "Migrated checkpointed difference artifact without rerunning reconstruction",
        config_id=row.config_id,
        previous_scale,
        new_scale=DIFFERENCE_SCALE,
    )
    return merge(row, (
        difference_png=relpath(difference_png, RUN_DIR),
        difference_nifti=relpath(difference_nifti, RUN_DIR),
        difference_scale=DIFFERENCE_SCALE,
    ))
end

function load_checkpoint(geo, reference_vmax::Real)
    path = checkpoint_path()
    rows = NamedTuple[]

    if isfile(path)
        serialized_rows = open(deserialize, path)
        serialized_rows isa AbstractVector ||
            error("Invalid checkpoint contents at $path")
        rows = NamedTuple[serialized_rows...]
    else
        # A process can be interrupted after the human-readable CSV was
        # written but before the serialized checkpoint was committed. Recover
        # those completed rows rather than repeating an expensive 3D
        # reconstruction.
        csv_path = joinpath(RUN_DIR, "runs_checkpoint.csv")
        isfile(csv_path) || return rows
        rows = NamedTuple[
            NamedTuple(row)
            for row in CSV.File(csv_path; missingstring=["missing", ""])
        ]
        @info "Recovered checkpoint rows from CSV" csv_path count=length(rows)
    end

    migrated = NamedTuple[]
    for row in rows
        updated = migrate_difference_artifact(row, geo, reference_vmax)
        updated === nothing || push!(migrated, updated)
    end
    return migrated
end

function save_serialized_checkpoint(rows)
    path = checkpoint_path()
    temporary = "$(path).tmp"
    open(temporary, "w") do io
        serialize(io, rows)
    end
    mv(temporary, path; force=true)
    return path
end

function write_result_tables(rows)
    # Commit the resume-critical checkpoint first. Derived CSV tables can then
    # be regenerated without losing a completed reconstruction.
    save_serialized_checkpoint(rows)
    write_csv(joinpath(RUN_DIR, "runs_checkpoint.csv"), rows)
    write_csv(joinpath(RUN_DIR, "runs.csv"), rows)

    rank_rows = NamedTuple[row for row in rows if row.in_rank_sweep]
    sort!(rank_rows; by=row -> row.L_rank)
    tolerance_rows = NamedTuple[row for row in rows if row.in_tolerance_sweep]
    sort!(tolerance_rows; by=row -> row.shared_basis_tol, rev=true)
    write_csv(joinpath(RUN_DIR, "rank_sweep.csv"), rank_rows)
    write_csv(joinpath(RUN_DIR, "basis_tol_sweep.csv"), tolerance_rows)
    return nothing
end

function write_configuration_table(dataset, reference_vmax::Real, reference_artifacts)
    row = (
        data_file=DATA_FILE,
        reference_file=REFERENCE_FILE,
        run_directory=RUN_DIR,
        matrix_size=join(dataset.matrix_size, 'x'),
        nSample=dataset.nSample,
        nDynamic=dataset.nDynamic,
        nCha=dataset.nCha,
        nVox=sum(dataset.mask),
        gpu_ids=join(GPU_IDS, ','),
        rank_sweep=join(RANKS, ','),
        rank_sweep_tolerance=RANK_SWEEP_TOL,
        tolerance_sweep=join(BASIS_TOLS, ','),
        tolerance_sweep_rank=TOL_SWEEP_RANK,
        shared_rank_max=SHARED_RANK_MAX,
        rsvd_seed=RSVD_SEED,
        rsvd_chunk=RSVD_CHUNK,
        rsvd_oversample=RSVD_OVERSAMPLE,
        cg_iterations=CG_ITERATIONS,
        regularization=REGULARIZATION,
        recon_terms=RECON_TERMS,
        warmup_rank=WARMUP_RANK,
        warmup_tolerance=RANK_SWEEP_TOL,
        reference_vmax=reference_vmax,
        reference_vmax_percentile=REFERENCE_VMAX_PERCENTILE,
        difference_scale=DIFFERENCE_SCALE,
        reference_png=reference_artifacts.reference_png,
        reference_mag_nifti=reference_artifacts.reference_mag_nifti,
        reference_phase_nifti=reference_artifacts.reference_phase_nifti,
        excluded_tolerance=1e-3,
        excluded_tolerance_reason="shared rank exceeded 128 at dynamic 463",
        timing_kind="first_run_after_rank2_warmup",
        julia_version=string(VERSION),
    )
    write_csv(joinpath(RUN_DIR, "configuration.csv"), [row])
    return row
end


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

function run_3d_sweep!()
    isfile(DATA_FILE) || error(
        "3D benchmark data file was not found: $DATA_FILE\n" *
        "Set HIGHORDER_3D_DATA to override it.",
    )
    isfile(REFERENCE_FILE) || error(
        "Kernel reference file was not found: $REFERENCE_FILE\n" *
        "Set HIGHORDER_3D_REFERENCE to override it.",
    )

    mkpath(RUN_DIR)
    AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())

    @info(
        "Loading 3D benchmark inputs",
        DATA_FILE,
        REFERENCE_FILE,
        RUN_DIR,
        GPU_IDS,
    )
    dataset = load_3d_dataset(DATA_FILE)
    x_reference = reshape(
        Complex{T}.(load_reference(REFERENCE_FILE)),
        dataset.matrix_size,
    )
    all(isfinite, x_reference) || throw(DomainError(
        nothing,
        "Kernel reference contains non-finite values",
    ))

    reference_vmax = quantile(
        vec(abs.(x_reference)),
        REFERENCE_VMAX_PERCENTILE / 100,
    )
    isfinite(reference_vmax) && reference_vmax > 0 || throw(DomainError(
        reference_vmax,
        "Reference magnitude percentile must be finite and positive",
    ))
    roi = mask_bounding_box(dataset.mask)

    reference_artifacts = export_reference!(
        x_reference,
        dataset.geo,
        reference_vmax,
    )
    write_configuration_table(dataset, reference_vmax, reference_artifacts)

    WARMUP && warmup!(dataset)

    configs = experiment_configurations()
    requested_config_ids = Set(config.config_id for config in configs)
    unknown_rerun_ids = setdiff(Set(RERUN_CONFIG_IDS), requested_config_ids)
    isempty(unknown_rerun_ids) || throw(ArgumentError(
        "HIGHORDER_3D_RERUN_CONFIGS contains unknown configuration IDs: " *
        join(sort!(collect(unknown_rerun_ids)), ", "),
    ))
    configs_to_execute = isempty(RERUN_CONFIG_IDS) ?
        configs :
        [config for config in configs if config.config_id in Set(RERUN_CONFIG_IDS)]

    rows = NamedTuple[
        refresh_sweep_membership(row)
        for row in load_checkpoint(dataset.geo, reference_vmax)
    ]
    if RETRY_ERRORS
        rows = NamedTuple[row for row in rows if row.status == "ok"]
    end
    if !isempty(RERUN_CONFIG_IDS)
        rerun_ids = Set(RERUN_CONFIG_IDS)
        rows = NamedTuple[row for row in rows if !(String(row.config_id) in rerun_ids)]
        @info "Selected completed configurations for replacement" config_ids=RERUN_CONFIG_IDS
    end
    completed_ids = Set(String(row.config_id) for row in rows)

    @info(
        "Starting 3D LowRank sweep",
        total_unique_configurations=length(configs),
        configurations_selected_for_this_process=length(configs_to_execute),
        already_completed=length(completed_ids),
        warmup=WARMUP,
        shared_rank_max=SHARED_RANK_MAX,
        reference_vmax,
    )

    for (config_index, config) in enumerate(configs_to_execute)
        if config.config_id in completed_ids
            @info "Skipping checkpointed 3D configuration" config_id=config.config_id
            continue
        end

        row = benchmark_configuration(
            config,
            config_index,
            length(configs_to_execute),
            dataset,
            x_reference,
            reference_vmax,
            roi,
        )
        push!(rows, row)
        push!(completed_ids, config.config_id)
        write_result_tables(rows)
    end

    successful = count(row -> row.status == "ok", rows)
    failed = count(row -> row.status != "ok", rows)
    println("\n3D LowRank sweep complete")
    println("Run directory: $RUN_DIR")
    println("Successful configurations: $successful")
    println("Failed configurations: $failed")
    println("Reference vmax (99.9th percentile): $reference_vmax")
    println("Rank results: $(joinpath(RUN_DIR, "rank_sweep.csv"))")
    println("Tolerance results: $(joinpath(RUN_DIR, "basis_tol_sweep.csv"))")
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && run_3d_sweep!()
