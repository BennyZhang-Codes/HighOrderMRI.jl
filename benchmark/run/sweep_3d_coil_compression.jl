"""
Benchmark 3D reconstruction quality and speed versus global coil-compression
rank. The saved `HighOrderOp_Kernel` image remains the fixed reference and is
never reconstructed by this script.

The default comparison sweeps 4:4:32 channels with ten reconstructions per
channel count. One shared, full-data SVD coil basis is truncated into nested
virtual-coil bases and applied identically to k-space data and CSMs. Only
`HighOrderLowRankOp` is tested; the saved Kernel image is used only as the
fixed quality reference.

Run from the HighOrderMRI-benchmark repository root:

    julia --project=benchmark --threads=3 benchmark/run/sweep_3d_coil_compression.jl

Useful overrides:

    HIGHORDER_3D_GPUS=2,3,4,5,6
    HIGHORDER_3D_RUN_DIR=/path/to/run_directory
    HIGHORDER_3D_COIL_COUNTS=4,8,12,16,20,24,28,32
    HIGHORDER_3D_COIL_REPEATS=10
    HIGHORDER_3D_COIL_L_RANK=6
    HIGHORDER_3D_COIL_BASIS_TOL=1e-2
    HIGHORDER_3D_WARMUP=false
"""

using Dates

const CC_RESULTS_ROOT = normpath(joinpath(@__DIR__, "..", "results"))
const CC_TIMESTAMP = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
get!(ENV, "HIGHORDER_3D_RUN_DIR") do
    joinpath(CC_RESULTS_ROOT, "3d_coil_compression_$(CC_TIMESTAMP)")
end
get!(ENV, "HIGHORDER_3D_GPUS", "2,3,4,5,6")

# Reuse the established 3D data preparation, fixed-reference metrics,
# artifact export, LowRank construction, GPU timing, and rank-2 warmup.
include(joinpath(@__DIR__, "sweep_3d_lowrank_parameters.jl"))


# -----------------------------------------------------------------------------
# Coil-compression configuration
# -----------------------------------------------------------------------------

function parse_env_int_list(name::AbstractString, default::AbstractString)
    raw = strip(get(ENV, name, default))
    isempty(raw) && return Int[]
    return parse.(Int, strip.(split(raw, ',')))
end

const CC_COIL_COUNTS = parse_env_int_list(
    "HIGHORDER_3D_COIL_COUNTS",
    "4,8,12,16,20,24,28,32",
)
const CC_REPEATS = parse_env_int("HIGHORDER_3D_COIL_REPEATS", 10)
const CC_L_RANK = parse_env_int("HIGHORDER_3D_COIL_L_RANK", 6)
const CC_BASIS_TOL = parse_env_float("HIGHORDER_3D_COIL_BASIS_TOL", 1e-2)
const CC_CHECKPOINT_FILE = joinpath(RUN_DIR, "coil_compression_checkpoint.jls")

isempty(CC_COIL_COUNTS) && throw(ArgumentError(
    "HIGHORDER_3D_COIL_COUNTS must contain at least one channel count",
))
length(unique(CC_COIL_COUNTS)) == length(CC_COIL_COUNTS) || throw(ArgumentError(
    "HIGHORDER_3D_COIL_COUNTS contains duplicate values: $CC_COIL_COUNTS",
))
all(>(0), CC_COIL_COUNTS) || throw(ArgumentError(
    "Every requested channel count must be positive",
))
CC_REPEATS > 0 || throw(ArgumentError(
    "HIGHORDER_3D_COIL_REPEATS must be positive",
))
CC_L_RANK > 0 || throw(ArgumentError("HIGHORDER_3D_COIL_L_RANK must be positive"))
CC_BASIS_TOL > 0 || throw(ArgumentError(
    "HIGHORDER_3D_COIL_BASIS_TOL must be positive",
))

coil_artifact_id(ncoil::Integer) = @sprintf(
    "coil%02d_L%02d_tol%s_seed%d",
    ncoil,
    CC_L_RANK,
    tolerance_tag(CC_BASIS_TOL),
    RSVD_SEED,
)

coil_config_id(ncoil::Integer, repeat_id::Integer) = @sprintf(
    "%s_rep%02d",
    coil_artifact_id(ncoil),
    repeat_id,
)

function coil_config(ncoil::Integer, repeat_id::Integer)
    return (
        config_id=coil_config_id(ncoil, repeat_id),
        artifact_id=coil_artifact_id(ncoil),
        repeat=Int(repeat_id),
        L_rank=CC_L_RANK,
        shared_basis_tol=Float64(CC_BASIS_TOL),
        in_rank_sweep=false,
        in_tolerance_sweep=false,
    )
end


# -----------------------------------------------------------------------------
# Shared SVD fit and nested transforms
# -----------------------------------------------------------------------------

function elapsed_host_seconds(f::F) where {F}
    GC.gc(true)
    t0 = time_ns()
    value = f()
    return value, (time_ns() - t0) * 1e-9
end

function truncate_coil_transform(
    full_transform::CoilCompressionTransform{T},
    n_virtual_coils::Integer,
) where {T<:AbstractFloat}
    input_coils = size(full_transform, 1)
    1 <= n_virtual_coils <= input_coils || throw(ArgumentError(
        "n_virtual_coils=$n_virtual_coils must be in 1:$input_coils",
    ))

    singular_values = copy(full_transform.singular_values)
    total_energy = sum(abs2, singular_values)
    retained_energy = sum(abs2, @view(singular_values[1:n_virtual_coils])) /
        total_energy
    compression_matrix = Matrix{Complex{T}}(
        @view full_transform.compression_matrix[:, 1:n_virtual_coils]
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

function write_coil_spectrum(full_transform)
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
    write_csv(joinpath(RUN_DIR, "coil_spectrum.csv"), rows)
    return nothing
end

function prepare_coil_dataset(dataset, data_host, full_transform, ncoil::Int)
    ncoil < dataset.nCha || throw(ArgumentError(
        "prepare_coil_dataset is only for compressed ranks below $(dataset.nCha)",
    ))
    transform = truncate_coil_transform(full_transform, ncoil)

    data_cc, data_compression_s = elapsed_host_seconds(
        () -> apply_coil_compression(data_host, transform; coil_dim=2),
    )
    csm_cc, csm_compression_s = elapsed_host_seconds(
        () -> apply_coil_compression(
            dataset.csm,
            transform;
            coil_dim=ndims(dataset.csm),
        ),
    )

    CUDA.device!(first(GPU_IDS))
    data_device, data_to_gpu_s = elapsed_seconds(
        () -> CuArray(data_cc),
        GPU_IDS,
    )
    prepared = merge(dataset, (
        data=data_device,
        csm=csm_cc,
        nCha=ncoil,
    ))

    data_cc = nothing
    csm_cc = nothing
    GC.gc(true)
    return prepared, transform, data_compression_s, csm_compression_s, data_to_gpu_s
end

function release_unused_gpu_memory!()
    for gpu_id in GPU_IDS
        CUDA.device!(gpu_id)
        CUDA.reclaim()
    end
    CUDA.device!(first(GPU_IDS))
    return nothing
end

"""
Replace only the coil-dependent part of an existing LowRank operator.

The global temporal/spatial phase basis, NFFT plan, trajectory, mask, and
workspaces are unchanged. The lazily created channel-distributed normal
backend is released before changing the CSM and is rebuilt by the next
`recon_HOOp` call for the new channel count.
"""
function replace_operator_coils!(op::HighOrderLowRankOp{T}, dataset) where {T}
    release_highorder_normal_backend!(op)
    ncoil = dataset.nCha
    mask_flat = vec(dataset.mask)
    csm_masked = Matrix{Complex{T}}(
        reshape(dataset.csm, :, ncoil)[mask_flat, :]
    )

    CUDA.device!(first(GPU_IDS))
    op.csm = CuArray(csm_masked)
    op.nCha = ncoil
    op.nrow = op.nSam * op.nDyn * ncoil
    op.Mv = similar(op.q, Complex{T}, 0)
    op.Mtu = similar(op.q, Complex{T}, 0)
    return op
end

function warmup_reconstruction!(op, dataset)
    warmup_params = copy(dataset.rec_params)
    warmup_params[:iterations] = 1
    @info(
        "Warming up coil-dependent normal backend and CG path",
        nCha=dataset.nCha,
        iterations=1,
    )
    CUDA.device!(first(GPU_IDS))
    x = recon_HOOp(op, dataset.data, dataset.weight, warmup_params)
    require_reconstruction(x, "3D coil-compression reconstruction warmup")
    x = nothing
    GC.gc(true)
    release_unused_gpu_memory!()
    return nothing
end


# -----------------------------------------------------------------------------
# Timed reconstruction with one shared phase-basis operator
# -----------------------------------------------------------------------------

function benchmark_coil_configuration(
    config,
    config_index::Int,
    total_configs::Int,
    dataset,
    op,
    common_operator_setup_s::Real,
    common_shared_rank::Int,
    reconfigure_operator::Bool,
    x_reference,
    reference_vmax::Real,
    roi,
    save_artifacts::Bool,
)
    coil_reconfiguration_s = 0.0
    cg_recon_s = missing
    total_s = missing
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
            "3D coil-compression configuration",
            config_index,
            total_configs,
            config_id=config.config_id,
            repeat=config.repeat,
            nCha=dataset.nCha,
            shared_phase_rank=common_shared_rank,
        )
        GC.gc(true)
        free_before_mib = gpu_free_memory_mib(GPU_IDS)

        if reconfigure_operator
            _, coil_reconfiguration_s = elapsed_seconds(
                () -> replace_operator_coils!(op, dataset),
                GPU_IDS,
            )
        end
        free_with_operator_mib = gpu_free_memory_mib(GPU_IDS)

        CUDA.device!(first(GPU_IDS))
        x, cg_recon_s = elapsed_seconds(
            () -> recon_HOOp(op, dataset.data, dataset.weight, dataset.rec_params),
            GPU_IDS,
        )
        x = require_reconstruction(x, "3D coil-compression LowRank reconstruction")
        total_s = benchmark_stage_total(coil_reconfiguration_s, cg_recon_s)
        free_after_recon_mib = gpu_free_memory_mib(GPU_IDS)

        x_host = reshape(Array(x), dataset.matrix_size)
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

        if save_artifacts
            artifacts = export_configuration_artifacts!(
                config.artifact_id,
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
        end
    catch err
        status = "error"
        error_message = error_string(err, catch_backtrace())
        @error(
            "3D coil-compression configuration failed",
            config_id=config.config_id,
            nCha=dataset.nCha,
            error_message,
        )
    finally
        release_highorder_normal_backend!(op)
        GC.gc(true)
    end

    return (
        config_id=config.config_id,
        artifact_id=config.artifact_id,
        repeat=config.repeat,
        status=status,
        error=error_message,
        in_rank_sweep=false,
        in_tolerance_sweep=false,
        L_rank=CC_L_RANK,
        shared_basis_tol=CC_BASIS_TOL,
        shared_rank_max=SHARED_RANK_MAX,
        shared_rank=common_shared_rank,
        rsvd_seed=RSVD_SEED,
        rsvd_chunk=RSVD_CHUNK,
        rsvd_oversample=RSVD_OVERSAMPLE,
        rsvd_finalize="gram",
        rsvd_backend="kernel",
        rsvd_distribution=length(GPU_IDS) > 1 ? "voxel" : "single",
        normal_distribution=length(GPU_IDS) > 1 ? "channel" : "single",
        operator_setup_s=common_operator_setup_s,
        coil_reconfiguration_s=coil_reconfiguration_s,
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
        timing_kind="reused_phase_operator",
        warmup=WARMUP,
        julia_version=string(VERSION),
    )
end


# -----------------------------------------------------------------------------
# Checkpoint and comparison table
# -----------------------------------------------------------------------------

function load_coil_checkpoint()
    isfile(CC_CHECKPOINT_FILE) || return NamedTuple[]
    rows = deserialize(CC_CHECKPOINT_FILE)
    rows isa AbstractVector || error(
        "Invalid coil-compression checkpoint: $CC_CHECKPOINT_FILE",
    )
    return NamedTuple[row for row in rows]
end

function save_coil_results(rows::Vector{NamedTuple})
    temporary = CC_CHECKPOINT_FILE * ".tmp"
    open(temporary, "w") do io
        serialize(io, rows)
    end
    mv(temporary, CC_CHECKPOINT_FILE; force=true)
    write_csv(joinpath(RUN_DIR, "coil_compression_runs.csv"), rows)
    return nothing
end

cc_safe_mean(values) = mean(Float64.(values))
cc_safe_median(values) = median(Float64.(values))
cc_safe_std(values) = length(values) > 1 ? std(Float64.(values)) : 0.0

function write_coil_comparison(rows)
    successful = [row for row in rows if row.status == "ok"]
    isempty(successful) && return NamedTuple[]

    comparison = NamedTuple[]
    for ncoil in sort(unique(Int[row.virtual_coils for row in successful]))
        group = [row for row in successful if row.virtual_coils == ncoil]
        first_row = first(group)
        magnitude_nrmse = getproperty.(group, :magnitude_nrmse_masked)
        magnitude_ssim = getproperty.(group, :magnitude_ssim_roi)
        complex_nrmse = getproperty.(group, :complex_nrmse_masked)
        coil_reconfiguration = getproperty.(group, :coil_reconfiguration_s)
        cg_recon = getproperty.(group, :cg_recon_s)
        total = getproperty.(group, :total_s)
        total_with_compression = getproperty.(group, :total_with_compression_s)
        accounted_total = getproperty.(group, :accounted_total_s)

        push!(comparison, (
            status="ok",
            input_coils=first_row.input_coils,
            virtual_coils=ncoil,
            compression_ratio=first_row.compression_ratio,
            retained_energy=first_row.retained_energy,
            discarded_energy=first_row.discarded_energy,
            shared_rank=first_row.shared_rank,
            successful_repeats=length(group),
            requested_repeats=CC_REPEATS,
            magnitude_nrmse_mean=cc_safe_mean(magnitude_nrmse),
            magnitude_nrmse_std=cc_safe_std(magnitude_nrmse),
            magnitude_ssim_mean=cc_safe_mean(magnitude_ssim),
            magnitude_ssim_std=cc_safe_std(magnitude_ssim),
            raw_complex_nrmse_mean=cc_safe_mean(complex_nrmse),
            raw_complex_nrmse_std=cc_safe_std(complex_nrmse),
            compression_apply_s=first_row.compression_apply_s,
            data_to_gpu_s=first_row.data_to_gpu_s,
            common_operator_setup_s=first_row.operator_setup_s,
            coil_reconfiguration_median_s=cc_safe_median(coil_reconfiguration),
            coil_reconfiguration_mean_s=cc_safe_mean(coil_reconfiguration),
            cg_recon_median_s=cc_safe_median(cg_recon),
            cg_recon_mean_s=cc_safe_mean(cg_recon),
            cg_recon_std_s=cc_safe_std(cg_recon),
            total_median_s=cc_safe_median(total),
            total_mean_s=cc_safe_mean(total),
            total_std_s=cc_safe_std(total),
            total_with_compression_median_s=
                cc_safe_median(total_with_compression),
            total_with_compression_mean_s=
                cc_safe_mean(total_with_compression),
            total_with_compression_std_s=
                cc_safe_std(total_with_compression),
            accounted_total_median_s=cc_safe_median(accounted_total),
            timing_kind="reused_phase_operator_with_separately_measured_common_setup",
        ))
    end

    baseline_index = findfirst(
        row -> row.virtual_coils == row.input_coils,
        comparison,
    )
    isnothing(baseline_index) && error(
        "No successful uncompressed 32-channel LowRank baseline is available",
    )
    baseline = comparison[baseline_index]
    comparison = [merge(row, (
        cg_speedup_vs_32=baseline.cg_recon_median_s / row.cg_recon_median_s,
        total_speedup_vs_32=baseline.total_median_s / row.total_median_s,
        total_with_compression_speedup_vs_32=
            baseline.total_with_compression_median_s /
            row.total_with_compression_median_s,
        accounted_total_speedup_vs_32=
            baseline.accounted_total_median_s / row.accounted_total_median_s,
    )) for row in comparison]
    write_csv(joinpath(RUN_DIR, "coil_compression_comparison.csv"), comparison)
    return comparison
end


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

function run_3d_coil_compression_sweep!()
    isfile(DATA_FILE) || error("3D benchmark data file was not found: $DATA_FILE")
    isfile(REFERENCE_FILE) || error("Kernel reference file was not found: $REFERENCE_FILE")
    mkpath(RUN_DIR)
    AbstractNFFTs.set_active_backend!(NFFT.backend())
    validation_s = validate_3d_benchmark_run!(; extra=(
        experiment="3d_coil_compression", coils=CC_COIL_COUNTS,
        repeats=CC_REPEATS, local_rank=CC_L_RANK, shared_basis_tol=CC_BASIS_TOL))
    @info "Run contract validated before artifact export" validation_s

    @info(
        "Loading 3D coil-compression benchmark",
        DATA_FILE,
        REFERENCE_FILE,
        RUN_DIR,
        GPU_IDS,
        requested_coils=CC_COIL_COUNTS,
    )
    dataset = load_3d_dataset(DATA_FILE)
    input_coils = dataset.nCha
    input_coils in CC_COIL_COUNTS || throw(ArgumentError(
        "HIGHORDER_3D_COIL_COUNTS must include the uncompressed " *
        "$input_coils-channel baseline",
    ))
    all(count -> count <= input_coils, CC_COIL_COUNTS) || throw(ArgumentError(
        "Requested coil count exceeds the input count $input_coils",
    ))

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
    roi = mask_bounding_box(dataset.mask)
    reference_artifacts = export_reference!(
        x_reference,
        dataset.geo,
        reference_vmax,
    )

    @info "Copying original k-space data to host for one shared SVD fit"
    data_host, data_device_to_host_s = elapsed_host_seconds(() -> Array(dataset.data))
    full_transform, coil_fit_s = elapsed_host_seconds(
        () -> fit_coil_compression(
            data_host;
            coil_dim=2,
            n_virtual_coils=input_coils,
        ),
    )
    write_coil_spectrum(full_transform)
    orthogonality_error = norm(
        full_transform.compression_matrix' * full_transform.compression_matrix -
        Matrix{Complex{T}}(I, input_coils, input_coils),
    ) / sqrt(input_coils)

    @info(
        "Building one shared LowRank phase-basis operator",
        L_rank=CC_L_RANK,
        shared_basis_tol=CC_BASIS_TOL,
        note="This setup is reused for every coil count",
    )
    op, common_operator_setup_s = elapsed_seconds(
        () -> build_lowrank_operator(dataset, CC_L_RANK, CC_BASIS_TOL),
        GPU_IDS,
    )
    try
    common_shared_rank = size(op.basis, 2)

    configuration_row = (
        data_file=DATA_FILE,
        reference_file=REFERENCE_FILE,
        run_directory=RUN_DIR,
        reference_unchanged=true,
        matrix_size=join(dataset.matrix_size, 'x'),
        nSample=dataset.nSample,
        nDynamic=dataset.nDynamic,
        input_coils=input_coils,
        coil_sweep=join(CC_COIL_COUNTS, ','),
        repeats=CC_REPEATS,
        coil_compression_method="global_full_data_SVD_identity_noise_covariance",
        calibration_observations=size(data_host, 1),
        coil_fit_s=coil_fit_s,
        data_device_to_host_s=data_device_to_host_s,
        transform_orthogonality_error=orthogonality_error,
        L_rank=CC_L_RANK,
        shared_basis_tol=CC_BASIS_TOL,
        shared_rank_max=SHARED_RANK_MAX,
        common_shared_rank=common_shared_rank,
        common_operator_setup_s=common_operator_setup_s,
        operator_reused_across_coil_counts=true,
        gpu_ids=join(GPU_IDS, ','),
        cg_iterations=CG_ITERATIONS,
        regularization=REGULARIZATION,
        rsvd_seed=RSVD_SEED,
        rsvd_chunk=RSVD_CHUNK,
        rsvd_oversample=RSVD_OVERSAMPLE,
        reference_vmax=reference_vmax,
        reference_png=reference_artifacts.reference_png,
        reference_mag_nifti=reference_artifacts.reference_mag_nifti,
        reference_phase_nifti=reference_artifacts.reference_phase_nifti,
        warmup=WARMUP,
        warmup_kind=WARMUP ? "one_iteration_32_channel_CG_using_shared_operator" : "disabled",
        julia_version=string(VERSION),
    )
    write_csv(joinpath(RUN_DIR, "configuration.csv"), [configuration_row])

    WARMUP && warmup_reconstruction!(op, dataset)

    rows = load_coil_checkpoint()
    isempty(rows) || all(row -> hasproperty(row, :repeat), rows) || error(
        "The checkpoint predates the repeated 3D coil-compression benchmark. " *
        "Use a new HIGHORDER_3D_RUN_DIR instead of mixing CSV schemas.",
    )
    completed_ids = Set(
        String(row.config_id) for row in rows if row.status == "ok"
    )
    execution_coil_counts = [
        input_coils;
        filter(!=(input_coils), CC_COIL_COUNTS)
    ]
    total_configs = length(CC_COIL_COUNTS) * CC_REPEATS

    for ncoil in execution_coil_counts
        pending_repeats = [
            repeat_id for repeat_id in 1:CC_REPEATS
            if !(coil_config_id(ncoil, repeat_id) in completed_ids)
        ]
        if isempty(pending_repeats)
            @info "Skipping checkpointed coil count" ncoil repeats=CC_REPEATS
        else
            transform = ncoil == input_coils ? full_transform :
                truncate_coil_transform(full_transform, ncoil)
            data_compression_s = 0.0
            csm_compression_s = 0.0
            data_to_gpu_s = 0.0
            prepared = dataset

            if ncoil < input_coils
                prepared, transform, data_compression_s, csm_compression_s, data_to_gpu_s =
                    prepare_coil_dataset(dataset, data_host, full_transform, ncoil)
            end
            compression_apply_s = data_compression_s + csm_compression_s
            artifacts_saved = any(
                row -> row.status == "ok" &&
                    row.virtual_coils == ncoil &&
                    !isempty(String(row.reconstruction_png)),
                rows,
            )

            coil_position = findfirst(==(ncoil), CC_COIL_COUNTS)
            for repeat_id in pending_repeats
                config = coil_config(ncoil, repeat_id)
                config_index = (coil_position - 1) * CC_REPEATS + repeat_id
                filter!(row -> String(row.config_id) != config.config_id, rows)

                row = benchmark_coil_configuration(
                    config,
                    config_index,
                    total_configs,
                    prepared,
                    op,
                    common_operator_setup_s,
                    common_shared_rank,
                    op.nCha != ncoil,
                    x_reference,
                    reference_vmax,
                    roi,
                    !artifacts_saved,
                )
                total_with_compression_s = row.status == "ok" ?
                    benchmark_stage_total(compression_apply_s, data_to_gpu_s, row.total_s) : missing
                row = merge(row, (
                    input_coils=input_coils,
                    virtual_coils=ncoil,
                    compression_ratio=input_coils / ncoil,
                    coil_basis=ncoil == input_coils ?
                        "physical_channels" : "global_svd",
                    retained_energy=Float64(transform.retained_energy),
                    discarded_energy=1 - Float64(transform.retained_energy),
                    coil_fit_s=coil_fit_s,
                    data_compression_s=data_compression_s,
                    csm_compression_s=csm_compression_s,
                    compression_apply_s=compression_apply_s,
                    data_to_gpu_s=data_to_gpu_s,
                    total_with_compression_s=total_with_compression_s,
                    accounted_total_s=benchmark_stage_total(data_device_to_host_s,
                        coil_fit_s, common_operator_setup_s, total_with_compression_s),
                ))
                push!(rows, row)
                if row.status == "ok"
                    push!(completed_ids, config.config_id)
                    artifacts_saved = artifacts_saved ||
                        !isempty(row.reconstruction_png)
                end
                save_coil_results(rows)
            end

            prepared = nothing
            GC.gc(true)
            release_unused_gpu_memory!()
        end

        # The loader initially places the full data matrix on the primary GPU.
        # Once the 32-channel baseline is complete or skipped, keep only a tiny
        # placeholder so compressed configurations measure their own data use.
        if ncoil == input_coils
            CUDA.device!(first(GPU_IDS))
            dataset = merge(dataset, (data=CuArray(Complex{T}[]),))
            GC.gc(true)
            release_unused_gpu_memory!()
        end
    end

    comparison = write_coil_comparison(rows)

    successful = count(row -> row.status == "ok", rows)
    failed = count(row -> row.status != "ok", rows)
    println("\n3D coil-compression sweep complete")
    println("Run directory: $RUN_DIR")
    println("Successful configurations: $successful")
    println("Failed configurations: $failed")
    println("Coil fit time: $coil_fit_s s")
    println("Shared operator setup time: $common_operator_setup_s s")
    println("Results: $(joinpath(RUN_DIR, "coil_compression_comparison.csv"))")
    return rows
    finally
        close(op)
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && run_3d_coil_compression_sweep!()
