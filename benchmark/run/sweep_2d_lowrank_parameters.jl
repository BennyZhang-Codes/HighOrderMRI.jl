"""
Parameter sweep for the 2D `HighOrderLowRankOp` benchmark.

The explicit `HighOrderOp_Kernel` reconstruction is computed once and used as
an exact-model reference. The LowRank operator is then evaluated over a grid of

    L_rank × shared_basis_tol × rSVD seed × oversampling.

Default main experiment:

    L_rank            = 2, 4, 6, 8, 10, 15, 20
    shared_basis_tol   = 5e-2, 2e-2, 1e-2, 5e-3, 1e-3
    rSVD seeds         = 1234, 1235, 1236, 1237, 1238
    oversampling       = 5

This gives 175 LowRank reconstructions. Oversampling is fixed by default. A
supplementary oversampling scan can be requested by passing multiple values to
`HIGHORDER_SWEEP_OVERSAMPLINGS`.

Run from the repository root, for example:

    julia --project=. --threads=5 benchmark/run/sweep_2d_lowrank_parameters.jl

Useful overrides:

    HIGHORDER_BENCHMARK_GPUS=4,5,6,7 \\
    HIGHORDER_SWEEP_RANKS=2,4,6,8,10,15,20 \\
    HIGHORDER_SWEEP_TOLS=5e-2,2e-2,1e-2,5e-3,1e-3 \\
    HIGHORDER_SWEEP_SEEDS=1234,1235,1236,1237,1238 \\
    HIGHORDER_SWEEP_OVERSAMPLINGS=5 \\
    julia --project=. --threads=5 benchmark/run/sweep_2d_lowrank_parameters.jl

Supplementary oversampling experiment:

    HIGHORDER_SWEEP_RANKS=10 \\
    HIGHORDER_SWEEP_TOLS=1e-2 \\
    HIGHORDER_SWEEP_OVERSAMPLINGS=0,2,5,10 \\
    julia --project=. --threads=5 benchmark/run/sweep_2d_lowrank_parameters.jl

Timing excludes MAT-file loading, metric evaluation, CSV writing, and image
saving. It includes LowRank operator construction in `setup_s` and the complete
fixed-iteration reconstruction in `recon_s`.
"""

# Reuse the 2D run configuration, dataset preprocessing, exact Kernel
# constructor, and image helper from the single-configuration benchmark. Its
# `run_benchmark!()` entry point is guarded and is therefore not executed here.
include(joinpath(@__DIR__, "run_2d_kernel_vs_lowrank.jl"))

using Random


# -----------------------------------------------------------------------------
# Sweep configuration
# -----------------------------------------------------------------------------

function parse_env_int_list(name::AbstractString, default::AbstractVector{<:Integer})
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return Int.(default)
    values = parse.(Int, strip.(split(raw, ',')))
    isempty(values) && throw(ArgumentError("$name must contain at least one integer"))
    return values
end

function parse_env_float_list(name::AbstractString, default::AbstractVector{<:Real})
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return T.(default)
    values = T.(parse.(Float64, strip.(split(raw, ','))))
    isempty(values) && throw(ArgumentError("$name must contain at least one number"))
    return values
end

const SWEEP_RANKS = parse_env_int_list(
    "HIGHORDER_SWEEP_RANKS",
    [2, 4, 6, 8, 10, 15, 20],
)
const SWEEP_TOLS = parse_env_float_list(
    "HIGHORDER_SWEEP_TOLS",
    [5e-2, 2e-2, 1e-2, 5e-3, 1e-3],
)
const SWEEP_SEEDS = parse_env_int_list(
    "HIGHORDER_SWEEP_SEEDS",
    collect(1234:1238),
)
const SWEEP_OVERSAMPLINGS = parse_env_int_list(
    "HIGHORDER_SWEEP_OVERSAMPLINGS",
    [RSVD_OVERSAMPLE],
)

const SWEEP_WARMUP = parse_env_bool("HIGHORDER_SWEEP_WARMUP", true)
const SWEEP_RANDOMIZE_ORDER = parse_env_bool("HIGHORDER_SWEEP_RANDOMIZE_ORDER", true)
const SWEEP_ORDER_SEED = parse_env_int("HIGHORDER_SWEEP_ORDER_SEED", 2026)
const SWEEP_CONTINUE_ON_ERROR = parse_env_bool("HIGHORDER_SWEEP_CONTINUE_ON_ERROR", true)
const SWEEP_CHECKPOINT_EVERY = parse_env_int("HIGHORDER_SWEEP_CHECKPOINT_EVERY", 1)
const SWEEP_SAVE_BEST_IMAGES = parse_env_bool("HIGHORDER_SWEEP_SAVE_BEST_IMAGES", true)
const SWEEP_SHARED_RANK_MAX = parse_env_int(
    "HIGHORDER_SWEEP_SHARED_RANK_MAX",
    SHARED_RANK_MAX,
)
const SWEEP_OUTPUT_ROOT = get(
    ENV,
    "HIGHORDER_SWEEP_OUTPUT_DIR",
    OUTPUT_DIR,
)

all(>(0), SWEEP_RANKS) || throw(ArgumentError("All L_rank values must be positive"))
all(>=(zero(T)), SWEEP_TOLS) || throw(ArgumentError("All shared_basis_tol values must be non-negative"))
all(>=(0), SWEEP_OVERSAMPLINGS) || throw(ArgumentError("All oversampling values must be non-negative"))
length(SWEEP_SEEDS) >= 5 || @warn(
    "Fewer than five rSVD seeds were requested; at least five are recommended",
    seeds=SWEEP_SEEDS,
)
SWEEP_CHECKPOINT_EVERY >= 1 || throw(ArgumentError("SWEEP_CHECKPOINT_EVERY must be at least one"))


# -----------------------------------------------------------------------------
# Parameterized operator construction
# -----------------------------------------------------------------------------

function build_sweep_lowrank_operator(
    L_rank::Int,
    shared_basis_tol::T,
    rsvd_seed::Int,
    oversampling::Int,
)
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
        L_rank=L_rank,
        rsvd_seed=rsvd_seed,
        rsvd_chunk=RSVD_CHUNK,
        rsvd_oversample=oversampling,
        rsvd_finalize=:gram,
        rsvd_backend=:kernel,
        rsvd_distribution=distribution,
        shared_rank_max=SWEEP_SHARED_RANK_MAX,
        shared_basis_tol=shared_basis_tol,
        normal_distribution=normal_distribution,
        nfft_center_correction=NFFT_CENTER_CORRECTION,
        verbose=false,
    )
end


function error_string(err, bt)
    io = IOBuffer()
    showerror(io, CapturedException(err, bt))
    return replace(String(take!(io)), '\n' => " | ")
end


# -----------------------------------------------------------------------------
# Compilation warmup
# -----------------------------------------------------------------------------

"""
Compile the exact Kernel path and every distinct fused-rSVD sketch width.

The fused CUDA rSVD kernels specialize on `L_total = L_rank + oversampling`.
Without this warmup, the first configuration using a new `L_total` would include
CUDA compilation in its measured setup time.
"""
function warmup_sweep!()
    @info "Warmup: exact Kernel reconstruction"
    kernel_op = build_kernel_operator()
    try
        CUDA.device!(first(GPU_IDS))
        recon_HOOp(kernel_op, data, weight, rec_params)
    finally
        kernel_op = nothing
        GC.gc(true)
    end

    sketch_configs = unique([
        (L_rank=L_rank, oversampling=oversampling)
        for L_rank in SWEEP_RANKS
        for oversampling in SWEEP_OVERSAMPLINGS
    ])
    sort!(sketch_configs; by=c -> (c.L_rank + c.oversampling, c.L_rank, c.oversampling))

    first_config = true
    representative_tol = first(SWEEP_TOLS)
    representative_seed = first(SWEEP_SEEDS)

    for config in sketch_configs
        @info(
            "Warmup: LowRank sketch width",
            L_rank=config.L_rank,
            oversampling=config.oversampling,
            L_total=config.L_rank + config.oversampling,
        )

        op = build_sweep_lowrank_operator(
            config.L_rank,
            representative_tol,
            representative_seed,
            config.oversampling,
        )
        try
            # One full LowRank reconstruction is sufficient to compile the
            # solver and forward/adjoint paths. Remaining configurations only
            # need construction to compile their Val(L_total) rSVD kernels.
            if first_config
                recon_HOOp(op, data, weight, rec_params)
                first_config = false
            end
        finally
            release_benchmark_backend!(op)
            op = nothing
            GC.gc(true)
        end
    end

    return nothing
end


# -----------------------------------------------------------------------------
# One LowRank configuration
# -----------------------------------------------------------------------------

function benchmark_lowrank_configuration(
    config_id::Int,
    total_configs::Int,
    L_rank::Int,
    shared_basis_tol::T,
    rsvd_seed::Int,
    oversampling::Int,
    x_reference,
)
    op = nothing
    setup_s = missing
    recon_s = missing
    total_s = missing
    shared_rank = missing
    self_residual = missing
    complex_error = missing
    magnitude_error = missing
    scale_real = missing
    scale_imag = missing
    free_before = gpu_free_memory_mib(GPU_IDS)
    free_with_operator = missing
    status = "ok"
    error_message = ""
    x_host = nothing

    try
        @info(
            "LowRank parameter sweep",
            config_id,
            total_configs,
            L_rank,
            shared_basis_tol,
            rsvd_seed,
            oversampling,
        )

        op, setup_s = elapsed_seconds(
            () -> build_sweep_lowrank_operator(
                L_rank,
                shared_basis_tol,
                rsvd_seed,
                oversampling,
            ),
            GPU_IDS,
        )
        free_with_operator = gpu_free_memory_mib(GPU_IDS)
        # `gpu_free_memory_mib` visits every device and leaves the task on the
        # last one. Restore the LowRank operator's primary device before CG.
        CUDA.device!(first(GPU_IDS))

        x, recon_s = elapsed_seconds(
            () -> recon_HOOp(op, data, weight, rec_params),
            GPU_IDS,
        )
        x = require_reconstruction(x, "HighOrderLowRankOp parameter sweep")
        total_s = setup_s + recon_s

        self_residual = weighted_residual(op, x, data, weight)
        shared_rank = size(op.basis, 2)
        x_host = Array(x)

        scale = alignment_scale(x_host, x_reference)
        scale_real = real(scale)
        scale_imag = imag(scale)
        complex_error = aligned_relative_error(x_host, x_reference; scale=scale)
        magnitude_error = magnitude_nrmse(x_host, x_reference; scale=scale)
    catch err
        status = "error"
        error_message = error_string(err, catch_backtrace())
        @error(
            "LowRank sweep configuration failed",
            config_id,
            L_rank,
            shared_basis_tol,
            rsvd_seed,
            oversampling,
            error_message,
        )
        SWEEP_CONTINUE_ON_ERROR || rethrow()
    finally
        release_benchmark_backend!(op)
        op = nothing
        GC.gc(true)
    end

    row = (
        config_id=config_id,
        status=status,
        error=error_message,
        L_rank=L_rank,
        shared_basis_tol=shared_basis_tol,
        rsvd_seed=rsvd_seed,
        oversampling=oversampling,
        L_total=L_rank + oversampling,
        shared_rank=shared_rank,
        setup_s=setup_s,
        recon_s=recon_s,
        total_s=total_s,
        complex_error_vs_kernel=complex_error,
        magnitude_nrmse_vs_kernel=magnitude_error,
        alignment_scale_real=scale_real,
        alignment_scale_imag=scale_imag,
        lowrank_self_residual=self_residual,
        kernel_model_residual=missing,
        kernel_model_residual_aligned=missing,
        free_before_mib=free_before,
        free_with_operator_mib=free_with_operator,
        cg_iterations=CG_ITERATIONS,
        regularization=REGULARIZATION,
        shared_rank_max=SWEEP_SHARED_RANK_MAX,
        gpu_ids=join(GPU_IDS, ','),
        nfft_center_correction=NFFT_CENTER_CORRECTION,
        nfft_backend=string(AbstractNFFTs.active_backend()),
        julia_version=string(VERSION),
    )

    return row, x_host
end


# -----------------------------------------------------------------------------
# Summary utilities
# -----------------------------------------------------------------------------

function successful_values(rows, field::Symbol)
    return Float64[
        Float64(getproperty(row, field))
        for row in rows
        if row.status == "ok" && !ismissing(getproperty(row, field))
    ]
end

safe_mean(x) = isempty(x) ? missing : mean(x)
safe_std(x) = length(x) <= 1 ? missing : std(x)
safe_median(x) = isempty(x) ? missing : median(x)
safe_minimum(x) = isempty(x) ? missing : minimum(x)
safe_maximum(x) = isempty(x) ? missing : maximum(x)

function summarize_parameter_groups(rows::AbstractVector{<:NamedTuple})
    group_keys = unique([
        (
            L_rank=row.L_rank,
            shared_basis_tol=row.shared_basis_tol,
            oversampling=row.oversampling,
        )
        for row in rows
    ])
    sort!(group_keys; by=k -> (k.oversampling, k.L_rank, -Float64(k.shared_basis_tol)))

    summaries = NamedTuple[]
    for key in group_keys
        group = [
            row for row in rows
            if row.L_rank == key.L_rank &&
               row.shared_basis_tol == key.shared_basis_tol &&
               row.oversampling == key.oversampling
        ]

        setup = successful_values(group, :setup_s)
        recon = successful_values(group, :recon_s)
        total = successful_values(group, :total_s)
        shared_rank = successful_values(group, :shared_rank)
        complex_error = successful_values(group, :complex_error_vs_kernel)
        magnitude_error = successful_values(group, :magnitude_nrmse_vs_kernel)
        self_residual = successful_values(group, :lowrank_self_residual)
        kernel_residual = successful_values(group, :kernel_model_residual)
        kernel_residual_aligned = successful_values(group, :kernel_model_residual_aligned)

        push!(summaries, (
            L_rank=key.L_rank,
            shared_basis_tol=key.shared_basis_tol,
            oversampling=key.oversampling,
            L_total=key.L_rank + key.oversampling,
            requested_seeds=length(group),
            successful_seeds=count(row -> row.status == "ok", group),
            failed_seeds=count(row -> row.status != "ok", group),
            shared_rank_mean=safe_mean(shared_rank),
            shared_rank_std=safe_std(shared_rank),
            shared_rank_median=safe_median(shared_rank),
            shared_rank_min=safe_minimum(shared_rank),
            shared_rank_max_observed=safe_maximum(shared_rank),
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
            complex_error_mean=safe_mean(complex_error),
            complex_error_std=safe_std(complex_error),
            complex_error_median=safe_median(complex_error),
            complex_error_min=safe_minimum(complex_error),
            complex_error_max=safe_maximum(complex_error),
            magnitude_nrmse_mean=safe_mean(magnitude_error),
            magnitude_nrmse_std=safe_std(magnitude_error),
            magnitude_nrmse_median=safe_median(magnitude_error),
            magnitude_nrmse_min=safe_minimum(magnitude_error),
            magnitude_nrmse_max=safe_maximum(magnitude_error),
            lowrank_self_residual_mean=safe_mean(self_residual),
            lowrank_self_residual_std=safe_std(self_residual),
            kernel_model_residual_mean=safe_mean(kernel_residual),
            kernel_model_residual_std=safe_std(kernel_residual),
            kernel_model_residual_aligned_mean=safe_mean(kernel_residual_aligned),
            kernel_model_residual_aligned_std=safe_std(kernel_residual_aligned),
        ))
    end

    return summaries
end


# -----------------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------------

function run_parameter_sweep!()
    SWEEP_WARMUP && warmup_sweep!()

    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    data_name = splitext(basename(DATA_FILE))[1]
    run_dir = joinpath(
        SWEEP_OUTPUT_ROOT,
        "2d_lowrank_parameter_sweep_$(data_name)_$(timestamp)",
    )
    mkpath(run_dir)

    checkpoint_path = joinpath(run_dir, "runs_checkpoint.csv")
    runs_path = joinpath(run_dir, "runs.csv")
    summary_path = joinpath(run_dir, "summary_by_configuration.csv")
    kernel_path = joinpath(run_dir, "kernel_reference.csv")

    @info "Computing one exact Kernel reference reconstruction"
    kernel_run, x_reference = benchmark_kernel(1)
    write_csv(kernel_path, [kernel_run])

    configurations = [
        (
            L_rank=L_rank,
            shared_basis_tol=shared_basis_tol,
            rsvd_seed=rsvd_seed,
            oversampling=oversampling,
        )
        for oversampling in SWEEP_OVERSAMPLINGS
        for L_rank in SWEEP_RANKS
        for shared_basis_tol in SWEEP_TOLS
        for rsvd_seed in SWEEP_SEEDS
    ]

    if SWEEP_RANDOMIZE_ORDER
        Random.shuffle!(Random.Xoshiro(SWEEP_ORDER_SEED), configurations)
    end

    total_configs = length(configurations)
    @info(
        "Starting LowRank parameter sweep",
        total_configs,
        ranks=SWEEP_RANKS,
        tolerances=SWEEP_TOLS,
        seeds=SWEEP_SEEDS,
        oversamplings=SWEEP_OVERSAMPLINGS,
        randomized=SWEEP_RANDOMIZE_ORDER,
        order_seed=SWEEP_ORDER_SEED,
        run_dir,
    )

    rows = NamedTuple[]
    reconstructions = Vector{Any}(undef, total_configs)

    for (config_id, config) in enumerate(configurations)
        row, x_host = benchmark_lowrank_configuration(
            config_id,
            total_configs,
            config.L_rank,
            config.shared_basis_tol,
            config.rsvd_seed,
            config.oversampling,
            x_reference,
        )
        push!(rows, row)
        reconstructions[config_id] = x_host

        if config_id % SWEEP_CHECKPOINT_EVERY == 0 || config_id == total_configs
            write_csv(checkpoint_path, rows)
            @info "Sweep checkpoint written" config_id total_configs checkpoint_path
        end
    end

    # Evaluate every successful LowRank reconstruction under one persistent exact
    # Kernel operator. This is intentionally outside all timed regions and avoids
    # retaining a Kernel operator while constructing memory-intensive LowRank ops.
    @info "Evaluating completed reconstructions with the exact Kernel model"
    quality_kernel = build_kernel_operator()
    kernel_reference_residual = missing
    try
        CUDA.device!(first(GPU_IDS))
        kernel_reference_residual = weighted_residual(
            quality_kernel,
            x_reference,
            data,
            weight,
        )

        for i in eachindex(rows)
            row = rows[i]
            x_host = reconstructions[i]
            if row.status != "ok" || x_host === nothing
                continue
            end

            scale = ComplexF64(
                row.alignment_scale_real,
                row.alignment_scale_imag,
            )
            CUDA.device!(first(GPU_IDS))
            kernel_residual = weighted_residual(
                quality_kernel,
                x_host,
                data,
                weight,
            )
            kernel_residual_aligned = weighted_residual(
                quality_kernel,
                scale .* x_host,
                data,
                weight,
            )

            rows[i] = merge(row, (
                kernel_model_residual=kernel_residual,
                kernel_model_residual_aligned=kernel_residual_aligned,
            ))
        end
    finally
        quality_kernel = nothing
        GC.gc(true)
    end

    summaries = summarize_parameter_groups(rows)
    write_csv(runs_path, rows)
    write_csv(summary_path, summaries)

    successful_indices = findall(row -> row.status == "ok", rows)
    best_index = if isempty(successful_indices)
        nothing
    else
        successful_indices[argmin([
            rows[i].magnitude_nrmse_vs_kernel for i in successful_indices
        ])]
    end

    image_paths = String[]
    if SWEEP_SAVE_BEST_IMAGES && best_index !== nothing
        try
            best_row = rows[best_index]
            best_x = reconstructions[best_index]
            best_scale = ComplexF64(
                best_row.alignment_scale_real,
                best_row.alignment_scale_imag,
            )

            kernel_image = joinpath(run_dir, "kernel_reference.png")
            best_image = joinpath(
                run_dir,
                @sprintf(
                    "best_lowrank_L%d_tol%.0e_seed%d_os%d.png",
                    best_row.L_rank,
                    best_row.shared_basis_tol,
                    best_row.rsvd_seed,
                    best_row.oversampling,
                ),
            )
            difference_image = joinpath(run_dir, "best_lowrank_magnitude_difference.png")

            save_reconstruction_png(
                kernel_image,
                x_reference;
                vmaxp=IMAGE_VMAX_PERCENTILE,
            )
            save_reconstruction_png(
                best_image,
                best_scale .* best_x;
                vmaxp=IMAGE_VMAX_PERCENTILE,
            )
            save_reconstruction_png(
                difference_image,
                abs.(best_scale .* best_x) .- abs.(x_reference);
                vmaxp=IMAGE_VMAX_PERCENTILE,
            )
            append!(image_paths, (kernel_image, best_image, difference_image))
        catch err
            @warn(
                "Could not save sweep images; CSV results are complete",
                exception=(err, catch_backtrace()),
            )
        end
    end

    println("\n2D LowRank parameter sweep complete")
    println("Run directory: $run_dir")
    println("Kernel reference residual: $kernel_reference_residual")
    println("Successful configurations: $(length(successful_indices)) / $total_configs")

    if best_index !== nothing
        best = rows[best_index]
        @printf(
            "Best individual run by magnitude NRMSE: L=%d, tol=%.1e, seed=%d, oversampling=%d, shared rank=%d, NRMSE=%.3e, total=%.3f s\n",
            best.L_rank,
            best.shared_basis_tol,
            best.rsvd_seed,
            best.oversampling,
            best.shared_rank,
            best.magnitude_nrmse_vs_kernel,
            best.total_s,
        )
    end

    valid_summaries = [s for s in summaries if !ismissing(s.magnitude_nrmse_mean)]
    sort!(valid_summaries; by=s -> s.magnitude_nrmse_mean)
    println("\nTop configurations by mean magnitude NRMSE")
    for summary in Iterators.take(valid_summaries, min(10, length(valid_summaries)))
        @printf(
            "L=%2d, tol=%7.1e, os=%2d: NRMSE=%.3e ± %.1e, shared rank=%.1f, total=%.3f s\n",
            summary.L_rank,
            summary.shared_basis_tol,
            summary.oversampling,
            summary.magnitude_nrmse_mean,
            ismissing(summary.magnitude_nrmse_std) ? NaN : summary.magnitude_nrmse_std,
            summary.shared_rank_mean,
            summary.total_mean_s,
        )
    end

    @info(
        "Parameter sweep outputs written",
        run_dir,
        kernel_path,
        checkpoint_path,
        runs_path,
        summary_path,
        image_paths,
    )

    return (
        run_dir=run_dir,
        kernel_path=kernel_path,
        checkpoint_path=checkpoint_path,
        runs_path=runs_path,
        summary_path=summary_path,
        image_paths=image_paths,
        kernel_reference_residual=kernel_reference_residual,
        best_index=best_index,
        rows=rows,
        summaries=summaries,
    )
end


if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_parameter_sweep!()
end
