"""
Visualize the outputs of `sweep_2d_lowrank_parameters.jl` with PyPlot.jl.

The script reads

    summary_by_configuration.csv
    runs.csv

from one sweep output directory and saves, for every oversampling value:

    sweep_overview_os<os>.png/.pdf
    seed_stability_os<os>.png/.pdf
    pareto_tradeoff_os<os>.png/.pdf

Usage
-----

From the repository root:

    julia --project=. benchmark/analysis/plot_2d_lowrank_sweep.jl \
        benchmark/results/2d_lowrank_parameter_sweep_<dataset>_<timestamp>

If no directory is supplied, the newest directory beginning with
`2d_lowrank_parameter_sweep_` under `benchmark/results` is selected.

Environment variables
---------------------

    HIGHORDER_SWEEP_RUN_DIR       Explicit sweep directory.
    HIGHORDER_SWEEP_RESULTS_ROOT  Root used for automatic newest-run discovery.
    HIGHORDER_SWEEP_FIG_DPI       PNG DPI; default 300.
    HIGHORDER_SWEEP_SHOW_FIGURES  true/false; default false.
"""

using CSV
using PyPlot
using Printf
using Statistics


# -----------------------------------------------------------------------------
# Configuration and input discovery
# -----------------------------------------------------------------------------

parse_env_bool(name::AbstractString, default::Bool) = begin
    raw = lowercase(strip(get(ENV, name, string(default))))
    raw in ("true", "1", "yes", "on") && return true
    raw in ("false", "0", "no", "off") && return false
    throw(ArgumentError("$name must be true/false, 1/0, yes/no, or on/off; got $raw"))
end

const FIG_DPI = parse(Int, get(ENV, "HIGHORDER_SWEEP_FIG_DPI", "300"))
const SHOW_FIGURES = parse_env_bool("HIGHORDER_SWEEP_SHOW_FIGURES", false)
const DEFAULT_RESULTS_ROOT = get(
    ENV,
    "HIGHORDER_SWEEP_RESULTS_ROOT",
    normpath(joinpath(@__DIR__, "..", "results")),
)

function newest_sweep_directory(root::AbstractString)
    isdir(root) || throw(ArgumentError("Sweep results root does not exist: $root"))

    candidates = filter(readdir(root; join=true)) do path
        isdir(path) && startswith(basename(path), "2d_lowrank_parameter_sweep_")
    end

    isempty(candidates) && throw(ArgumentError(
        "No directory beginning with 2d_lowrank_parameter_sweep_ was found under $root",
    ))

    return candidates[argmax(mtime.(candidates))]
end

function resolve_run_directory()
    if !isempty(ARGS)
        return abspath(ARGS[1])
    end

    from_env = strip(get(ENV, "HIGHORDER_SWEEP_RUN_DIR", ""))
    !isempty(from_env) && return abspath(from_env)

    return newest_sweep_directory(DEFAULT_RESULTS_ROOT)
end


# -----------------------------------------------------------------------------
# Table utilities
# -----------------------------------------------------------------------------

function read_csv_rows(path::AbstractString)
    isfile(path) || throw(ArgumentError("CSV file not found: $path"))
    return collect(CSV.File(path; missingstring=["missing", ""]));
end

valid_number(x) = !ismissing(x) && x isa Number && isfinite(Float64(x))

function unique_sorted(rows, field::Symbol; rev::Bool=false)
    values = unique([
        getproperty(row, field)
        for row in rows
        if !ismissing(getproperty(row, field))
    ])
    sort!(values; rev=rev)
    return values
end

function matching_summary_row(rows, L_rank, tol, oversampling)
    matches = filter(rows) do row
        row.L_rank == L_rank &&
        isapprox(Float64(row.shared_basis_tol), Float64(tol); rtol=1e-6, atol=0.0) &&
        row.oversampling == oversampling
    end

    isempty(matches) && return nothing
    length(matches) == 1 || @warn(
        "Multiple summary rows matched one parameter combination; using the first",
        L_rank,
        tol,
        oversampling,
        count=length(matches),
    )
    return first(matches)
end

function parameter_matrix(
    rows,
    ranks,
    tolerances,
    oversampling,
    field::Symbol,
)
    matrix = fill(NaN, length(tolerances), length(ranks))

    for (iy, tol) in enumerate(tolerances), (ix, rank) in enumerate(ranks)
        row = matching_summary_row(rows, rank, tol, oversampling)
        row === nothing && continue
        value = getproperty(row, field)
        valid_number(value) || continue
        matrix[iy, ix] = Float64(value)
    end

    return matrix
end

function finite_values(matrix)
    return matrix[isfinite.(matrix)]
end

function tolerance_label(tol)
    return @sprintf("%.0e", Float64(tol))
end

function save_figure(fig, basepath::AbstractString)
    png_path = basepath * ".png"
    pdf_path = basepath * ".pdf"

    fig.savefig(
        png_path;
        dpi=FIG_DPI,
        bbox_inches="tight",
        pad_inches=0.04,
        transparent=false,
    )
    fig.savefig(
        pdf_path;
        bbox_inches="tight",
        pad_inches=0.04,
        transparent=false,
    )

    SHOW_FIGURES || close(fig)
    return (png_path, pdf_path)
end


# -----------------------------------------------------------------------------
# Plot styling and helpers
# -----------------------------------------------------------------------------

function configure_pyplot!()
    rc = PyPlot.matplotlib["rcParams"]
    rc["font.size"] = 10
    rc["axes.titlesize"] = 11
    rc["axes.labelsize"] = 10
    rc["xtick.labelsize"] = 9
    rc["ytick.labelsize"] = 9
    rc["legend.fontsize"] = 8
    rc["figure.dpi"] = 120
    rc["savefig.dpi"] = FIG_DPI
    rc["axes.spines.top"] = false
    rc["axes.spines.right"] = false
    return nothing
end

function annotate_heatmap!(ax, matrix; formatter=x -> @sprintf("%.2g", x))
    finite = finite_values(matrix)
    isempty(finite) && return nothing

    midpoint = (minimum(finite) + maximum(finite)) / 2

    for iy in axes(matrix, 1), ix in axes(matrix, 2)
        value = matrix[iy, ix]
        isfinite(value) || continue
        color = value > midpoint ? "white" : "black"
        ax.text(
            ix - 1,
            iy - 1,
            formatter(value),
            ha="center",
            va="center",
            fontsize=8,
            color=color,
        )
    end
    return nothing
end

function draw_heatmap!(
    ax,
    matrix,
    ranks,
    tolerances;
    title::AbstractString,
    colorbar_label::AbstractString,
    cmap::AbstractString="viridis",
    formatter=x -> @sprintf("%.2g", x),
)
    image = ax.imshow(matrix; aspect="auto", origin="upper", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(L"L_{rank}")
    ax.set_ylabel("shared_basis_tol")
    ax.set_xticks(0:(length(ranks)-1))
    ax.set_xticklabels(string.(ranks))
    ax.set_yticks(0:(length(tolerances)-1))
    ax.set_yticklabels(tolerance_label.(tolerances))
    annotate_heatmap!(ax, matrix; formatter=formatter)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    return image
end

function pareto_frontier_indices(times::AbstractVector, errors::AbstractVector)
    valid = findall(i -> isfinite(times[i]) && isfinite(errors[i]), eachindex(times))
    isempty(valid) && return Int[]

    ordered = sort(valid; by=i -> times[i])
    frontier = Int[]
    best_error = Inf

    for i in ordered
        if errors[i] < best_error
            push!(frontier, i)
            best_error = errors[i]
        end
    end

    return frontier
end


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

function save_overview_figure(
    run_dir::AbstractString,
    summaries,
    oversampling::Int,
)
    rows = filter(row -> row.oversampling == oversampling, summaries)
    ranks = unique_sorted(rows, :L_rank)
    tolerances = unique_sorted(rows, :shared_basis_tol; rev=true)

    nrmse = parameter_matrix(
        rows, ranks, tolerances, oversampling, :magnitude_nrmse_mean,
    )
    total_time = parameter_matrix(
        rows, ranks, tolerances, oversampling, :total_mean_s,
    )
    shared_rank = parameter_matrix(
        rows, ranks, tolerances, oversampling, :shared_rank_mean,
    )

    log_nrmse = copy(nrmse)
    for i in eachindex(log_nrmse)
        if isfinite(log_nrmse[i]) && log_nrmse[i] > 0
            log_nrmse[i] = log10(log_nrmse[i])
        else
            log_nrmse[i] = NaN
        end
    end

    fig, axes = subplots(2, 2; figsize=(12.5, 9.2), constrained_layout=true)

    draw_heatmap!(
        axes[1, 1],
        log_nrmse,
        ranks,
        tolerances;
        title="A  Magnitude NRMSE",
        colorbar_label=L"\log_{10}(\mathrm{NRMSE})",
        cmap="magma_r",
        formatter=x -> @sprintf("%.2e", 10.0^x),
    )

    draw_heatmap!(
        axes[1, 2],
        total_time,
        ranks,
        tolerances;
        title="B  Mean total time",
        colorbar_label="Time (s)",
        cmap="viridis",
        formatter=x -> @sprintf("%.1f", x),
    )

    draw_heatmap!(
        axes[2, 1],
        shared_rank,
        ranks,
        tolerances;
        title="C  Mean shared rank",
        colorbar_label="Shared rank",
        cmap="cividis",
        formatter=x -> @sprintf("%.1f", x),
    )

    valid_rows = filter(rows) do row
        valid_number(row.total_mean_s) &&
        valid_number(row.magnitude_nrmse_mean)
    end

    ax = axes[2, 2]
    if !isempty(valid_rows)
        times = Float64[row.total_mean_s for row in valid_rows]
        errors = Float64[row.magnitude_nrmse_mean for row in valid_rows]
        local_ranks = Float64[row.L_rank for row in valid_rows]

        scatter = ax.scatter(
            times,
            errors;
            c=local_ranks,
            cmap="viridis",
            s=55,
            alpha=0.85,
            edgecolors="none",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Mean total time (s)")
        ax.set_ylabel("Mean magnitude NRMSE")
        ax.set_title("D  Accuracy–time trade-off")
        ax.grid(true; alpha=0.25)

        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(L"L_{rank}")

        frontier = pareto_frontier_indices(times, errors)
        if !isempty(frontier)
            ax.plot(
                times[frontier],
                errors[frontier];
                linestyle="--",
                linewidth=1.4,
                marker="o",
                markersize=4,
                label="Pareto frontier",
            )
            ax.legend(loc="best")
        end

        best_order = sortperm(errors)
        for idx in Iterators.take(best_order, min(3, length(best_order)))
            row = valid_rows[idx]
            ax.annotate(
                @sprintf("L=%d, tol=%s", row.L_rank, tolerance_label(row.shared_basis_tol)),
                (times[idx], errors[idx]);
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        end
    else
        ax.text(0.5, 0.5, "No successful configurations", ha="center", va="center")
        ax.set_axis_off()
    end

    fig.suptitle(
        @sprintf("2D LowRank parameter sweep — oversampling = %d", oversampling),
        fontsize=13,
    )

    return save_figure(
        fig,
        joinpath(run_dir, "sweep_overview_os$(oversampling)"),
    )
end

function save_seed_stability_figure(
    run_dir::AbstractString,
    summaries,
    oversampling::Int,
)
    rows = filter(row -> row.oversampling == oversampling, summaries)
    ranks = unique_sorted(rows, :L_rank)
    tolerances = unique_sorted(rows, :shared_basis_tol; rev=true)

    fig, ax = subplots(; figsize=(8.6, 5.8), constrained_layout=true)

    for tol in tolerances
        group = [
            matching_summary_row(rows, rank, tol, oversampling)
            for rank in ranks
        ]

        xs = Int[]
        means = Float64[]
        stds = Float64[]

        for (rank, row) in zip(ranks, group)
            row === nothing && continue
            valid_number(row.magnitude_nrmse_mean) || continue
            push!(xs, rank)
            push!(means, Float64(row.magnitude_nrmse_mean))
            push!(stds, valid_number(row.magnitude_nrmse_std) ?
                        Float64(row.magnitude_nrmse_std) : 0.0)
        end

        isempty(xs) && continue
        ax.errorbar(
            xs,
            means;
            yerr=stds,
            marker="o",
            markersize=4,
            linewidth=1.3,
            capsize=3,
            label="tol=$(tolerance_label(tol))",
        )
    end

    ax.set_yscale("log")
    ax.set_xticks(ranks)
    ax.set_xlabel(L"L_{rank}")
    ax.set_ylabel("Magnitude NRMSE vs Kernel")
    ax.set_title(@sprintf("Seed variability — oversampling = %d", oversampling))
    ax.grid(true; which="both", alpha=0.25)
    ax.legend(title="shared_basis_tol", ncol=2)

    return save_figure(
        fig,
        joinpath(run_dir, "seed_stability_os$(oversampling)"),
    )
end

function save_pareto_figure(
    run_dir::AbstractString,
    summaries,
    oversampling::Int,
)
    rows = filter(summaries) do row
        row.oversampling == oversampling &&
        valid_number(row.total_mean_s) &&
        valid_number(row.magnitude_nrmse_mean) &&
        valid_number(row.shared_rank_mean)
    end

    fig, ax = subplots(; figsize=(8.2, 6.0), constrained_layout=true)

    if isempty(rows)
        ax.text(0.5, 0.5, "No successful configurations", ha="center", va="center")
        ax.set_axis_off()
    else
        times = Float64[row.total_mean_s for row in rows]
        errors = Float64[row.magnitude_nrmse_mean for row in rows]
        ranks = Float64[row.L_rank for row in rows]
        shared_ranks = Float64[row.shared_rank_mean for row in rows]

        minimum_shared = minimum(shared_ranks)
        maximum_shared = maximum(shared_ranks)
        size_span = maximum_shared > minimum_shared ?
                    (shared_ranks .- minimum_shared) ./ (maximum_shared - minimum_shared) :
                    fill(0.5, length(shared_ranks))
        marker_sizes = 45 .+ 130 .* size_span

        scatter = ax.scatter(
            times,
            errors;
            c=ranks,
            s=marker_sizes,
            cmap="viridis",
            alpha=0.8,
            edgecolors="none",
        )

        frontier = pareto_frontier_indices(times, errors)
        if !isempty(frontier)
            ax.plot(
                times[frontier],
                errors[frontier];
                linestyle="--",
                marker="o",
                markersize=4,
                linewidth=1.4,
                label="Pareto frontier",
            )
        end

        ax.set_yscale("log")
        ax.set_xlabel("Mean total time (s)")
        ax.set_ylabel("Mean magnitude NRMSE")
        ax.set_title(@sprintf("Accuracy–time–rank trade-off — oversampling = %d", oversampling))
        ax.grid(true; which="both", alpha=0.25)

        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(L"L_{rank}")

        ax.text(
            0.02,
            0.02,
            "Marker size: mean shared rank";
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
        )
        ax.legend(loc="best")
    end

    return save_figure(
        fig,
        joinpath(run_dir, "pareto_tradeoff_os$(oversampling)"),
    )
end


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

function visualize_sweep!(run_dir::AbstractString=resolve_run_directory())
    configure_pyplot!()

    summary_path = joinpath(run_dir, "summary_by_configuration.csv")
    runs_path = joinpath(run_dir, "runs.csv")

    summaries = read_csv_rows(summary_path)
    runs = read_csv_rows(runs_path)

    oversamplings = unique_sorted(summaries, :oversampling)
    isempty(oversamplings) && error("No oversampling values found in $summary_path")

    @info(
        "Visualizing 2D LowRank parameter sweep",
        run_dir,
        configurations=length(summaries),
        individual_runs=length(runs),
        oversamplings,
    )

    figure_paths = String[]

    for oversampling in oversamplings
        append!(figure_paths, save_overview_figure(run_dir, summaries, oversampling))
        append!(figure_paths, save_seed_stability_figure(run_dir, summaries, oversampling))
        append!(figure_paths, save_pareto_figure(run_dir, summaries, oversampling))
    end

    println("\nSaved sweep figures:")
    foreach(path -> println("  $path"), figure_paths)

    SHOW_FIGURES && show()
    return figure_paths
end


if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    visualize_sweep!()
end
