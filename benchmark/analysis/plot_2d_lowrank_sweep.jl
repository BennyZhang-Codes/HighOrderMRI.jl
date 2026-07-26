"""
Visualize a 2D single-shot local-low-rank sweep.

The corresponding run fixes `shared_basis_tol = 0` and rSVD oversampling to
five. Figures therefore compare local rank and rSVD seed only; they intentionally
contain no shared-basis tolerance heatmaps.

Usage:

    julia --project=. benchmark/analysis/plot_2d_lowrank_sweep.jl \
        benchmark/results/2d_local_lowrank_rank_sweep_<dataset>_<timestamp>
"""

using CSV
using PyPlot
using Printf
using Statistics

parse_env_bool(name::AbstractString, default::Bool) = begin
    raw = lowercase(strip(get(ENV, name, string(default))))
    raw in ("true", "1", "yes", "on") && return true
    raw in ("false", "0", "no", "off") && return false
    throw(ArgumentError("$name must be true/false; got $raw"))
end

const FIG_DPI = parse(Int, get(ENV, "HIGHORDER_SWEEP_FIG_DPI", "300"))
const SHOW_FIGURES = parse_env_bool("HIGHORDER_SWEEP_SHOW_FIGURES", false)
const DEFAULT_RESULTS_ROOT = get(
    ENV,
    "HIGHORDER_SWEEP_RESULTS_ROOT",
    normpath(joinpath(@__DIR__, "..", "results")),
)

valid_number(x) = !ismissing(x) && x isa Number && isfinite(Float64(x))

function newest_sweep_directory(root::AbstractString)
    isdir(root) || throw(ArgumentError("Sweep results root does not exist: $root"))
    candidates = filter(readdir(root; join=true)) do path
        isdir(path) && startswith(basename(path), "2d_local_lowrank_rank_sweep_")
    end
    isempty(candidates) && throw(ArgumentError("No local-low-rank sweep directory found under $root"))
    return candidates[argmax(mtime.(candidates))]
end

function resolve_run_directory()
    !isempty(ARGS) && return abspath(ARGS[1])
    from_env = strip(get(ENV, "HIGHORDER_SWEEP_RUN_DIR", ""))
    !isempty(from_env) && return abspath(from_env)
    return newest_sweep_directory(DEFAULT_RESULTS_ROOT)
end

function read_csv_rows(path::AbstractString)
    isfile(path) || throw(ArgumentError("CSV file not found: $path"))
    return collect(CSV.File(path; missingstring=["missing", ""]))
end

function save_figure(fig, basepath::AbstractString)
    paths = (basepath * ".png", basepath * ".pdf")
    for path in paths
        fig.savefig(path; dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.04, transparent=false)
    end
    SHOW_FIGURES || close(fig)
    return paths
end

function configure_pyplot!()
    rc = PyPlot.matplotlib["rcParams"]
    rc["font.size"] = 10
    rc["axes.titlesize"] = 11
    rc["axes.labelsize"] = 10
    rc["legend.fontsize"] = 8
    rc["figure.dpi"] = 120
    rc["savefig.dpi"] = FIG_DPI
    rc["axes.spines.top"] = false
    rc["axes.spines.right"] = false
    return nothing
end

function pareto_frontier_indices(times::AbstractVector, errors::AbstractVector)
    order = sortperm(times)
    frontier = Int[]
    best_error = Inf
    for index in order
        if isfinite(times[index]) && isfinite(errors[index]) && errors[index] < best_error
            push!(frontier, index)
            best_error = errors[index]
        end
    end
    return frontier
end

function successful_rank_rows(summaries)
    rows = [row for row in summaries if
        valid_number(row.magnitude_nrmse_mean) && valid_number(row.total_mean_s)]
    sort!(rows; by=row -> row.L_rank)
    return rows
end

function save_rank_summary_figure(run_dir::AbstractString, summaries)
    rows = successful_rank_rows(summaries)
    fig, axes = subplots(2, 2; figsize=(12, 8.5), constrained_layout=true)

    if isempty(rows)
        for ax in axes
            ax.text(0.5, 0.5, "No successful configurations", ha="center", va="center")
            ax.set_axis_off()
        end
        return save_figure(fig, joinpath(run_dir, "rank_summary"))
    end

    ranks = Int[row.L_rank for row in rows]
    nrmse = Float64[row.magnitude_nrmse_mean for row in rows]
    nrmse_std = Float64[valid_number(row.magnitude_nrmse_std) ? row.magnitude_nrmse_std : 0.0 for row in rows]
    complex_error = Float64[row.complex_error_mean for row in rows]
    complex_std = Float64[valid_number(row.complex_error_std) ? row.complex_error_std : 0.0 for row in rows]
    setup = Float64[row.setup_mean_s for row in rows]
    recon = Float64[row.recon_mean_s for row in rows]
    total = Float64[row.total_mean_s for row in rows]
    total_std = Float64[valid_number(row.total_std_s) ? row.total_std_s : 0.0 for row in rows]
    actual_rank = Float64[row.shared_rank_mean for row in rows]

    ax = axes[1, 1]
    ax.errorbar(ranks, nrmse; yerr=nrmse_std, marker="o", capsize=3, label="Magnitude NRMSE")
    ax.errorbar(ranks, complex_error; yerr=complex_std, marker="s", capsize=3, label="Complex relative error")
    ax.set_yscale("log")
    ax.set_xticks(ranks)
    ax.set_xlabel(L"L_{rank}")
    ax.set_ylabel("Error vs Kernel")
    ax.set_title("A  Local low-rank approximation error")
    ax.grid(true; which="both", alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1, 2]
    ax.plot(ranks, setup; marker="o", label="Setup")
    ax.plot(ranks, recon; marker="s", label="CG reconstruction")
    ax.errorbar(ranks, total; yerr=total_std, marker="^", capsize=3, label="Total")
    ax.set_xticks(ranks)
    ax.set_xlabel(L"L_{rank}")
    ax.set_ylabel("Time (s)")
    ax.set_title("B  End-to-end runtime")
    ax.grid(true; alpha=0.25)
    ax.legend(loc="best")

    ax = axes[2, 1]
    ax.plot(ranks, actual_rank; marker="o", label="Observed basis rank")
    ax.plot(ranks, ranks; linestyle="--", color="black", label="Requested local rank")
    ax.set_xticks(ranks)
    ax.set_xlabel(L"L_{rank}")
    ax.set_ylabel("Rank")
    ax.set_title("C  Rank retained with shared_basis_tol = 0")
    ax.grid(true; alpha=0.25)
    ax.legend(loc="best")

    ax = axes[2, 2]
    scatter = ax.scatter(total, nrmse; c=ranks, cmap="viridis", s=62, edgecolors="none")
    frontier = pareto_frontier_indices(total, nrmse)
    !isempty(frontier) && ax.plot(total[frontier], nrmse[frontier]; linestyle="--", marker="o", label="Pareto frontier")
    for index in eachindex(ranks)
        ax.annotate("L=$(ranks[index])", (total[index], nrmse[index]); xytext=(4, 4), textcoords="offset points", fontsize=8)
    end
    ax.set_yscale("log")
    ax.set_xlabel("Mean total time (s)")
    ax.set_ylabel("Mean magnitude NRMSE")
    ax.set_title("D  Accuracy–time trade-off")
    ax.grid(true; which="both", alpha=0.25)
    ax.legend(loc="best")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(L"L_{rank}")

    fig.suptitle("2D single-shot local LowRank sweep (five rSVD seeds, oversampling = 5)", fontsize=13)
    return save_figure(fig, joinpath(run_dir, "rank_summary"))
end

function save_seed_stability_figure(run_dir::AbstractString, runs)
    rows = [row for row in runs if row.status == "ok" && valid_number(row.magnitude_nrmse_vs_kernel)]
    sort!(rows; by=row -> (row.L_rank, row.rsvd_seed))
    fig, ax = subplots(; figsize=(9, 5.8), constrained_layout=true)

    if isempty(rows)
        ax.text(0.5, 0.5, "No successful seed runs", ha="center", va="center")
        ax.set_axis_off()
    else
        ranks = sort(unique(Int[row.L_rank for row in rows]))
        seeds = sort(unique(Int[row.rsvd_seed for row in rows]))
        for (seed_index, seed) in enumerate(seeds)
            group = [row for row in rows if row.rsvd_seed == seed]
            ax.plot(
                Int[row.L_rank for row in group],
                Float64[row.magnitude_nrmse_vs_kernel for row in group];
                marker="o", markersize=4, linewidth=1.1, alpha=0.8, label="seed=$seed",
            )
        end
        ax.set_yscale("log")
        ax.set_xticks(ranks)
        ax.set_xlabel(L"L_{rank}")
        ax.set_ylabel("Magnitude NRMSE vs Kernel")
        ax.set_title("rSVD seed stability")
        ax.grid(true; which="both", alpha=0.25)
        ax.legend(ncol=2, loc="best")
    end

    return save_figure(fig, joinpath(run_dir, "seed_stability"))
end

function visualize_sweep!(run_dir::AbstractString=resolve_run_directory())
    configure_pyplot!()
    summaries = read_csv_rows(joinpath(run_dir, "summary_by_rank.csv"))
    runs = read_csv_rows(joinpath(run_dir, "runs.csv"))
    paths = vcat(save_rank_summary_figure(run_dir, summaries), save_seed_stability_figure(run_dir, runs))
    println("\nSaved local-low-rank sweep figures:")
    foreach(path -> println("  $path"), paths)
    SHOW_FIGURES && show()
    return paths
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    visualize_sweep!()
end
