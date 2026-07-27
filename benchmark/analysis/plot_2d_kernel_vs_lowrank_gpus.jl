"""
Plot the output of `run_2d_kernel_vs_lowrank_gpus.jl`.

The 1 × 3 summary figure contains:

1. Kernel and LowRank total time versus GPU count;
2. multi-GPU reconstruction speedup versus the baseline GPU count;
3. LowRank speedup over Kernel for setup, reconstruction, and total time.

Usage:

    julia --project=. benchmark/analysis/plot_2d_kernel_vs_lowrank_gpus.jl \
        benchmark/results/2d_kernel_lowrank_gpus_<dataset>_<timestamp>
"""

using CSV
using PyPlot


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Leave empty to automatically use the newest benchmark directory under
# `SWEEP_RESULTS_ROOT`. A command-line path overrides this value.
const SWEEP_RUN_DIR = ""
# const SWEEP_RUN_DIR = raw"E:\HighOrderMRI\benchmark\results\2d_kernel_lowrank_gpus_xxx"

const SWEEP_RESULTS_ROOT = normpath(joinpath(@__DIR__, "..", "results"))
const FIG_DPI = 900
const SHOW_FIGURES = false
const FONT_FAMILY = get(ENV, "HIGHORDER_SWEEP_FONT_FAMILY", "Arial")

FIG_DPI > 0 || throw(ArgumentError("FIG_DPI must be positive"))

const figure_width = 18 / 2.53999863
const figure_height = 6 / 2.53999863

const linewidth = 0.5
const linewidth_plot = 0.6
const linewidth_errorbar = 0.4
const ticklength = 1.5

const fontsize_subfigure = 9
const fontsize_label = 7
const fontsize_legend = 6
const fontsize_ticklabel = 6

const pad_labeltick = 2
const pad_label = 2
const markersize_plot = 2.5
const capsize_errorbar = 1.5

const color_facecolor = "#FFFFFF"
const color_label = "#000000"
const color_reference = "#777777"
const color_1 = "#3E9DDE"
const color_2 = "#FF9F47"
const color_3 = "#30AE30"


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------

valid_number(x) = !ismissing(x) && x isa Number && isfinite(Float64(x))
float_values(rows, name::Symbol) = Float64[getproperty(row, name) for row in rows]
float_stds(rows, name::Symbol) = Float64[valid_number(getproperty(row, name)) ? getproperty(row, name) : 0.0 for row in rows]

function newest_run_directory(root::AbstractString)
    isdir(root) || throw(ArgumentError("Results root does not exist: $root"))
    candidates = filter(path -> isdir(path) && startswith(basename(path), "2d_kernel_lowrank_gpus_"), readdir(root; join=true))
    isempty(candidates) && throw(ArgumentError("No 2D Kernel/LowRank GPU-scaling directory found under $root"))
    return candidates[argmax(mtime.(candidates))]
end

function resolve_run_directory()
    !isempty(ARGS) && return abspath(ARGS[1])
    !isempty(strip(SWEEP_RUN_DIR)) && return abspath(strip(SWEEP_RUN_DIR))
    return newest_run_directory(SWEEP_RESULTS_ROOT)
end

function read_csv_rows(path::AbstractString)
    isfile(path) || throw(ArgumentError("CSV file not found: $path"))
    return collect(CSV.File(path; missingstring=["missing", ""]))
end

function method_rows(summary, method::AbstractString)
    names = (:recon_mean_s, :total_mean_s, :recon_speedup_vs_baseline)
    rows = [row for row in summary if row.method == method && all(name -> valid_number(getproperty(row, name)), names)]
    sort!(rows; by=row -> row.gpu_count)
    return rows
end


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

function configure_pyplot!()
    matplotlib.rc("mathtext", default="regular")
    matplotlib.rc("figure", dpi=200)
    matplotlib.rc("font", family=FONT_FAMILY)
    matplotlib.rcParams["font.family"] = FONT_FAMILY
    matplotlib.rcParams["mathtext.default"] = "regular"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    @info "PyPlot style configured" font=FONT_FAMILY dpi=FIG_DPI
    return nothing
end

function style_axis!(ax, gpu_counts)
    ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
    for spine in ax.spines
        ax.spines[spine].set_color(color_label)
        ax.spines[spine].set_visible(false)
    end
    ax.set_facecolor(color_facecolor)
    ax.set_xticks(gpu_counts)
    return ax
end

function save_figure(fig, outpath::AbstractString, name::AbstractString)
    mkpath(outpath)
    paths = [joinpath(outpath, "$name.$ext") for ext in ("png", "svg")]
    foreach(path -> fig.savefig(path; dpi=FIG_DPI, transparent=false, bbox_inches="tight", pad_inches=0.05), paths)
    SHOW_FIGURES || close(fig)
    return Tuple(paths)
end

plot_errorbar!(ax, x, y, yerr; color, marker, label) = ax.errorbar(
    x, y; yerr=yerr, color=color, marker=marker, markersize=markersize_plot,
    linewidth=linewidth_plot, capsize=capsize_errorbar, elinewidth=linewidth_errorbar,
    capthick=linewidth_errorbar, label=label,
)

plot_line!(ax, x, y; color, marker, label, linestyle="-") = ax.plot(
    x, y; color=color, marker=marker, markersize=markersize_plot,
    linewidth=linewidth_plot, linestyle=linestyle, label=label,
)

function add_legend!(ax; ncols)
    ax.legend(loc="center left", bbox_to_anchor=(-0.05, 1.08), fontsize=fontsize_legend, labelcolor=color_label,
        ncols=ncols, frameon=false, handlelength=1, handletextpad=0.3, columnspacing=0.7, labelspacing=0.2)
    return nothing
end


# -----------------------------------------------------------------------------
# GPU-scaling summary figure
# -----------------------------------------------------------------------------

function save_gpu_summary_figure(outpath::AbstractString, summary, comparison)
    kernel = method_rows(summary, "HighOrderOp_Kernel")
    lowrank = method_rows(summary, "HighOrderLowRankOp")
    isempty(kernel) && throw(ArgumentError("No successful Kernel summary rows"))
    isempty(lowrank) && throw(ArgumentError("No successful LowRank summary rows"))

    sort!(comparison; by=row -> row.gpu_count)
    gpu_counts = Int[row.gpu_count for row in kernel]
    gpu_counts == Int[row.gpu_count for row in lowrank] == Int[row.gpu_count for row in comparison] ||
        throw(ArgumentError("Kernel, LowRank, and comparison GPU counts do not match"))

    lowrank_rank = Int(first(comparison).lowrank_rank)
    kernel_label = "Kernel"
    lowrank_label = "LowRank (L = $lowrank_rank)"

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(figure_width, figure_height), facecolor=color_facecolor, squeeze=false)
    foreach(ax -> style_axis!(ax, gpu_counts), axs)

    ax = axs[1, 1]
    plot_errorbar!(ax, gpu_counts, float_values(kernel, :total_mean_s), float_stds(kernel, :total_std_s); color=color_1, marker="o", label=kernel_label)
    plot_errorbar!(ax, gpu_counts, float_values(lowrank, :total_mean_s), float_stds(lowrank, :total_std_s); color=color_2, marker="s", label=lowrank_label)
    ax.set_xlabel("Number of GPUs", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_ylabel("Total time [s]", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    add_legend!(ax; ncols=2)

    ax = axs[1, 2]
    plot_line!(ax, gpu_counts, float_values(kernel, :recon_speedup_vs_baseline); color=color_1, marker="o", label=kernel_label)
    plot_line!(ax, gpu_counts, float_values(lowrank, :recon_speedup_vs_baseline); color=color_2, marker="s", label=lowrank_label)
    baseline_gpu_count = Int(first(kernel).baseline_gpu_count)
    plot_line!(ax, gpu_counts, Float64.(gpu_counts) ./ baseline_gpu_count; color=color_reference, marker="", linestyle="--", label="Ideal")
    ax.set_xlabel("Number of GPUs", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_ylabel("Reconstruction speedup", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    add_legend!(ax; ncols=3)

    ax = axs[1, 3]
    plot_line!(ax, gpu_counts, float_values(comparison, :lowrank_speedup_vs_kernel_setup); color=color_1, marker="o", label="Setup")
    plot_line!(ax, gpu_counts, float_values(comparison, :lowrank_speedup_vs_kernel_recon); color=color_2, marker="s", label="CG reconstruction")
    plot_line!(ax, gpu_counts, float_values(comparison, :lowrank_speedup_vs_kernel_total); color=color_3, marker="^", label="Total")
    ax.axhline(1.0; color=color_reference, linestyle="--", linewidth=linewidth, zorder=0)
    ax.set_xlabel("Number of GPUs", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_ylabel("Kernel time / LowRank time", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    add_legend!(ax; ncols=3)

    fig.align_ylabels()
    fig.tight_layout(pad=0, h_pad=0, w_pad=0.8)
    fig.text(0.00-0.01, 1.00, "(a)", ha="left", va="top", fontsize=fontsize_subfigure, color=color_label)
    fig.text(0.33-0.01, 1.00, "(b)", ha="left", va="top", fontsize=fontsize_subfigure, color=color_label)
    fig.text(0.67-0.01, 1.00, "(c)", ha="left", va="top", fontsize=fontsize_subfigure, color=color_label)
    return save_figure(fig, outpath, "gpus_summary")
end


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

function plot_gpus!(run_dir::AbstractString=resolve_run_directory())
    configure_pyplot!()
    paths = save_gpu_summary_figure(
        joinpath(run_dir, "figure"),
        read_csv_rows(joinpath(run_dir, "summary.csv")),
        read_csv_rows(joinpath(run_dir, "comparison.csv")),
    )
    println("\nSaved GPU-scaling figures:")
    foreach(path -> println("  $path"), paths)
    SHOW_FIGURES && show()
    return paths
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    plot_gpus!()
end
