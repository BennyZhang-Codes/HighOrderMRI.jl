"""
Visualize a 2D single-shot local-low-rank sweep.

The corresponding run fixes `shared_basis_tol = 0` and rSVD oversampling to
five. Figures therefore compare local rank and rSVD seed only.

The rank-summary figure contains:

1. magnitude NRMSE and raw complex NRMSE versus local rank;
2. magnitude SSIM versus local rank;
3. setup, CG reconstruction, and total runtime versus local rank.

Usage:

    julia --project=. benchmark/analysis/plot_2d_lowrank_sweep.jl \
        benchmark/results/2d_local_lowrank_rank_sweep_<dataset>_<timestamp>
"""

using CSV
using PyPlot


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Leave empty to automatically use the newest sweep directory under
# `SWEEP_RESULTS_ROOT`. A command-line path overrides this value.
const SWEEP_RUN_DIR = ""
const SWEEP_RUN_DIR = raw"/home/jyzhang/Desktop/Julia_pkg/HighOrderMRI-benchmark/benchmark/results/2d_local_lowrank_rank_sweep_7T_2D_Spiral_1p0_200_r4_2026-07-27_135909"

const SWEEP_RESULTS_ROOT = normpath(joinpath(@__DIR__, "..", "results"))
const FIG_DPI = 900
const SHOW_FIGURES = false
const FONT_FAMILY = get(ENV, "HIGHORDER_SWEEP_FONT_FAMILY", "Arial")

FIG_DPI > 0 || throw(ArgumentError("FIG_DPI must be positive"))

const figure_width_summary = 22.5 / 2.53999863
const figure_height_summary = 7 / 2.53999863
const figure_width_seed = 10 / 2.53999863
const figure_height_seed = 7 / 2.53999863

const linewidth = 0.5
const linewidth_plot = 0.6
const linewidth_errorbar = 0.4
const ticklength = 1.5

const fontsize_subfigure = 9
const fontsize_label = 8
const fontsize_legend = 7
const fontsize_ticklabel = 7

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
const color_4 = "#E46A6A"
const color_5 = "#9467BD"
const color_6 = "#17BECF"
const color_7 = "#BCBD22"
const color_8 = "#E377C2"
const color_seed = (color_1, color_2, color_3, color_4, color_5, color_6, color_7, color_8)


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------

valid_number(x) = !ismissing(x) && x isa Number && isfinite(Float64(x))
float_values(rows, name::Symbol) = Float64[getproperty(row, name) for row in rows]
float_stds(rows, name::Symbol) = Float64[valid_number(getproperty(row, name)) ? getproperty(row, name) : 0.0 for row in rows]

function newest_sweep_directory(root::AbstractString)
    isdir(root) || throw(ArgumentError("Sweep results root does not exist: $root"))
    candidates = filter(path -> isdir(path) && startswith(basename(path), "2d_local_lowrank_rank_sweep_"), readdir(root; join=true))
    isempty(candidates) && throw(ArgumentError("No local-low-rank sweep directory found under $root"))
    return candidates[argmax(mtime.(candidates))]
end

function resolve_run_directory()
    !isempty(ARGS) && return abspath(ARGS[1])
    !isempty(strip(SWEEP_RUN_DIR)) && return abspath(strip(SWEEP_RUN_DIR))
    return newest_sweep_directory(SWEEP_RESULTS_ROOT)
end

function read_csv_rows(path::AbstractString)
    isfile(path) || throw(ArgumentError("CSV file not found: $path"))
    return collect(CSV.File(path; missingstring=["missing", ""]))
end

function successful_rank_rows(summaries)
    rows = [row for row in summaries if all(name -> valid_number(getproperty(row, name)), (
        :magnitude_nrmse_mean, :complex_error_mean, :magnitude_ssim_mean, :setup_mean_s, :recon_mean_s, :total_mean_s,
    ))]
    sort!(rows; by=row -> row.L_rank)
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

function style_axis!(ax)
    ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
    for spine in ax.spines
        ax.spines[spine].set_color(color_label)
        ax.spines[spine].set_visible(false)
    end
    ax.set_facecolor(color_facecolor)
    return ax
end

function save_figure(fig, outpath::AbstractString, name::AbstractString)
    mkpath(outpath)
    paths = [joinpath(outpath, "$name.$ext") for ext in ("png", "svg")]
    foreach(path -> fig.savefig(path; dpi=FIG_DPI, transparent=false, bbox_inches="tight", pad_inches=0.05), paths)
    SHOW_FIGURES || close(fig)
    return Tuple(paths)
end

function add_nrmse_reference_lines!(ax, ranks)
    isempty(ranks) && return nothing
    for (level, label) in ((0.01, "1%"), (0.02, "2%"))
        ax.axhline(level; color=color_reference, linestyle="--", linewidth=linewidth, zorder=0)
        ax.annotate(label, (minimum(ranks), level); xytext=(5, 2), textcoords="offset points", ha="right", va="bottom", fontsize=fontsize_ticklabel, color=color_reference)
    end
    return nothing
end

plot_errorbar!(ax, x, y, yerr; color, marker, label) = ax.errorbar(
    x, y; yerr=yerr, color=color, marker=marker, markersize=markersize_plot,
    linewidth=linewidth_plot, capsize=capsize_errorbar, elinewidth=linewidth_errorbar,
    capthick=linewidth_errorbar, label=label,
)


# -----------------------------------------------------------------------------
# Rank summary figure
# -----------------------------------------------------------------------------

function save_rank_summary_figure(outpath::AbstractString, summaries)
    rows = successful_rank_rows(summaries)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(figure_width_summary, figure_height_summary), facecolor=color_facecolor, squeeze=false)
    foreach(style_axis!, axs)

    if isempty(rows)
        for ax in axs
            ax.text(0.5, 0.5, "No successful configurations", ha="center", va="center", fontsize=fontsize_label, color=color_label)
            ax.set_axis_off()
        end
        return save_figure(fig, outpath, "rank_summary")
    end

    ranks = Int[row.L_rank for row in rows]
    nrmse = float_values(rows, :magnitude_nrmse_mean)
    complex_error = float_values(rows, :complex_error_mean)
    ssim = float_values(rows, :magnitude_ssim_mean)
    setup = float_values(rows, :setup_mean_s)
    recon = float_values(rows, :recon_mean_s)
    total = float_values(rows, :total_mean_s)

    ax = axs[1, 1]
    plot_errorbar!(ax, ranks, nrmse, float_stds(rows, :magnitude_nrmse_std); color=color_1, marker="o", label="Magnitude NRMSE")
    plot_errorbar!(ax, ranks, complex_error, float_stds(rows, :complex_error_std); color=color_2, marker="s", label="Raw complex NRMSE")
    add_nrmse_reference_lines!(ax, ranks)
    ax.set_yscale("log")
    ax.set_xticks(ranks)
    ax.set_xlabel(L"L_{rank}", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_ylabel("Error vs. Kernel", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.legend(loc="center left", bbox_to_anchor=(-0.05, 1.08), fontsize=fontsize_legend, labelcolor=color_label, ncols=2, frameon=false, handlelength=1, handletextpad=0.3, columnspacing=0.7, labelspacing=0.2)

    ax = axs[1, 2]
    plot_errorbar!(ax, ranks, ssim, float_stds(rows, :magnitude_ssim_std); color=color_3, marker="^", label="Magnitude SSIM")
    ax.set_xticks(ranks)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel(L"L_{rank}", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_ylabel("SSIM vs. Kernel", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.legend(loc="center left", bbox_to_anchor=(-0.05, 1.08), fontsize=fontsize_legend, labelcolor=color_label, ncols=1, frameon=false, handlelength=1, handletextpad=0.3, columnspacing=0.7, labelspacing=0.2)

    ax = axs[1, 3]
    plot_errorbar!(ax, ranks, setup, float_stds(rows, :setup_std_s); color=color_1, marker="o", label="Setup")
    plot_errorbar!(ax, ranks, recon, float_stds(rows, :recon_std_s); color=color_2, marker="s", label="CG reconstruction")
    plot_errorbar!(ax, ranks, total, float_stds(rows, :total_std_s); color=color_3, marker="^", label="Total")
    ax.set_xticks(ranks)
    ax.set_xlabel(L"L_{rank}", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_ylabel("Time [s]", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.legend(loc="center left", bbox_to_anchor=(-0.05, 1.08), fontsize=fontsize_legend, labelcolor=color_label, ncols=3, frameon=false, handlelength=1, handletextpad=0.3, columnspacing=0.6, labelspacing=0.2)

    fig.align_ylabels()
    fig.tight_layout(pad=0, h_pad=0, w_pad=0.8)
    fig.text(0.00, 1.00, "(a)", ha="left", va="top", fontsize=fontsize_subfigure, color=color_label)
    fig.text(0.33, 1.00, "(b)", ha="left", va="top", fontsize=fontsize_subfigure, color=color_label)
    fig.text(0.67, 1.00, "(c)", ha="left", va="top", fontsize=fontsize_subfigure, color=color_label)
    return save_figure(fig, outpath, "rank_summary")
end


# -----------------------------------------------------------------------------
# Seed stability figure
# -----------------------------------------------------------------------------

function save_seed_stability_figure(outpath::AbstractString, runs)
    rows = [row for row in runs if row.status == "ok" && valid_number(row.magnitude_nrmse_vs_kernel)]
    sort!(rows; by=row -> (row.rsvd_seed, row.L_rank))

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(figure_width_seed, figure_height_seed), facecolor=color_facecolor, squeeze=false)
    ax = style_axis!(axs[1])

    if isempty(rows)
        ax.text(0.5, 0.5, "No successful seed runs", ha="center", va="center", fontsize=fontsize_label, color=color_label)
        ax.set_axis_off()
    else
        ranks = sort(unique(Int[row.L_rank for row in rows]))
        seeds = sort(unique(Int[row.rsvd_seed for row in rows]))

        for (seed_index, seed) in enumerate(seeds)
            group = sort([row for row in rows if row.rsvd_seed == seed]; by=row -> row.L_rank)
            ax.plot(Int[row.L_rank for row in group], float_values(group, :magnitude_nrmse_vs_kernel); color=color_seed[mod1(seed_index, length(color_seed))], marker="o", markersize=markersize_plot, linewidth=linewidth_plot, label="seed = $seed")
        end

        add_nrmse_reference_lines!(ax, ranks)
        ax.set_yscale("log")
        ax.set_xticks(ranks)
        ax.set_xlabel(L"L_{rank}", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
        ax.set_ylabel("Magnitude NRMSE vs. Kernel", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
        ax.legend(loc="center left", bbox_to_anchor=(-0.05, 1.08), fontsize=fontsize_legend, labelcolor=color_label, ncols=min(length(seeds), 3), frameon=false, handlelength=1, handletextpad=0.3, columnspacing=0.7, labelspacing=0.2)
    end

    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    return save_figure(fig, outpath, "seed_stability")
end


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

function visualize_sweep!(run_dir::AbstractString=resolve_run_directory())
    configure_pyplot!()
    outpath = joinpath(run_dir, "figure")
    paths = vcat(
        save_rank_summary_figure(outpath, read_csv_rows(joinpath(run_dir, "summary_by_rank.csv"))),
        save_seed_stability_figure(outpath, read_csv_rows(joinpath(run_dir, "runs.csv"))),
    )
    println("\nSaved local-low-rank sweep figures:")
    foreach(path -> println("  $path"), paths)
    SHOW_FIGURES && show()
    return paths
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && visualize_sweep!()
