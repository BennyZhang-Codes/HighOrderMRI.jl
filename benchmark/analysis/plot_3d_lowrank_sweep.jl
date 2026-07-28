"""
Plot the completed 3D LowRank rank and shared-basis-tolerance sweeps.

Usage:

    julia --project=. benchmark/analysis/plot_3d_lowrank_sweep.jl \
        benchmark/results/3d_lowrank_sweep_<timestamp>

If no directory is supplied, the newest `3d_lowrank_sweep_*` directory under
`benchmark/results` is used.
"""

using CSV
using PyPlot


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Leave empty to automatically use the newest sweep directory under
# `SWEEP_RESULTS_ROOT`. A command-line path overrides this value.
const SWEEP_RUN_DIR = ""
const SWEEP_RESULTS_ROOT = normpath(joinpath(@__DIR__, "..", "results"))
const FIG_DPI = 900
const SHOW_FIGURES = false
const FONT_FAMILY = get(ENV, "HIGHORDER_SWEEP_FONT_FAMILY", "Arial")

FIG_DPI > 0 || throw(ArgumentError("FIG_DPI must be positive"))

# The 3D summary has four panels, so it is one panel wider than the 2D
# three-panel summary while retaining the same per-panel size.
const figure_width_summary = 30 / 2.53999863
const figure_height_summary = 6 / 2.53999863
const figure_width_empty = 10 / 2.53999863
const figure_height_empty = 7 / 2.53999863

const linewidth = 0.5
const linewidth_plot = 0.6
const ticklength = 1.5

const fontsize_subfigure = 9
const fontsize_label = 8
const fontsize_legend = 7
const fontsize_ticklabel = 7

const pad_labeltick = 2
const pad_label = 2
const markersize_plot = 2.5

const color_facecolor = "#FFFFFF"
const color_label = "#000000"
const color_1 = "#3E9DDE"
const color_2 = "#FF9F47"
const color_3 = "#30AE30"
const color_4 = "#E46A6A"
const color_5 = "#9467BD"
const color_6 = "#17BECF"
const color_7 = "#333333"


valid_number(value) =
    !ismissing(value) && value isa Number && isfinite(Float64(value))

function newest_run_directory(root::AbstractString)
    isdir(root) || throw(ArgumentError("Results directory does not exist: $root"))
    candidates = filter(path -> isdir(path) && startswith(basename(path), "3d_lowrank_sweep_"), readdir(root; join=true))
    isempty(candidates) && throw(ArgumentError("No 3d_lowrank_sweep_* directory was found under $root"))
    return candidates[argmax(mtime.(candidates))]
end

function resolve_run_directory()
    !isempty(ARGS) && return abspath(ARGS[1])
    !isempty(strip(SWEEP_RUN_DIR)) && return abspath(strip(SWEEP_RUN_DIR))
    return newest_run_directory(SWEEP_RESULTS_ROOT)
end

function read_rows(path::AbstractString)
    isfile(path) || throw(ArgumentError("CSV file not found: $path"))
    return collect(CSV.File(path; missingstring=["missing", ""]))
end

function successful_rows(rows, xfield::Symbol)
    required = (
        xfield,
        :magnitude_nrmse_masked,
        :complex_nrmse_masked,
        :magnitude_ssim_roi,
        :operator_setup_s,
        :cg_recon_s,
        :total_s,
        :shared_rank,
    )
    selected = [
        row for row in rows
        if row.status == "ok" &&
           all(field -> valid_number(getproperty(row, field)), required)
    ]
    sort!(selected; by=row -> getproperty(row, xfield))
    return selected
end

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

function save_figure(fig, output_dir::AbstractString, name::AbstractString)
    mkpath(output_dir)
    paths = [joinpath(output_dir, "$name.png"), joinpath(output_dir, "$name.svg")]
    for path in paths
        fig.savefig(path; dpi=FIG_DPI, transparent=false, bbox_inches="tight", pad_inches=0.05)
    end
    SHOW_FIGURES || close(fig)
    return paths
end

function empty_figure(output_dir, name, message)
    fig, axis = plt.subplots(figsize=(figure_width_empty, figure_height_empty), facecolor=color_facecolor)
    axis.text(0.5, 0.5, message; ha="center", va="center", fontsize=fontsize_label, color=color_label)
    axis.set_axis_off()
    return save_figure(fig, output_dir, name)
end

function save_sweep_figure(
    output_dir::AbstractString,
    rows,
    xfield::Symbol,
    figure_name::AbstractString,
    xlabel::AbstractString;
    logarithmic_x::Bool=false,
)
    rows = successful_rows(rows, xfield)
    isempty(rows) && return empty_figure(output_dir, figure_name, "No successful configurations")

    x = Float64[getproperty(row, xfield) for row in rows]
    nrmse = Float64[row.magnitude_nrmse_masked for row in rows]
    complex_error = Float64[row.complex_nrmse_masked for row in rows]
    ssim = Float64[row.magnitude_ssim_roi for row in rows]
    setup = Float64[row.operator_setup_s for row in rows]
    recon = Float64[row.cg_recon_s for row in rows]
    total = Float64[row.total_s for row in rows]
    shared_rank = Int[row.shared_rank for row in rows]

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(figure_width_summary, figure_height_summary), facecolor=color_facecolor, squeeze=false)
    axes = [axes[1, column] for column in 1:4]
    foreach(style_axis!, axes)

    axis = axes[1]
    axis.plot(x, nrmse; color=color_1, marker="o", markersize=markersize_plot, linewidth=linewidth_plot, label="Magnitude NRMSE")
    axis.plot(x, complex_error; color=color_2, marker="s", markersize=markersize_plot, linewidth=linewidth_plot, label="Complex NRMSE")
    axis.set_yscale("log")
    axis.set_yticks((0.005, 0.01, 0.02, 0.05, 0.1))
    axis.set_yticklabels(("0.005", "0.01", "0.02", "0.05", "0.1"))
    axis.set_ylabel("Error vs. Kernel", fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    axis.yaxis.label.set_fontsize(fontsize_label)
    axis.legend(loc="center left", bbox_to_anchor=(-0.05, 1.08), fontsize=fontsize_legend, labelcolor=color_label, ncols=2,
        frameon=false, handlelength=1, handletextpad=0.3, columnspacing=0.7, labelspacing=0.2)

    axis = axes[2]
    axis.plot(x, ssim; color=color_3, marker="^", markersize=markersize_plot, linewidth=linewidth_plot)
    axis.set_ylim(0.998, 1.0)
    axis.set_ylabel("SSIM"; fontsize=fontsize_label, color=color_label, labelpad=pad_label)

    axis = axes[3]
    axis.plot(x, setup; color=color_5, marker="o", markersize=markersize_plot, linewidth=linewidth_plot, label="Operator setup")
    axis.plot(x, recon; color=color_4, marker="s", markersize=markersize_plot, linewidth=linewidth_plot, label="CG reconstruction")
    axis.plot(x, total; color=color_6, marker="^", markersize=markersize_plot, linewidth=linewidth_plot, label="Total")
    axis.set_ylabel("Time [s]"; fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    axis.legend(loc="center left", bbox_to_anchor=(-0.05, 1.08), fontsize=fontsize_legend, labelcolor=color_label, ncols=3,
        frameon=false, handlelength=1, handletextpad=0.3, columnspacing=0.6, labelspacing=0.2)

    axis = axes[4]
    axis.plot(x, shared_rank; color=color_7, marker="o", markersize=markersize_plot, linewidth=linewidth_plot)
    axis.set_ylabel("Final shared rank"; fontsize=fontsize_label, color=color_label, labelpad=pad_label)

    for axis in axes
        if logarithmic_x
            tolerance_ticks = sort(unique(x))
            axis.set_xscale("log")
            axis.set_xticks(tolerance_ticks)
            axis.set_xticklabels(string.(tolerance_ticks))
        else
            axis.set_xticks(x)
        end
        axis.set_xlabel(xlabel; fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    end

    fig.align_ylabels()
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    for (axis, label) in zip(axes, ("(a)", "(b)", "(c)", "(d)"))
        axis.text(-0.15, 1.11, label; transform=axis.transAxes, ha="left", va="top",
            fontsize=fontsize_subfigure, color=color_label, clip_on=false)
    end
    return save_figure(fig, output_dir, figure_name)
end

function plot_3d_sweep!(run_dir::AbstractString=resolve_run_directory())
    configure_pyplot!()
    output_dir = joinpath(run_dir, "figure")

    paths = String[]
    append!(paths, save_sweep_figure(output_dir, read_rows(joinpath(run_dir, "rank_sweep.csv")),
        :L_rank, "rank_summary", "Local rank"))
    append!(paths, save_sweep_figure(output_dir, read_rows(joinpath(run_dir, "basis_tol_sweep.csv")),
        :shared_basis_tol, "basis_tolerance_summary", "Shared-basis tolerance"; logarithmic_x=true))

    println("\nSaved 3D sweep figures:")
    foreach(path -> println("  $path"), paths)
    SHOW_FIGURES && show()
    return paths
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && plot_3d_sweep!()
