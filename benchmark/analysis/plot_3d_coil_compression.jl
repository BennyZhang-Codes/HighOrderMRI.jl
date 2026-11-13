"""
Plot the summary produced by `sweep_3d_coil_compression.jl`.

Usage:

    /home/jyzhang/julia/julia-1.12.7/bin/julia --project=benchmark \
        benchmark/analysis/plot_3d_coil_compression.jl \
        benchmark/results/3d_coil_compression_<timestamp>

With no command-line path, the newest matching result directory is used.
The plotted total time is coil reconfiguration plus `recon_HOOp`; the common
coil-independent LowRank phase-basis setup is reported separately in the CSV.
"""

using PyPlot

include(joinpath(@__DIR__, "..", "common", "benchmark_utils.jl"))


# -----------------------------------------------------------------------------
# Plot configuration
# -----------------------------------------------------------------------------

const SWEEP_RUN_DIR = ""
const SWEEP_RESULTS_ROOT = normpath(joinpath(@__DIR__, "..", "results"))
const FIG_DPI = 900
const SHOW_FIGURES = false
const FONT_FAMILY = get(ENV, "HIGHORDER_3D_CC_FONT_FAMILY", "Arial")

const figure_width_summary = 18.0 / 2.53999863
const figure_height_summary = 6.5 / 2.53999863

const linewidth = 0.5
const linewidth_plot = 0.6
const linewidth_errorbar = 0.4
const ticklength = 1.5

const fontsize_subfigure = 9
const fontsize_label = 8
const fontsize_legend = 6.8
const fontsize_ticklabel = 7

const pad_labeltick = 2
const pad_label = 2
const markersize_plot = 2.8
const capsize_errorbar = 1.5

const color_facecolor = "#FFFFFF"
const color_label = "#000000"
const color_1 = "#3E9DDE"
const color_2 = "#FF9F47"
const color_3 = "#30AE30"
const color_4 = "#9467BD"


# -----------------------------------------------------------------------------
# Input and style helpers
# -----------------------------------------------------------------------------

function newest_3d_coil_directory(root::AbstractString)
    return newest_benchmark_directory(root, "3d_coil_compression_", ("coil_compression_comparison.csv",))
end

function resolve_3d_coil_run_directory()
    !isempty(ARGS) && return abspath(ARGS[1])
    !isempty(strip(SWEEP_RUN_DIR)) && return abspath(strip(SWEEP_RUN_DIR))
    return newest_3d_coil_directory(SWEEP_RESULTS_ROOT)
end

function configure_3d_coil_plot!()
    matplotlib.rc("mathtext", default="regular")
    matplotlib.rc("figure", dpi=200)
    matplotlib.rc("font", family=FONT_FAMILY)
    matplotlib.rcParams["font.family"] = FONT_FAMILY
    matplotlib.rcParams["mathtext.default"] = "regular"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    return nothing
end

function style_3d_coil_axis!(axis)
    axis.tick_params(
        axis="both",
        length=ticklength,
        width=linewidth,
        pad=pad_labeltick,
        color=color_label,
        labelcolor=color_label,
        labelsize=fontsize_ticklabel,
    )
    for spine in axis.spines
        axis.spines[spine].set_visible(false)
    end
    axis.set_facecolor(color_facecolor)
    axis.grid(true; color="#D9D9D9", linewidth=linewidth_errorbar, alpha=0.6)
    return axis
end

function finite_3d_coil_summary_rows(path::AbstractString)
    rows = read_benchmark_csv(path)
    isempty(rows) && throw(ArgumentError("Summary CSV contains no data rows: $path"))
    required = (
        :status,
        :virtual_coils,
        :magnitude_nrmse_mean,
        :magnitude_nrmse_std,
        :magnitude_ssim_mean,
        :magnitude_ssim_std,
        :total_median_s,
        :total_std_s,
        :retained_energy,
    )
    missing_columns = filter(name -> !hasproperty(first(rows), name), required)
    isempty(missing_columns) || throw(ArgumentError(
        "Summary CSV is missing columns: $(join(missing_columns, ", "))",
    ))
    numeric_columns = filter(name -> !(name in (:status, :operator)), required)
    rows = [row for row in rows if isequal(row.status, "ok") &&
        all(name -> getproperty(row, name) isa Real && isfinite(getproperty(row, name)), numeric_columns)]
    sort!(rows; by=row -> row.virtual_coils)
    isempty(rows) && throw(ArgumentError("Summary contains no successful rows"))
    return rows
end


# -----------------------------------------------------------------------------
# Summary figure
# -----------------------------------------------------------------------------

function plot_3d_coil_compression!(
    run_dir::AbstractString=resolve_3d_coil_run_directory(),
)
    configure_3d_coil_plot!()
    rows = finite_3d_coil_summary_rows(joinpath(
        run_dir,
        "coil_compression_comparison.csv",
    ))
    channels = Int[row.virtual_coils for row in rows]
    nrmse = Float64[row.magnitude_nrmse_mean for row in rows]
    nrmse_plot = [value > 0 ? value : NaN for value in nrmse]
    nrmse_std = Float64[row.magnitude_nrmse_std for row in rows]
    ssim = Float64[row.magnitude_ssim_mean for row in rows]
    ssim_std = Float64[row.magnitude_ssim_std for row in rows]
    total_time = Float64[row.total_median_s for row in rows]
    total_time_std = Float64[row.total_std_s for row in rows]
    retained = Float64[row.retained_energy for row in rows]

    fig, axes = subplots(
        nrows=1,
        ncols=3,
        figsize=(figure_width_summary, figure_height_summary),
        facecolor=color_facecolor,
        constrained_layout=true,
        squeeze=false,
    )
    axes = vec(axes)
    foreach(style_3d_coil_axis!, axes)

    axes[1].errorbar(
        channels,
        nrmse_plot;
        yerr=nrmse_std,
        color=color_2,
        marker="s",
        markersize=markersize_plot,
        linewidth=linewidth_plot,
        elinewidth=linewidth_errorbar,
        capsize=capsize_errorbar,
        label="LowRank",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel(
        "Magnitude NRMSE vs.\nKernel reference",
        fontsize=fontsize_label,
        labelpad=pad_label,
    )
    axes[1].legend(
        loc="best",
        fontsize=fontsize_legend,
        frameon=false,
        handlelength=1.2,
    )

    axes[2].errorbar(
        channels,
        ssim;
        yerr=ssim_std,
        color=color_2,
        marker="s",
        markersize=markersize_plot,
        linewidth=linewidth_plot,
        elinewidth=linewidth_errorbar,
        capsize=capsize_errorbar,
        label="LowRank",
    )
    ssim_min = minimum(ssim .- ssim_std)
    ssim_max = maximum(ssim .+ ssim_std)
    ssim_span = max(ssim_max - ssim_min, 1e-4)
    ssim_padding = 0.1 * ssim_span
    axes[2].set_ylim(
        max(0.0, ssim_min - ssim_padding),
        ssim_max + ssim_padding,
    )
    axes[2].set_ylabel("Magnitude SSIM", fontsize=fontsize_label, labelpad=pad_label)
    axes[2].legend(
        loc="best",
        fontsize=fontsize_legend,
        frameon=false,
        handlelength=1.2,
    )

    axes[3].errorbar(
        channels,
        total_time;
        yerr=total_time_std,
        color=color_2,
        marker="s",
        markersize=markersize_plot,
        linewidth=linewidth_plot,
        elinewidth=linewidth_errorbar,
        capsize=capsize_errorbar,
        label="LowRank",
    )
    time_min = minimum(total_time .- total_time_std)
    time_max = maximum(total_time .+ total_time_std)
    time_span = max(time_max - time_min, eps(max(abs(time_max), 1.0)))
    time_padding = 0.1 * time_span
    axes[3].set_ylim(max(0.0, time_min - time_padding), time_max + time_padding)
    axes[3].set_ylabel(
        "Reused basis: update + CG [s]",
        fontsize=fontsize_label,
        labelpad=pad_label,
    )
    axes[3].legend(
        loc="best",
        fontsize=fontsize_legend,
        frameon=false,
        handlelength=1.2,
    )

    energy_axis = axes[3].twinx()
    energy_axis.plot(
        channels,
        retained;
        color=color_4,
        marker="d",
        markersize=markersize_plot,
        linewidth=linewidth_plot,
        linestyle=":",
        label="Retained energy",
    )
    retained_min = minimum(retained)
    retained_max = maximum(retained)
    retained_span = max(retained_max - retained_min, 1e-4)
    retained_padding = 0.1 * retained_span
    energy_axis.set_ylim(
        max(0.0, retained_min - retained_padding),
        retained_max + retained_padding,
    )
    energy_axis.set_ylabel(
        "Retained coil energy",
        fontsize=fontsize_label,
        color=color_4,
    )
    energy_axis.tick_params(
        axis="y",
        labelsize=fontsize_ticklabel,
        colors=color_4,
        length=ticklength,
        width=linewidth,
    )
    for spine in energy_axis.spines
        energy_axis.spines[spine].set_visible(false)
    end
    energy_axis.legend(
        loc="lower right",
        fontsize=fontsize_legend,
        frameon=false,
        handlelength=1.2,
    )

    for (index, axis) in enumerate(axes)
        axis.set_xticks(channels)
        axis.set_xlabel(
            "Number of channels",
            fontsize=fontsize_label,
            labelpad=pad_label,
        )
        axis.text(
            -0.16,
            1.08,
            "($(Char(Int('a') + index - 1)))";
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize_subfigure,
            color=color_label,
        )
    end

    figure_dir = joinpath(run_dir, "figure")
    mkpath(figure_dir)
    paths = String[]
    for extension in ("png", "svg")
        path = joinpath(figure_dir, "coil_compression_summary.$extension")
        fig.savefig(
            path;
            dpi=FIG_DPI,
            transparent=false,
            bbox_inches="tight",
            pad_inches=0.03,
        )
        push!(paths, path)
    end
    SHOW_FIGURES || close(fig)

    println("\nSaved 3D coil-compression summary figures:")
    foreach(path -> println("  $path"), paths)
    SHOW_FIGURES && show()
    return paths
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && plot_3d_coil_compression!()
