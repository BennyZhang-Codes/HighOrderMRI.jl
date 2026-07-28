"""
Plot middle slices from the completed 3D local-rank sweep.

For each anatomical direction, the output contains two rows:

1. magnitude reconstruction;
2. 20× magnitude absolute difference from the Kernel reference.

The first eight columns follow `L_rank = 2, 4, 6, 8, 10, 15, 20, 25`. The
rightmost cell of the first row contains the Kernel reference; the cell below
it is blank. The shared-basis-tolerance sweep is not included.

Usage:

    julia --project=. benchmark/analysis/plot_3d_rank_mid_slices.jl

An alternative sweep directory can be supplied as the first argument.
"""

using CSV
using HighOrderMRI: plt_image
using NIfTI
using PyPlot


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

const SWEEP_RUN_DIR = raw"/home/jyzhang/Desktop/Julia_pkg/HighOrderMRI-benchmark/benchmark/results/3d_lowrank_sweep_2026-07-28_135204"
const RANKS = (2, 4, 6, 8, 10, 15, 20, 25)
const REFERENCE_MAG_FILENAME = "HighOrderOp_Kernel_reference_mag.nii.gz"
const DIRECTIONS = (
    (name="axial", label="Axial", dim=3, rotation=:right),
    (name="coronal", label="Coronal", dim=2, rotation=:left),
    (name="sagittal", label="Sagittal", dim=1, rotation=:left),
)

const FIG_DPI = 900
const SHOW_FIGURES = false
const FONT_FAMILY = get(ENV, "HIGHORDER_SWEEP_FONT_FAMILY", "Arial")

const panel_width_cm = 2.8
const fontsize_column = 8
const fontsize_row = 8
const color_facecolor = "#FFFFFF"
const color_label = "#000000"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

function resolve_run_directory()
    return !isempty(ARGS) ? abspath(ARGS[1]) : SWEEP_RUN_DIR
end

function read_rank_rows(run_dir::AbstractString)
    csv_path = joinpath(run_dir, "rank_sweep.csv")
    isfile(csv_path) || throw(ArgumentError("Rank-sweep CSV not found: $csv_path"))

    rows = [
        row for row in CSV.File(csv_path; missingstring=["missing", ""])
        if row.status == "ok" && row.in_rank_sweep
    ]
    rows_by_rank = Dict(Int(row.L_rank) => row for row in rows)

    missing_ranks = [rank for rank in RANKS if !haskey(rows_by_rank, rank)]
    isempty(missing_ranks) || throw(ArgumentError(
        "Successful rank-sweep rows are missing for L_rank=$(join(missing_ranks, ", "))",
    ))
    return [rows_by_rank[rank] for rank in RANKS]
end

function read_nifti_volume(path::AbstractString)
    isfile(path) || throw(ArgumentError("NIfTI file not found: $path"))
    volume = Array(NIfTI.niread(path).raw)
    ndims(volume) == 3 || throw(ArgumentError(
        "Expected a 3D NIfTI volume at $path, got size $(size(volume))",
    ))
    return volume
end

function middle_slice(volume::AbstractArray{<:Real,3}, direction)
    index = cld(size(volume, direction.dim), 2)
    slice = Float32.(abs.(Array(selectdim(volume, direction.dim, index))))
    return direction.rotation == :right ? rotr90(slice) : rotl90(slice)
end

function load_middle_slices(run_dir::AbstractString, rows)
    reconstruction_slices = [Matrix{Float32}[] for _ in DIRECTIONS]
    difference_slices = [Matrix{Float32}[] for _ in DIRECTIONS]

    for row in rows
        reconstruction_path = joinpath(run_dir, String(row.reconstruction_mag_nifti))
        difference_path = joinpath(run_dir, String(row.difference_nifti))

        reconstruction = read_nifti_volume(reconstruction_path)
        for (direction_index, direction) in enumerate(DIRECTIONS)
            push!(reconstruction_slices[direction_index], middle_slice(reconstruction, direction))
        end
        reconstruction = nothing

        difference = read_nifti_volume(difference_path)
        for (direction_index, direction) in enumerate(DIRECTIONS)
            push!(difference_slices[direction_index], 2f0 .* middle_slice(difference, direction))
        end
        difference = nothing
        GC.gc()
    end

    reference_path = joinpath(run_dir, "reference", REFERENCE_MAG_FILENAME)
    reference = read_nifti_volume(reference_path)
    reference_slices = [middle_slice(reference, direction) for direction in DIRECTIONS]

    return reconstruction_slices, difference_slices, reference_slices
end


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

function configure_pyplot!()
    matplotlib.rc("font", family=FONT_FAMILY)
    matplotlib.rcParams["font.family"] = FONT_FAMILY
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    return nothing
end

function save_direction_figure(
    output_dir::AbstractString,
    direction,
    reconstruction_slices,
    difference_slices,
    reference_slice,
    reference_vmax::Real,
)
    blank_slice = zeros(Float32, size(reference_slice))
    images = vcat(reconstruction_slices, [reference_slice], difference_slices, [blank_slice])
    column_labels = vcat(["L = $rank" for rank in RANKS], ["Reference"])

    fig = plt_image(images; nRow=2, nCol=length(column_labels), width=panel_width_cm, vmin=0, vmax=reference_vmax,
        cmap="gray", color_facecolor=color_facecolor, color_label=color_label)
    axis = fig.axes[1]

    for (column, label) in enumerate(column_labels)
        x_position = (column - 0.5) / length(column_labels)
        axis.text(x_position, 1.01, label; transform=axis.transAxes, ha="center", va="bottom", 
            fontsize=fontsize_column, color=color_label, clip_on=false)
    end

    axis.text(-0.01, 0.75, "Recon"; transform=axis.transAxes, rotation=90, ha="right", va="center",
        fontsize=fontsize_row, color=color_label, clip_on=false)
    axis.text(-0.01, 0.25, "Error ×20"; transform=axis.transAxes, rotation=90, ha="right", va="center",
        fontsize=fontsize_row, color=color_label, clip_on=false)

    mkpath(output_dir)
    output_path = joinpath(output_dir, "rank_sweep_$(direction.name)_mid_slice.png")
    try
        fig.savefig(output_path; dpi=FIG_DPI, transparent=false, bbox_inches="tight", pad_inches=0.03)
    finally
        SHOW_FIGURES || close(fig)
    end
    return output_path
end


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

function plot_3d_rank_mid_slices!(run_dir::AbstractString=resolve_run_directory())
    configure_pyplot!()
    rows = read_rank_rows(run_dir)

    reference_vmax = Float64(rows[1].reference_vmax)
    reference_vmax > 0 || throw(ArgumentError("reference_vmax must be positive, got $reference_vmax"))
    all(row -> isapprox(Float64(row.reference_vmax), reference_vmax; rtol=0, atol=eps(reference_vmax)), rows) || throw(ArgumentError("Rank-sweep rows do not share one reference_vmax"))

    reconstruction_slices, difference_slices, reference_slices = load_middle_slices(run_dir, rows)
    output_dir = joinpath(run_dir, "figure")

    output_paths = String[]
    for (direction_index, direction) in enumerate(DIRECTIONS)
        push!(output_paths,
            save_direction_figure(output_dir, direction, reconstruction_slices[direction_index], difference_slices[direction_index], reference_slices[direction_index], reference_vmax),
        )
    end

    println("\nSaved 3D rank-sweep middle-slice figures:")
    foreach(path -> println("  $path"), output_paths)
    SHOW_FIGURES && show()
    return output_paths
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && plot_3d_rank_mid_slices!()
