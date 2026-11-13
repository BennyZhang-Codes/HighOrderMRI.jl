"""Plot an explicit result path or the newest completed 0.5-mm sweep."""

get!(ENV, "HIGHORDER_SWEEP_NAME_FILTER", "_0p5_")
include(joinpath(@__DIR__, "plot_2d_lowrank_sweep.jl"))

abspath(PROGRAM_FILE) == abspath(@__FILE__) && visualize_sweep!()
