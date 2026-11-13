"""Run the fixed 2D comparison on the selected 0.5-mm spiral/EPI acquisition."""

include(joinpath(@__DIR__, "..", "common", "highres_input.jl"))
prepare_highres_benchmark!()
include(joinpath(@__DIR__, "run_2d_kernel_vs_lowrank.jl"))

abspath(PROGRAM_FILE) == abspath(@__FILE__) && run_benchmark!()
