"""Sweep local rank and seed on the selected 0.5-mm spiral/EPI acquisition."""

include(joinpath(@__DIR__, "..", "common", "highres_input.jl"))
prepare_highres_benchmark!()
get!(ENV, "HIGHORDER_BENCHMARK_SHARED_BASIS_TOL", "0")
get!(ENV, "HIGHORDER_BENCHMARK_SHARED_RANK_MAX", "120")
get!(ENV, "HIGHORDER_SWEEP_RANKS", "10,20,25,40,60,80,100,120")
get!(ENV, "HIGHORDER_SWEEP_SEEDS", "1234,1235,1236,1237,1238")
include(joinpath(@__DIR__, "sweep_2d_lowrank_parameters.jl"))

abspath(PROGRAM_FILE) == abspath(@__FILE__) && run_parameter_sweep!()
