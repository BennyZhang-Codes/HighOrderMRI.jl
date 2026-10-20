using Test
using HighOrderMRI
using LinearAlgebra

include("test_HighOrderLowRank.jl")
include("test_HighOrderExplicit.jl")
include("test_explicit_gpu.jl")
include("test_explicit_multi_gpu.jl")
include("test_explicit_phase_cache.jl")
include("test_kernel_interface.jl")
include("test_CoilCompression.jl")
include("test_ImageMetrics.jl")
