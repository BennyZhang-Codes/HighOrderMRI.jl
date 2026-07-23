using LinearOperatorCollection

include("HighOrderLowRank/test_utils.jl")
include("HighOrderLowRank/test_lifecycle.jl")
include("HighOrderLowRank/test_rsvd.jl")
include("HighOrderLowRank/test_cuda_kernels.jl")
include("HighOrderLowRank/test_distributed_rsvd.jl")
include("HighOrderLowRank/test_operator.jl")
include("HighOrderLowRank/test_multi_gpu_normal_operator.jl")
