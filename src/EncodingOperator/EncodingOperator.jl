abstract type HOOp{T} <: AbstractLinearOperator{T} end

include("prep_kspha.jl")

# Explicit family and its Array/CUDA implementations.
include("HighOrderExplicitOp.jl")
include("Explicit/CUDA_kernel.jl")   # Kernel functions for CUDA
include("Explicit/array_operator.jl")            # Array-based implementation
include("Explicit/kernel_operator.jl")      # Kernel-based implementation
include("Explicit/explicit_gpu_operator.jl")


include("LowRank/CUDA_kernel.jl")   # Kernel functions for CUDA
include("LowRank/perform_rsvd.jl")          # randomized SVD for low-rank approximation
include("LowRank/distributed_gpu_worker.jl")
include("LowRank/distributed_rsvd.jl")      # distributed randomized SVD for low-rank approximation
include("LowRank/distributed_factor_gather.jl")
include("LowRank/shared_spatial_basis.jl")
include("LowRank/joint_spatial_basis.jl")
include("LowRank/joint_adaptive_spatial_basis.jl")
include("LowRank/distributed_shared_spatial_basis.jl")
include("LowRank/shared_basis_sampling.jl")
include("LowRank/regenerated_spatial_basis.jl")
include("LowRank/per_readout_basis.jl")
include("LowRank/per_readout_cache.jl")
include("LowRank/streaming_nfft_plan.jl")
include("HighOrderLowRankOp.jl")    # LowRank family construction and products
include("LowRank/batched_normal_operator.jl")
include("LowRank/multi_gpu_normal_operator.jl")
include("LowRank/rank_sharded_shared.jl")
include("Explicit/multi_gpu_operator.jl")
