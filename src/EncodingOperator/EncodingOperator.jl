abstract type HOOp{T} <: AbstractLinearOperator{T} end

include("prep_kspha.jl")

# the formal version HighOrderOp, the extended signal model with field dynamics and off-resonance
include("Explicit/CUDA_kernel.jl")   # Kernel functions for CUDA
include("HighOrderOp.jl")            # Array-based implementation
include("HighOrderKernelOp.jl")      # Kernel-based implementation
include("Explicit/explicit_gpu_operator.jl")


include("LowRankApproximation/CUDA_kernel.jl")   # Kernel functions for CUDA
include("LowRankApproximation/perform_rsvd.jl")          # randomized SVD for low-rank approximation
include("LowRankApproximation/distributed_gpu_worker.jl")
include("LowRankApproximation/distributed_rsvd.jl")      # distributed randomized SVD for low-rank approximation
include("LowRankApproximation/distributed_factor_gather.jl")
include("LowRankApproximation/shared_spatial_basis.jl")
include("LowRankApproximation/joint_spatial_basis.jl")
include("LowRankApproximation/distributed_shared_spatial_basis.jl")
include("LowRankApproximation/regenerated_spatial_basis.jl")
include("LowRankApproximation/per_readout_basis.jl")
include("LowRankApproximation/per_readout_cache.jl")
include("LowRankApproximation/streaming_nfft_plan.jl")
include("HighOrderLowRankOp.jl")    # low rank approximation of temporal-spatial varing phase
include("LowRankApproximation/multi_gpu_normal_operator.jl")
include("LowRankApproximation/rank_sharded_shared.jl")
include("Explicit/multi_gpu_operator.jl")
