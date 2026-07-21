abstract type HOOp{T} <: AbstractLinearOperator{T} end

include("prep_kspha.jl")

# the formal version HighOrderOp, the extended signal model with field dynamics and off-resonance
include("Explicit/CUDA_kernel.jl")   # Kernel functions for CUDA
include("HighOrderOp.jl")            # Array-based implementation
include("HighOrderOp_Kernel.jl")     # Kernel-based implementation


include("LowRankApproximation/perform_rsvd.jl")          # randomized SVD for low-rank approximation
include("LowRankApproximation/shared_spatial_basis.jl")
include("LowRankApproximation/streaming_nfft_plan.jl")
include("HighOrderLowRankOp.jl")    # low rank approximation of temporal-spatial varing phase
