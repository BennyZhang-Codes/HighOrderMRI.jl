abstract type HOOp{T} <: AbstractLinearOperator{T} end

include("prep_kspha.jl")

# the formal version HighOrderOp, the extended signal model with field dynamics and off-resonance
include("HighOrderOp.jl")         # Array-based implementation

include("CUDA_kernel.jl")         # Kernel functions for CUDA
include("HighOrderOp_Kernel.jl")  # Kernel-based implementation


include("perform_rsvd.jl")          # randomized SVD for low-rank approximation
include("HighOrderLowRankOp.jl")    # low rank approximation of temporal-spatial varing phase (NFFT-based implementation)