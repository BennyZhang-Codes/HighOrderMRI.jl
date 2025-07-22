using LinearOperators
using MRIOperators
abstract type HOOp{T} <: AbstractLinearOperator{T} end
abstract type fHOOp{T} <: AbstractLinearOperator{T} end

# the formal version HighOrderOp, the extended signal model with field dynamics and off-resonance
include("HighOrderOp.jl")
include("HighOrderOp3D.jl")
include("fastHigherOrderOp.jl")