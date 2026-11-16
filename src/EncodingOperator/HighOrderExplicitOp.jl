export HighOrderExplicitOp

"""
    HighOrderExplicitOp

Explicit MRI encoding operator with Array or CUDA execution. Construct it
with `HighOrderExplicitOp(grid, kspha, times; implementation, kwargs...)`.
Concrete implementation types are internal. All implementations use the same
positive phase, mask, sample/coil ordering and voxel normalization.
"""
abstract type HighOrderExplicitOp{T} <: HOOp{T} end

"""
    HighOrderExplicitOp(grid::Grid{T}, kspha::AbstractMatrix{T},
        times::AbstractVector{T}; implementation::Symbol=:array,
        kwargs...) where T<:AbstractFloat

Construct an Explicit encoding operator with `implementation=:array` or
`:kernel`. `:array` uses array operations on CPU or CUDA; `:kernel` uses
direct CUDA kernels or resident matrix execution. The returned operator is
an instance of `HighOrderExplicitOp`.

The default is `implementation=:array, arrayType=CuArray`. Pass
`arrayType=Array` for CPU execution. The CUDA implementation requires
`CuArray`; its default `backend=:legacy` computes on CUDA with host solver
vectors. `backend=:direct`, `:blocked`, `:cached`, and `:cached_full` keep
solver vectors on the primary GPU. Select participating GPUs with `gpus`.

With `backend=:blocked`, `phase_backend=:gemm, gemm_backend=:cublas`
generates the dynamic phase by real GEMM and reuses each complex phase block
across coils and both normal products. `phase_cache_bytes=0` disables the
optional prefix cache. Matrix execution introduces no low-rank truncation.

Inputs contain one flattened sample axis: `kspha` is `(9,N)` or `(16,N)`
and `times` has length N. For multiple readouts of one image, flatten samples
before readouts in both model inputs and data. Weights are passed to
`recon_HOOp` or the normal operator, not this constructor.

Physical, block, precision and execution options are documented in the
[Explicit guide](/guide/explicit) and implementation option pages. Call
`close(op)` after the final use; CUDA normals share their parent's mutable
workspaces. Array operators own no persistent execution resources.
"""
function HighOrderExplicitOp(
    grid::Grid{T}, kspha::AbstractMatrix{T}, times::AbstractVector{T};
    implementation::Symbol=:array, kwargs...,
) where T<:AbstractFloat
    implementation === :array && return ArrayExplicitOp(grid, kspha, times; kwargs...)
    implementation === :kernel && return CUDAExplicitOp(grid, kspha, times; kwargs...)
    throw(ArgumentError("implementation must be :array or :kernel"))
end
