# `HighOrderKernelOp`

Explicit CUDA encoding with legacy kernels and GPU-resident direct/cuBLAS backends. One public constructor selects single- or multi-GPU execution through `gpus`.

```julia
HighOrderKernelOp(
    grid::Grid{T},
    kspha::AbstractArray{T,2},
    times::AbstractVector{T};
    fieldmap=zeros(T, grid.matrixSize...),
    csm=ones(Complex{T}, grid.matrixSize..., 1),
    mask=trues(grid.matrixSize...),
    concomitant=nothing,
    recon_terms=nothing,
    k_nominal=kspha[2:4, :],
    kspha_dt=nothing,
    nBlock=50,
    arrayType=CuArray,
    gpus=[0],
    backend=:legacy,
    block_size=320,
    phase_backend=:kernel,
    gemm_backend=:cublas,
    phase_cache_bytes=0,
    forward_threads=256,
    adjoint_threads=512,
    verbose=false,
) where T<:AbstractFloat
```

## Arguments

- `grid`: physical reconstruction grid, with `T` equal to `Float32` or `Float64`.
- `kspha`: time-integrated spherical-harmonic coefficients `(nTerm, nSam)`, with 9 or 16 rows.
- `times`: ADC sampling times.

## Keywords

| Keyword | Default | Meaning |
|:--|:--|:--|
| `fieldmap`, `csm`, `mask` | As above | Static off-resonance, complex coil sensitivities, and reconstruction mask. |
| `concomitant` | `nothing` | Optional `(4,nSam)` integrated coefficients in cycles/m², ordered as `z²`, `x²+y²`, `xz`, `yz`; see [concomitant phase](/theory/encoding-model#optional-concomitant-phase). |
| `recon_terms`, `k_nominal` | As above | Spherical-harmonic order selection and nominal trajectory. Concomitant terms are independent of `recon_terms`. |
| `gpus` | `[0]` | Unique, zero-based CUDA-visible IDs. One entry selects single-GPU execution; several select multi-GPU execution. The first owns resident solver vectors. |
| `backend` | `:legacy` | `:legacy`, `:direct`, `:blocked`, `:cached`, or `:cached_full`; see below. |
| `block_size` | `320` | Sample rows per block for resident matrix backends. |
| `phase_backend` | `:kernel` | Fused phase/exponential generation, or `:gemm` for real GEMM followed by the exponential. |
| `gemm_backend` | `:cublas` | Native cuBLAS calls; `:auto` retains generic `mul!` dispatch for comparisons. |
| `phase_cache_bytes` | `0` | Per-GPU byte budget for a complex phase prefix with `backend=:blocked`. |
| `forward_threads`, `adjoint_threads` | `256`, `512` | Direct-kernel launch settings; the Float64 adjoint is capped at 128 threads. |
| `kspha_dt` | `nothing` | Delay-estimation derivative coefficients; requires `backend=:legacy`. This path ignores `concomitant` and keeps the existing derivative model. |
| `nBlock` | `50` | Legacy derivative block count; distinct from resident `block_size`. |
| `arrayType` | `CuArray` | Must be `CuArray`. |
| `verbose` | `false` | Report progress. |

## Backends and storage

- `:legacy` preserves the original fused kernels and CPU solver storage. Multi-GPU execution partitions masked voxels.
- `:direct` retains the direct CUDA encoding kernels with resident workspaces and GPU solver storage.
- `:blocked` generates a complex phase block, applies all coils, and reuses that block for both products of the weighted normal.
- `:cached` retains the full complex phase matrix and applies it in sample blocks.
- `:cached_full` retains the same full matrix and applies one complete GEMM per forward or adjoint; it also needs a full sample-by-coil workspace.

Both full-cache modes initialize the matrix in blocks. For `ComplexF32`, the full matrix alone requires `8 * nSam * count(mask)` bytes per single-GPU problem. Include phase scratch, CSM, coil images, solver vectors, data, and library workspaces. A two-dimensional dataset does not necessarily fit.

For resident multi-GPU backends, each device stores all masked voxels and coils but only its contiguous sample range. Phase-cache feasibility must hold on each device. The resident implementation types are internal; use `HighOrderKernelOp(...; gpus=...)` for every backend.

## Partial phase caching

With `backend=:blocked`, `phase_cache_bytes` retains a prefix of complex phase blocks and regenerates the rest. The budget counts cached complex entries only, rounds down to complete blocks, and applies to each GPU independently. It excludes the other workspaces and the allocator/library reserve. A positive budget therefore does not guarantee that construction fits available memory.

The budget is explicit; the constructor does not automatically choose a cache
size from free memory. For full caching or a nonempty prefix cache, its
allocation estimate includes model, scratch, and solver allowances and must
be below `0.8 * CUDA.free_memory()` on that worker. This leaves a 20% margin
for libraries and allocator overhead, rather than an unconditional fit guarantee.

The default zero recomputes every block. Cache initialization belongs in one-shot reconstruction time. Partial caching preserves all samples, coils, mask voxels, and phase terms. It is separate from the complete-matrix `:cached` and `:cached_full` modes.

## Matrix execution

The resident matrix backends store $P_b$ as `(block_samples,nVox)`, with
entries $\exp(+i2\pi\phi_{jv})$. With `phase_backend=:gemm`, a real GEMM
evaluates the prepared dynamic coefficient/basis product; the phase kernel
adds `times * fieldmap` and evaluates the complex exponential. With
`:kernel`, that kernel also evaluates the dynamic phase sum.

For a normal application, `Z` contains the masked image multiplied by every
CSM. `explicit_gpu_normal!` computes `Y = P_b * Z`, applies the already
squared diagonal weights to `Y`, and accumulates `Q += P_b' * Y`. The same
`P_b` is reused for both products. Native cuBLAS uses transpose flags `N,N`
and `C,N`, so it does not allocate a conjugate-transposed copy of `P_b`.
After all blocks, the coil-reduction kernel multiplies by conjugated CSM,
sums coils, and applies `1/nVox`; scattering restores the full image shape.
Standalone forward and adjoint each apply `1/sqrt(nVox)`.

The `:direct` backend instead calls the resident direct forward and adjoint
kernels. It does not construct a phase matrix or use the matrix-block reuse
described above.

## Returns and lifecycle

A linear operator with `nSam * nCha` rows and `prod(grid.matrixSize)` columns. Resident input/output vectors must be `CuArray`s on `first(gpus)`; supply host model arrays at construction. Pass GPU data and weights to `recon_HOOp` to keep CG vectors on that device. The returned image is a CPU array.

Operators and derived normals share mutable workspaces and are not reentrant. Call `close(op)` after their final use. The reconstruction wrapper does not automatically close resident explicit operators.

## Weighted normal

```julia
using LinearOperatorCollection

N = explicit_gpu_normal(op; weights=weights_gpu)  # raw square-root weights
D = abs2.(weights_gpu)
N_same = normalOperator(op, D)                   # already squared WᴴW
```

`explicit_gpu_normal` accepts one raw weight per sample or per sample and coil, and squares it once; `nothing` means unit weights. `normalOperator(op, D)` expects an already squared real nonnegative diagonal. `normalOperator(W ∘ op)` supplies that diagonal automatically. Regularization remains in the solver, outside these data normal operators.

## Resident reconstruction example

```julia
using CUDA

CUDA.math_mode!(CUDA.PEDANTIC_MATH)
op = HighOrderKernelOp(grid, kspha, times;
    fieldmap, csm, mask,
    gpus=[0, 1],
    backend=:blocked,
    block_size=1024,
    phase_backend=:gemm,
    gemm_backend=:cublas,
    phase_cache_bytes=8 * 2^30,
)
try
    image = recon_HOOp(op, CuArray(ComplexF32.(data_matrix)),
        CuArray(ComplexF32.(weight_vector)), rec_params)
finally
    close(op)
end
```

This example assumes Float32 model inputs and sufficient memory on both GPUs; tune block size and cache budget on the complete problem. Device IDs are relative to `CUDA_VISIBLE_DEVICES` when set. See [Multi-GPU execution](/guide/multi-gpu#resident-explicit-encoding-sample-decomposition) for the sample decomposition.

[Source: `HighOrderKernelOp.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/main/src/EncodingOperator/HighOrderKernelOp.jl)
