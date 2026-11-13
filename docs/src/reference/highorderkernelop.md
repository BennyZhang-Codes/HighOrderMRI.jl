# `HighOrderKernelOp`

Use this constructor for explicit CUDA encoding. Legacy mode partitions voxels
and keeps solver vectors on the host. Resident modes partition sample rows
and keep solver vectors on the primary GPU. Read the
[operator guide](/guide/operators) for the choice of implementation.

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

## Complete resident example

This synthetic example requires one CUDA GPU. It computes a weighted normal
product, not an image reconstruction or a performance measurement.

```julia
using HighOrderMRI, CUDA

CUDA.functional() || error("CUDA is not functional")
CUDA.device!(0)
grid = Grid(8, 8, 1, 1f-3, 1f-3, 5f-3)
kspha = zeros(Float32, 9, 32)
kspha[2, :] .= range(-200f0, 200f0; length=32)
times = collect(range(0f0, 3f-3; length=32))
op = HighOrderKernelOp(grid, kspha, times;
    gpus=[0], backend=:blocked, block_size=16, phase_cache_bytes=0)
image = try
    x = CUDA.ones(ComplexF32, prod(grid.matrixSize))
    weights = CUDA.ones(ComplexF32, size(op, 1))
    normal = explicit_gpu_normal(op; weights)
    Array(reshape(normal * x, grid.matrixSize))
finally
    close(op)
end
```

For reconstruction, pass GPU data and weights to
[`recon_HOOp`](/reference/recon-hoop). The default legacy Float64 adjoint can
exceed its fixed CUDA launch's register budget. The resident direct Float64
cap does not apply to that legacy path. Derivative mode remains forward-only
in the sense explained in the generated constructor reference.

## Constructor API

<!--@include: ../../.generated/api/highorderkernelop.md-->

## Weighted normal API

<!--@include: ../../.generated/api/explicit-gpu-normal.md-->
