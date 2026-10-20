# Choose an operator

All HighOrderMRI encoding operators are linear operators with image-vector
columns and sample-vector rows. Select the implementation from the problem
size, required reference accuracy, and available hardware.

## Comparison

| | `HighOrderOp` | `HighOrderKernelOp` | `HighOrderLowRankOp` |
|---|---|---|---|
| Target model | Explicit | Explicit | Approximation of residual phase |
| Dynamics per object | One | One | One or many |
| CPU | Yes | No | Yes |
| CUDA | Array operations | Fused kernels | NFFT + fused/chunked rSVD |
| Multi-GPU | No | Voxel-sharded explicit evaluation | Optional voxel-sharded setup and channel-sharded normal operator |
| Main cost | Sample–voxel phase matrix blocks | Repeated sample–voxel phase evaluation | Setup plus ``R N_c`` NFFTs per forward/adjoint |
| Recommended role | Debugging and small CPU/GPU work | Explicit numerical reference | Large iterative reconstruction |

For iterative explicit reconstruction, [`HighOrderExplicitGPUOp`](@ref) and
[`HighOrderExplicitMultiGPUOp`](@ref) also provide persistent GPU workspaces
and GPU solver storage. They support direct kernels, cuBLAS sample blocks,
and complete complex-phase caching. Select a backend from measurements at
the actual voxel, sample, and coil counts; the low-rank path's speed advantage
depends on the problem and the explicit reference used.

## Shared inputs

For a grid of size `(nX,nY,nZ)`:

- `fieldmap`: `(nX,nY,nZ)` or `(nX,nY)` when `nZ == 1`;
- `csm`: `(nX,nY,nZ,nCha)` or `(nX,nY,nCha)` when `nZ == 1`;
- `mask`: the spatial grid shape;
- `k_nominal`: first-order coefficients ordered as `kx`, `ky`, `kz`;
- `recon_terms`: order-selection string described in the [encoding
  model](../theory/encoding-model.md).

The operator column count is always `prod(grid.matrixSize)`, even when a mask
is used. Values outside the mask are ignored by the encoding and restored as
zeros by the adjoint.

## Shared coil compression

All encoding operators accept compressed data and CSM without an
operator-specific interface. First fit a [`CoilCompressionTransform`](@ref),
then apply the same matrix along the data and CSM coil dimensions:

```julia
data_cc, csm_cc, transform = compress_coils(
    data,
    csm;
    data_coil_dim=2,
    n_virtual_coils=10,
    noise_covariance,
)

op = HighOrderOp(grid, kspha, times; csm=csm_cc, arrayType=Array)
image = recon_HOOp(op, data_cc, weights, rec_params)
```

The same `csm_cc` can instead be supplied to `HighOrderKernelOp` or
`HighOrderLowRankOp`. Compression preserves the signal equation because a
single right-side coil transform is applied to both measured signals and
sensitivity maps. Do not independently fit or phase-align those two
transforms.

When representative noise-only samples are available, obtain
`noise_covariance = estimate_noise_covariance(noise_data; coil_dim=...)` and
pass it as shown. This makes the retained global subspace noise-whitened; an
omitted covariance assumes independent, equal-variance channels.

## Array-based explicit operator

Use `arrayType=Array` for a CPU reference:

```julia
op = HighOrderOp(
    grid,
    kspha,  # (nTerm, nSam)
    times;  # (nSam,)
    fieldmap,
    csm,
    mask,
    recon_terms="111",
    k_nominal=kspha[2:4, :],
    nBlock=32,
    arrayType=Array,
)
```

Use `arrayType=CuArray` for array-based CUDA execution:

```julia
using CUDA

op = HighOrderOp(
    grid,
    kspha,
    times;
    fieldmap,
    csm,
    mask,
    arrayType=CuArray,
)
```

`nBlock` divides samples into smaller explicit phase blocks. More blocks
reduce peak temporary memory but add launch and allocation overhead.

Passing `kspha_dt` changes the forward action to the time-derivative product
``B x`` used internally by [`FindDelay`](@ref). The current adjoint action is
still the ordinary encoding adjoint; consequently, an operator constructed
with `kspha_dt` must not be treated as a general linear derivative operator or
used in an adjointness test. The legacy synchronization routine uses only its
forward action.

## Explicit CUDA kernel

`HighOrderKernelOp` requires CUDA:

```julia
using CUDA

reference_op = HighOrderKernelOp(
    grid,
    kspha,
    times;
    fieldmap,
    csm,
    mask,
    recon_terms="111",
    arrayType=CuArray,
    gpus=[0, 1],
)
```

Masked voxels are split into approximately equal contiguous ranges. Every GPU
produces a partial forward signal, which is summed on the host; the adjoint
collects the disjoint voxel shards. This changes data placement, not the
mathematical model.

Use this operator as a numerical reference only after checking the same grid,
mask, field coefficients, coil maps, normalization, and reconstruction
settings as the candidate operator.

The current `kspha_dt` forward path of `HighOrderKernelOp` falls back to the
array-based CPU derivative implementation. Its adjoint retains the ordinary
kernel encoding action described above. Use
`HighOrderOp(...; arrayType=CuArray, kspha_dt=...)` when the derivative forward
product must run on CUDA. The fused kernel path currently accelerates only the
ordinary explicit encoding.

## GPU-resident explicit reconstruction

Construct spatial model inputs on the host, then keep data, weights, and
solver vectors on the GPU selected during construction:

```julia
using CUDA

op = HighOrderExplicitGPUOp(grid, kspha, times;
    fieldmap, csm, mask,
    gpus=[0],
    backend=:blocked,
    phase_backend=:gemm,
    gemm_backend=:cublas,
    block_size=1024,
)
try
    image = recon_HOOp(op, CuArray(ComplexF32.(data_matrix)),
        CuArray(ComplexF32.(weight_vector)), rec_params)
finally
    close(op)
end
```

Use model and data precision consistently (`Float32`/`ComplexF32` above).
`gpus=[id]` selects one CUDA device; omitting it uses the current device.
IDs are zero-based and relative to `CUDA_VISIBLE_DEVICES` when that variable
filters the device list. Use `HighOrderExplicitMultiGPUOp(...; gpus=[...])`
to select multiple GPUs; its first ID owns solver vectors.
`block_size` counts sample rows, whereas the older `nBlock` keyword counts
blocks. The operator's specialized `normalOperator` generates each complex
phase block once and reuses it for all coils and both matrix products.

`backend=:direct` reuses the existing CUDA kernels. `:cached` retains the full
complex ``P`` but applies it in sample blocks; `:cached_full` applies the same
cached matrix with one full GEMM per forward or adjoint. Cache initialization
is blocked in both modes. Include ``8N_tN_v`` bytes for ComplexF32 ``P`` plus
the real phase block, model, coil buffers, solver, and library workspace when
checking feasibility. Cached setup belongs in one-shot reconstruction timing.

[`explicit_gpu_normal`](@ref) accepts raw weights and squares them once.
`normalOperator(op, D)` expects the already squared diagonal ``D=W^H W``;
`normalOperator(W ∘ op)` supplies it automatically. Regularization remains in
the solver. Operators and their derived normals share mutable workspaces
and must be used sequentially; close the operator after the final use.

For sample-distributed execution on several GPUs, use
[`HighOrderExplicitMultiGPUOp`](@ref) as described in
[Multi-GPU execution](multi-gpu.md#GPU-resident-sample-distributed-explicit-operator).

## Low-rank operator

The dynamic constructor accepts `(nTerm,nSam,nDyn)` coefficients and
`(nSam,nDyn)` times:

```julia
lowrank_op = HighOrderLowRankOp(
    grid,
    kspha_dynamic,
    times_dynamic;
    fieldmap,
    csm,
    mask,
    recon_terms="111",
    k_nominal=kspha_dynamic[2:4, :, :],
    arrayType=Array,
    L_rank=8,
    rsvd_oversample=5,
    rsvd_seed=1234,
    rsvd_backend=:chunked,
    rsvd_finalize=:svd,
    shared_basis_tol=1f-2,
    shared_rank_max=64,
)
```

A two-dimensional `kspha` and vector `times` are accepted by a convenience
overload that inserts a singleton dynamic dimension.

For single-GPU CUDA:

```julia
lowrank_op = HighOrderLowRankOp(
    grid,
    kspha_dynamic,
    times_dynamic;
    fieldmap,
    csm,
    mask,
    arrayType=CuArray,
    gpus=[0],
    L_rank=10,
    rsvd_backend=:kernel,
    rsvd_finalize=:gram,
)
```

The fused CUDA rSVD backend requires
`L_rank + rsvd_oversample <= 32`. The final shared rank is not subject to this
kernel-width limit.

## Validate an operator

For random complex vectors `x` and `y`, check adjointness:

```julia
using LinearAlgebra, Random

Random.seed!(2026)
x = randn(ComplexF32, size(op, 2))
y = randn(ComplexF32, size(op, 1))

adjoint_error =
    abs(dot(op * x, y) - dot(x, op' * y)) /
    max(abs(dot(op * x, y)), eps(Float32))
```

When comparing a low-rank operator to an explicit reference, report raw
forward, adjoint, and normal-operator relative errors before reporting
image-only metrics.
