# Encoding operators

HighOrderMRI.jl provides three linear-operator implementations of the field-aware signal model. The appropriate implementation depends on whether the calculation requires an explicit reference, fused CUDA execution, or repeated large-scale evaluations using the low-rank approximation.

## Implementations

| | `HighOrderOp` | `HighOrderKernelOp` | `HighOrderLowRankOp` |
|---|---|---|---|
| Signal model | Explicit | Explicit | Residual-phase approximation |
| Dynamics per object | One | One | One or many |
| CPU | Yes | No | Yes |
| CUDA | Array operations | Fused kernels | NFFT + chunked/fused rSVD |
| Multi-GPU | No | Voxel-sharded explicit evaluation | Optional voxel-sharded setup and channel-sharded normal operator |
| Dominant numerical cost | Sample–voxel phase blocks | Repeated sample–voxel phase evaluation | Setup plus $RN_c$ NFFTs per forward/adjoint |
| Typical numerical role | Small explicit calculations and derivative products | Explicit GPU consistency reference | Repeated large-scale forward/adjoint evaluations |

The explicit operators target the same signal model. `HighOrderLowRankOp` changes only the numerical representation of the residual spatial phase term; the first-order Fourier trajectory remains in the NFFT.

## Common inputs

For a grid of size `(nX,nY,nZ)`:

- `fieldmap`: `(nX,nY,nZ)`, or `(nX,nY)` when `nZ == 1`;
- `csm`: `(nX,nY,nZ,nCha)`, or `(nX,nY,nCha)` when `nZ == 1`;
- `mask`: the spatial grid shape;
- `k_nominal`: first-order coefficients ordered as `kx`, `ky`, `kz`;
- `recon_terms`: order-selection string defined in the [expanded encoding model](../theory/encoding-model.md#selecting-field-orders).

The operator column count is always `prod(grid.matrixSize)`. A mask removes background voxels from the encoding calculation, while the adjoint restores unencoded voxels as zeros in the full image vector.

## Coil compression

All three operators accept compressed data and compressed CSM through the same interface. Fit one `CoilCompressionTransform` and apply the same right-side transform to the data and sensitivity maps:

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

The same `csm_cc` can be supplied to `HighOrderKernelOp` or `HighOrderLowRankOp`. Using one transform for both acquired signals and coil sensitivities preserves the transformed signal equation; independently fitted transforms generally do not.

When representative noise-only samples are available, estimate

```julia
noise_covariance = estimate_noise_covariance(noise_data; coil_dim=...)
```

before fitting the transform. Omitting the covariance corresponds to an independent, equal-variance noise model. The whitening and compression equations are given in [Coil compression](/guide/coil-compression).

## `HighOrderOp`: array-based explicit evaluation

A CPU explicit operator can be constructed with:

```julia
op = HighOrderOp(
    grid,
    kspha,
    times;
    fieldmap,
    csm,
    mask,
    recon_terms="111",
    k_nominal=kspha[2:4, :],
    nBlock=32,
    arrayType=Array,
)
```

Array-based CUDA execution is selected with:

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

`nBlock` divides the samples into explicit phase blocks. Increasing the number of blocks reduces peak temporary memory at the cost of additional launches and allocations.

Supplying `kspha_dt` changes the forward action to the time-derivative product $Bx$ used internally by `FindDelay`. The adjoint remains the ordinary encoding adjoint. An operator constructed with `kspha_dt` should therefore not be interpreted as a general derivative linear operator, and its forward/adjoint pair is not intended for an adjointness test.

## `HighOrderKernelOp`: fused explicit CUDA evaluation

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

Masked voxels are divided into approximately equal contiguous ranges. Each GPU evaluates a partial forward signal; these partial signals are summed on the host. For the adjoint, disjoint voxel results are gathered into their original masked positions. The decomposition changes data placement and reduction order but not the explicit phase model.

For a controlled comparison with another implementation, the grid, mask, field coefficients, coil maps, normalization, and reconstruction settings should be identical.

When `kspha_dt` is supplied, the current derivative forward path falls back to the array-based CPU derivative implementation. Its adjoint remains the ordinary kernel encoding adjoint. Use `HighOrderOp(...; arrayType=CuArray, kspha_dt=...)` when the derivative forward product itself must execute on CUDA.

## `HighOrderLowRankOp`: shared spatial representation

The dynamic constructor accepts `(nTerm,nSam,nDyn)` field coefficients and `(nSam,nDyn)` sampling times:

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

A two-dimensional `kspha` array and vector `times` are accepted by a convenience overload that inserts a singleton dynamic dimension.

For single-GPU CUDA execution:

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

For the fused CUDA rSVD backend, the randomized sketch width must satisfy

$$
L+p\leq 32,
$$

where $L$ is `L_rank` and $p$ is `rsvd_oversample`. This implementation limit does not constrain the final shared spatial rank. The meaning of the local rank, shared rank, and incremental coefficients is derived in [Low-rank shared subspace](/theory/low-rank).

## Numerical consistency checks

For random complex vectors `x` and `y`, the adjoint identity can be evaluated as:

```julia
using LinearAlgebra, Random

Random.seed!(2026)
x = randn(ComplexF32, size(op, 2))
y = randn(ComplexF32, size(op, 1))

adjoint_error =
    abs(dot(op * x, y) - dot(x, op' * y)) /
    max(abs(dot(op * x, y)), eps(Float32))
```

When evaluating a low-rank operator against an explicit implementation, report the raw forward, adjoint, and normal-operator relative errors before relying on image-domain metrics alone. The complete comparison hierarchy is described in [Scientific validation strategy](/guide/validation).
