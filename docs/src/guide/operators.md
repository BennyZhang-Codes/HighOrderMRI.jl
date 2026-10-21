# Encoding operators

HighOrderMRI.jl provides three linear-operator implementations of the field-aware signal model. The appropriate implementation depends on whether the calculation requires an explicit reference, resident CUDA/cuBLAS execution, or repeated large-scale evaluations using the low-rank approximation.

## Implementations

| | `HighOrderOp` | `HighOrderKernelOp` | `HighOrderLowRankOp` |
|---|---|---|---|
| Signal model | Explicit | Explicit | Residual-phase approximation |
| Dynamics per object | One | One | One or many |
| CPU | Yes | No | Yes |
| CUDA | Array operations | Direct kernels or resident cuBLAS | NFFT + rSVD or joint setup |
| Multi-GPU | No | Legacy voxel shards or resident sample shards | Voxel-sharded rSVD; channel- or readout-sharded normal |
| Dominant numerical cost | Sample–voxel phase blocks | Repeated sample–voxel phase evaluation | Setup plus $RN_c$ NFFTs for shared or $N_dLN_c$ for per-readout |
| Typical numerical role | Small explicit calculations and derivative products | Explicit GPU reference and iterative reconstruction | Repeated large-scale forward/adjoint evaluations |

The explicit operators target the same signal model. `HighOrderLowRankOp` changes only the numerical representation of the residual spatial phase term; the first-order Fourier trajectory remains in the NFFT.

## Common inputs

For a grid of size `(nX,nY,nZ)`:

- `fieldmap`: `(nX,nY,nZ)`, or `(nX,nY)` when `nZ == 1`;
- `csm`: `(nX,nY,nZ,nCha)`, or `(nX,nY,nCha)` when `nZ == 1`;
- `mask`: the spatial grid shape;
- `k_nominal`: first-order coefficients ordered as `kx`, `ky`, `kz`;
- `recon_terms`: order-selection string defined in the [expanded encoding model](../theory/encoding-model.md#selecting-field-orders);
- `concomitant`: optional integrated coefficients `(4,nSam)` for the explicit/single-dynamic operators or `(4,nSam,nDyn)` for dynamic LowRank, ordered as `z²`, `x²+y²`, `xz`, `yz`. Supply cycles/m² in the grid frame and origin; see [concomitant phase](/theory/encoding-model#optional-concomitant-phase).

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

## `HighOrderKernelOp`: explicit CUDA evaluation

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

With the default `backend=:legacy`, masked voxels are divided into approximately equal contiguous ranges. Each GPU evaluates a partial forward signal; these partial signals are summed on the host. For the adjoint, disjoint voxel results are gathered into their original masked positions. Resident backends instead partition samples across GPUs, as described below. Both decompositions preserve the explicit phase model.

For a controlled comparison with another implementation, the grid, mask, field coefficients, coil maps, normalization, and reconstruction settings should be identical.

When `kspha_dt` is supplied, the current derivative forward path falls back to the array-based CPU derivative implementation. Its adjoint remains the ordinary kernel encoding adjoint. Use `HighOrderOp(...; arrayType=CuArray, kspha_dt=...)` when the derivative forward product itself must execute on CUDA. Derivative mode requires `backend=:legacy` and ignores `concomitant`; ordinary legacy and resident encoding support supplied concomitant phase.

## GPU-resident explicit reconstruction

Use the same public `HighOrderKernelOp` with `backend=:direct`, `:blocked`, `:cached`, or `:cached_full`. `gpus=[id]` selects one GPU and multiple IDs select the internal multi-GPU implementation. The default remains `backend=:legacy`; choose a resident backend explicitly.

```julia
using CUDA

CUDA.math_mode!(CUDA.PEDANTIC_MATH)
op = HighOrderKernelOp(grid, kspha, times;
    fieldmap, csm, mask,
    gpus=[0, 1],
    backend=:blocked,
    phase_backend=:gemm,
    gemm_backend=:cublas,
    block_size=1024,
    phase_cache_bytes=8 * 2^30,
)
try
    image = recon_HOOp(op, CuArray(ComplexF32.(data_matrix)),
        CuArray(ComplexF32.(weight_vector)), rec_params)
finally
    close(op)
end
```

Use Float32 model inputs with ComplexF32 data, or matching Float64/ComplexF64 inputs. Construct the model from host arrays. The first selected GPU owns the data, weights, and solver vectors; the CPU receives only the returned image. IDs are zero-based and relative to `CUDA_VISIBLE_DEVICES` when the visible list is filtered.

`block_size` counts sample rows, whereas legacy `nBlock` counts blocks. A blocked normal generates each complex phase block once and reuses it for all coils, forward multiplication, weighting, and adjoint multiplication. `:direct` uses direct encoding kernels; `:cached` stores the complete phase matrix but applies it in blocks; `:cached_full` uses one full GEMM per forward/adjoint. The two full-cache modes initialize in blocks.

For ComplexF32, the full complex matrix requires `8 * nSam * count(mask)` bytes, before phase scratch, CSM, coil images, data, solver, and library workspace. Feasibility must hold on each participating GPU. Include cache initialization and all required transfers in complete reconstruction timing.

With `backend=:blocked`, `phase_cache_bytes` limits a cached prefix on each GPU separately; zero is the default. The budget counts only complex phase entries and rounds down to complete sample blocks. Remaining samples are regenerated. It does not reduce the problem or approximate the phase, and excludes other workspaces. Tune the budget and block size together on the complete problem.

`explicit_gpu_normal(op; weights)` accepts raw square-root weights and squares them once. `normalOperator(op, D)` instead accepts the already squared diagonal `D=WᴴW`; `normalOperator(W ∘ op)` supplies that diagonal automatically. Regularization remains in the solver. See the [API](/reference/highorderkernelop#weighted-normal).

Operators and their normals share mutable workspaces. Use them sequentially and call `close(op)` after the final use, including after `recon_HOOp`. For multi-GPU sample partitioning, see [Multi-GPU execution](/guide/multi-gpu#resident-explicit-encoding-sample-decomposition).

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

The fused CUDA backend handles wide sketches by batching columns in groups
of 16; a sketch of at most
16 columns still uses one launch per product. Matrix-dimension limits still
apply. The final shared rank is separate from the sketch width.

### Independent per-readout factors

Use `basis_layout=:per_readout` for a local singular-vector baseline without
shared-space compression. For a factor bank that fits host RAM but exceeds
device memory:

```julia
lowrank_op = HighOrderLowRankOp(grid, kspha_dynamic, times_dynamic;
    fieldmap, csm, mask,
    arrayType=CuArray,
    gpus=[0, 1],
    basis_layout=:per_readout,
    L_rank=10,
    rsvd_oversample=5,
    rsvd_backend=:kernel,
    rsvd_finalize=:gram,
    rsvd_distribution=:voxel,
    per_readout_storage=:cpu,
    per_readout_cache_size=5,
    normal_distribution=:readout,
)
```

Each readout retains its local rank-$L$ temporal and spatial factors. The
bounded GPU queue streams factors ahead of use; the complete spatial bank
still occupies host RAM. If that bank also exceeds host memory, change
`per_readout_storage` to `:regenerate`. This keeps Q/Z and generates spatial
factors into bounded GPU slots, with `per_readout_host_cache_bytes=0` by
default. Lookahead can overlap work but does not remove regeneration cost.

For an approximation comparison, first vary only `basis_layout`, holding
the local rank, seed, field model, precision, weights, and solver fixed.
Then measure storage and scheduling choices separately. Shared compression
parameters are inactive in the per-readout layout. See the
[parameter reference](/reference/highorderlowrankop#per-readout-parameters),
[factorization](/theory/low-rank#independent-per-readout-representation), and
[readout-distributed normal](/guide/multi-gpu#iterative-reconstruction-readout-distributed-normal-operator).

### Automatic joint shared basis

The experimental `shared_basis_method=:joint` option builds one spatial basis
from representative phase snapshots across dynamics. It automatically selects
the final shared rank under sampled error checks, bypassing local rSVD.
See the [API reference](/reference/highorderlowrankop#shared-basis-parameters) for parameters
and defaults, and [theory](/theory/low-rank#direct-joint-shared-basis) for the
snapshot, fitting, and audit stages.

The following illustrates joint construction, with input arrays already
prepared using the usual units and layouts:

```julia
using CUDA

report = Ref{Any}()
lowrank_op = HighOrderLowRankOp(
    grid,
    kspha_dynamic,
    times_dynamic;
    fieldmap,
    csm,
    mask,
    arrayType=CuArray,
    gpus=[2, 3, 4, 5, 6],
    shared_basis_method=:joint,
    shared_basis_tol=1f-2,
    shared_rank_max=32,
    joint_snapshots=128,
    joint_samples=512,
    joint_basis_report=report,
    global_basis_tol=nothing,
    rsvd_distribution=:single,
    rsvd_chunk=32768,
    rsvd_seed=1234,
    normal_distribution=:channel,
)

report[].rank
maximum(report[].audit_errors)
maximum(report[].roi_errors)
```

Device IDs above assume those devices are visible without remapping. Tolerance
literals use Float32 to match Float32 inputs; use matching Float64 values for
Float64 inputs. This configuration selected $R=16$ on the frozen 3D dataset;
it is not a universal rank prescription.

For more demanding phase histories, increase `joint_snapshots` and
`shared_rank_max` together as needed, then inspect the returned errors and
memory cost. A larger budget does not guarantee a smaller final rank.
Joint setup must keep `global_basis_tol=nothing` because recompressing the
factors would invalidate their audit. That option remains available for rSVD.

::: warning Random-sample acceptance is not ROI acceptance
The main 3D and EPI joint validations passed the sampled random-voxel 1%
tolerance, but their sampled high-$|B_0|$ errors were 4.03% and 141.47%.
Add `joint_roi_tol=1f-2` when that ROI must also pass 1%; these recorded
configurations would then be rejected. The setting adds an acceptance gate,
not a new sampling or basis-construction strategy.
:::

The existing default remains rSVD. See the
[performance results](/guide/performance#representative-measurements) for
the cost and accuracy of complete reconstructions.

## Implementation paths

The public constructors select the following paths in the current implementation:

| Entry point or setting | Implementation path |
|:--|:--|
| Common field preparation | `basisfunc_spha` builds spatial columns; `prep_kspha` selects the original harmonic rows; `prep_concomitant` appends optional coefficient rows and spatial columns. |
| `HighOrderKernelOp`, `backend=:legacy` | `prod_HighOrderKernelOp` and `ctprod_HighOrderKernelOp` submit the original CUDA kernels. |
| Resident `HighOrderKernelOp` | One GPU constructs the internal `ExplicitGPUOp`; multiple GPUs construct `ExplicitMultiGPUOp`. Both remain behind the public `HighOrderKernelOp(...; gpus=...)` interface. |
| Resident matrix normal | `explicit_gpu_normal!` reuses each phase block for forward and adjoint GEMMs; `explicit_gemm!` calls native cuBLAS when `gemm_backend=:cublas`. Multi-GPU workers evaluate sample shards and reduce image contributions. |
| `HighOrderLowRankOp`, shared rSVD | Local rSVD products feed incremental shared-basis updates; the resulting `q` and `basis` drive NFFT forward/adjoint products. |
| `HighOrderLowRankOp`, per-readout | `per_readout_basis` skips shared compression. `PerReadoutCache` supplies CPU-stored or regenerated factors to the NFFT loop. |
| Distributed LowRank normal | `normalOperator` calls `ensure_highorder_normal_backend!` lazily. `local_highorder_normal!` selects shared, resident per-readout, or streamed per-readout evaluation. |
| `recon_HOOp` | Forms `E = W ∘ HOOp`, requests `normalOperator(E)`, constructs the configured solver, and solves against weighted data. |

These helper types and functions describe internal dispatch, not additional
public operator names. Source entry points are
[`HighOrderKernelOp.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/main/src/EncodingOperator/HighOrderKernelOp.jl),
[`explicit_gpu_operator.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/main/src/EncodingOperator/Explicit/explicit_gpu_operator.jl),
[`HighOrderLowRankOp.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/main/src/EncodingOperator/HighOrderLowRankOp.jl),
[`per_readout_basis.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/main/src/EncodingOperator/LowRankApproximation/per_readout_basis.jl),
[`per_readout_cache.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/main/src/EncodingOperator/LowRankApproximation/per_readout_cache.jl),
[`multi_gpu_normal_operator.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/main/src/EncodingOperator/LowRankApproximation/multi_gpu_normal_operator.jl),
and [`recon_HOOp.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/main/src/Recon/recon_HOOp.jl).

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
