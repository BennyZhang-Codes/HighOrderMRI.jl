# Multi-GPU execution

HighOrderMRI.jl can distribute operator construction and reconstruction over
several NVIDIA GPUs. These stages use different decompositions. Choose the
reconstruction strategy according to the arrays that must fit on each device;
the sum of the GPUs' memory capacities does not establish that an operator fits.

For `HighOrderLowRankOp`, `rsvd_distribution` selects how the factors are built,
and `normal_distribution` selects how the completed operator is evaluated during
iterative reconstruction. The default reconstruction strategy is `:single`.

## Choose a decomposition

| Operator and stage | Selection | Distributed work and storage |
|:--|:--|:--|
| Explicit legacy encoding | `HighOrderKernelOp(...; backend=:legacy)` | Masked voxels; each worker retains its spatial rows. |
| Explicit resident encoding | `backend=:direct`, `:blocked`, `:cached`, or `:cached_full` | Samples; each worker retains all masked voxels and coils for its sample range. |
| Low-rank rSVD setup | `rsvd_distribution=:voxel` | Masked voxels during factor construction and shared-basis updates. |
| Low-rank reconstruction | `normal_distribution=:single` | CPU execution with `arrayType=Array`, or primary-GPU execution with `CuArray`. |
| Low-rank reconstruction | `normal_distribution=:channel` | Receive coils; shared temporal and spatial factors are retained in full on each worker. |
| Low-rank reconstruction | `normal_distribution=:rank` | Shared temporal and spatial factor columns; coil-map columns are also distributed, and samples are processed in batches. |
| Low-rank reconstruction | `normal_distribution=:readout` | Independent readouts; requires `basis_layout=:per_readout`. |

Changing the GPU decomposition preserves the selected encoding representation,
including its rank and phase terms. Floating-point reduction order can change.
Low-rank approximation is controlled separately by the factor-construction
parameters described in [Low-rank shared subspace](/theory/low-rank).

## Runtime requirements

Start Julia with enough default threads for the selected workers. For example,
to expose two physical GPUs and provide two worker threads plus a coordinator:

```bash
CUDA_VISIBLE_DEVICES=0,1 julia --threads=3 --project=.
```

Device IDs in `gpus` are zero based and relative to `CUDA_VISIBLE_DEVICES`.
With `CUDA_VISIBLE_DEVICES=4,5`, use `gpus=[0, 1]` to select physical GPUs 4
and 5. The public low-rank constructor uses `first(gpus)` as the primary GPU,
which holds the solver vectors. It does not expose a separate `primary_gpu`
keyword.

Check CUDA before constructing an operator:

```julia
using CUDA

CUDA.functional() || error("CUDA is not functional")
collect(CUDA.devices())
```

Provide at least one default Julia thread per active GPU worker for effective
concurrent submission; an additional coordinator thread is useful. Fewer threads
can serialize transfers and kernel submission. GPU IDs must be unique.

## Legacy explicit encoding: voxel decomposition

`HighOrderKernelOp` uses voxel decomposition with the default `backend=:legacy`.
Each GPU retains its field-map values, basis rows, coil-map rows, and workspaces.
Partial forward signals are summed on the host; adjoint voxel results are
gathered into their original masked positions. This backend uses CPU solver
storage.

## Resident explicit encoding: sample decomposition

For resident solver storage and sample decomposition, select a resident backend:

```julia
op = HighOrderKernelOp(grid, kspha, times;
    fieldmap, csm, mask,
    gpus=[0, 1],
    backend=:blocked,
    phase_backend=:gemm,
    gemm_backend=:cublas,
    block_size=1024,
    phase_cache_bytes=0,
)
```

The first selected GPU owns the solver vectors. Each worker evaluates its
contiguous sample range using the complete masked spatial model and all coils.
The input image and partial output images pass through reusable host buffers;
GPU peer access is not required. Forward samples are assembled in their original
order, while adjoint and normal image contributions are summed.

`:blocked` reuses each complex phase block in both products of the weighted
normal. `:direct` evaluates the direct kernels. `:cached` and `:cached_full`
retain all local phase rows; `:cached_full` also uses a full sample-by-coil
workspace. With `:blocked`, `phase_cache_bytes` optionally retains a prefix of
complete phase blocks on each GPU. Its default is zero. The cache budget counts
complex phase entries, so include model arrays, scratch, solver storage, and
library workspaces when checking memory on every device.

Pass GPU data and weights to `recon_HOOp` for resident execution. Call `close(op)`
after the final use; the wrapper does not release resident explicit operators.
See [Choose an operator](/guide/operators) for backend settings.

## Low-rank setup: voxel-distributed rSVD

Use these constructor keywords to build local low-rank factors on voxel shards:

```julia
arrayType=CuArray,
gpus=[0, 1],
rsvd_backend=:kernel,
rsvd_finalize=:gram,
rsvd_distribution=:voxel,
```

For readout $i$, let $H_i$ be the residual phase matrix. Voxel distribution
partitions its columns and the corresponding rows of the random test matrix:

$$
H_i\Omega_i = \sum_g H_{i,g}\Omega_{i,g},
\qquad
B_i^H B_i = \sum_g B_{i,g}^H B_{i,g},
\qquad B_i = H_i^H Q_i.
$$

The sample-domain sketches and small Gram contributions are reduced on the host.
Large spatial factors and the incremental shared basis remain voxel distributed
during setup. The completed shared factors are gathered into host arrays. For
`:single` and `:channel` GPU execution they are then placed on the primary GPU;
`:rank` retains the complete factor masters in host RAM and uploads only each
worker's rank columns when its execution backend is first needed.

Explicit `rsvd_distribution=:voxel` requires CUDA, a nonempty list of unique GPU IDs,
`rsvd_finalize=:gram`, and `rsvd_backend=:kernel` or `:chunked`.
Explicit voxel setup accepts a single GPU; the number of selected devices
must not exceed `count(mask)`. With `rsvd_distribution=:auto`, voxel distribution is selected for CUDA with
multiple GPUs, the kernel backend, and Gram finalization. The chunked backend
requires explicit `:voxel` selection.

The fused kernels process sketch columns in batches of at most 16. This is a
kernel launch detail, not a rank limit: the sketch-width bound is
`L_rank + rsvd_oversample <= min(nSam, nVox)`. `rsvd_chunk` bounds the voxel
blocks used by the chunked implementation; it does not control reconstruction
sample batches.

### Optional no-add check during shared-basis construction

`shared_noadd_check=false` is the default. Set it to `true` only with
voxel-distributed shared rSVD (`basis_layout=:shared`,
`shared_basis_method=:rsvd`, CUDA and Gram finalization). It is independent of
whether reconstruction later uses `:single`, `:channel`, or `:rank`.

The update still computes the projection coefficients and the residual after
both orthogonalization passes. It then accumulates the residual's squared norm
in FP64. If that energy is safely below the allowance from `shared_basis_tol`,
the existing basis needs no new columns and the update skips residual-Gram
construction and its eigendecomposition. Cases near the threshold, non-finite
energy, or an insufficient numerical guard use the original Gram-based path.
The initial update also uses the original path because no shared basis exists.

This option does not loosen the tolerance, truncate the requested local rank, or
replace the projection. On the early-return path, the reported relative error
uses the full residual norm; it can differ from the original diagnostic, which
discards tiny residual eigenvalues. That diagnostic difference does not feed
back into factor construction. The option affects setup work, not CG batching
or the GPU communication strategy.

## Iterative reconstruction: channel-distributed normal operator

Select `normal_distribution=:channel` on a CUDA operator with at least two GPUs.
Each worker evaluates its assigned coils using the full shared `q` and `basis`.
For coil groups $\mathcal C_g$ and diagonal sample weights $W$,

$$
A^H W^H W A x
= \sum_g \sum_{c\in\mathcal C_g} A_c^H W^H W A_c x.
$$

The input image is broadcast through host memory, and the workers' image
contributions are summed and returned to the primary GPU. The backend is created
lazily on the first normal-operator request. Once it exists, shared-basis GPU
adjoints also reuse the channel workers, including the solver's initial
right-hand side. A standalone adjoint before backend creation uses the primary
GPU.

On the primary worker, full shared factors are views of the source operator's
storage; the other workers receive copies. Coil maps are partitioned. Adding
workers therefore reduces coil-map storage per worker but does not reduce the
full shared-factor storage required on each GPU. For independent per-readout
factors, `:cpu` or `:regenerate` storage instead uses bounded GPU queues.

## Rank-distributed shared execution

Select `normal_distribution=:rank` when shared-factor replication is the memory
constraint. The public constructor requires:

- `arrayType=CuArray` and at least two unique GPU IDs;
- `basis_layout=:shared` and `shared_basis_method=:rsvd`;
- voxel-distributed rSVD, either explicit `:voxel` or eligible `:auto`;
- `rsvd_finalize=:gram` and a compatible kernel or chunked setup backend.

The parent operator retains complete `q`, `basis`, and CSM arrays in host RAM.
Persistent workers and NFFT plans are created on the first forward, adjoint, or
normal request. Each worker receives all sample rows of its temporal rank
columns, the matching spatial columns, and a subset of coil-map columns. The
number of active workers is at most `min(length(gpus), shared_rank, nCha)`.
When at least two workers are active, the public constructor's primary worker
receives one rank column to leave room for solver storage; remaining columns
are divided among the other workers. A rank or coil count of one limits
execution to one worker even if more GPUs are listed.
Coil-map assignment also accounts for the estimated factor and primary-device
storage.

Coil-map ownership is storage placement: all rank workers evaluate every coil.
For a coil whose map is owned by another worker, the map is uploaded from the
host master into a reusable single-coil buffer. Each worker still needs full
image-grid/NFFT workspaces and its factor columns, so rank distribution does not
guarantee equal memory use or that a particular reconstruction fits.

### Cross-rank terms and communication

For one coil, write $A_c = \sum_g A_{c,g}$, where each worker evaluates its
rank columns. The weighted normal is

$$
A^H W^H W A x
= \sum_c \sum_g A_{c,g}^H W^H W
  \left(\sum_h A_{c,h}x\right).
$$

For each sample batch and coil, workers download their partial forward signals.
The host sums these signals and broadcasts the complete signal back to every
rank worker. Each worker applies `abs2.(weights)` and its local adjoint. After
all batches and coils, partial images are summed on the host and copied to the
primary GPU. This order retains every cross-rank term. Summing independent
per-rank normal products would omit those terms and compute a different
operator.

Compared with channel distribution, rank distribution adds sample-domain
reduction and broadcast for each batch and coil. It uses host buffers and does
not require GPU peer access. Lower factor storage per GPU is accompanied by
additional communication; runtime depends on the actual problem and hardware.

### Sample batching and memory

`normal_sample_batch_size::Int=12_000_000` is a positive upper bound on samples
per NFFT batch in rank mode. The effective batch is
`min(nSam * nDyn, normal_sample_batch_size)`. Rows follow the flattened
sample-then-dynamic order; a batch may span readouts, and the final batch may be
shorter if the active NFFT backend supports changing the plan's output length.
All samples and factor columns are evaluated.

Smaller batches reduce sample-vector and NFFT node/interpolation storage. NFFT
nodes are updated as batches change, and more batches require more transforms
and transfers. Each worker's temporal factor still contains all sample rows for
its rank columns; spatial factors, coil maps, full image grids, solver vectors,
host factor masters, and complete input/output data are not bounded by this
keyword. It does not bound rSVD setup or constructor phase-folding buffers and
is unused by `:single`, `:channel`, and `:readout`, although the constructor
always requires a positive value.

With NFFT.jl 0.14.3 GPU plans, the package's current node-update path changes
the sample count but leaves the plan's declared output shape unchanged. A shorter
last batch can therefore raise `DimensionMismatch`. With this backend, use one
full batch or a batch size that divides `nSam * nDyn` exactly. The
NonuniformFFTs backend supports variable-length sample batches. If that optional package is installed in your Julia
environment, select it before creating the operator:

```julia
using AbstractNFFTs, NonuniformFFTs
AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
```

This selection applies to subsequently created plans. It does not update an
existing operator's plans.

### Minimal rank example

This small synthetic example demonstrates construction, sample batching, and
reconstruction using only the package's default NFFT backend. Run it from the
repository project with two visible GPUs and at least three Julia threads.
It is a usage example, not an accuracy or performance benchmark.

```julia
using HighOrderMRI, CUDA, RegularizedLeastSquares

T = Float32
grid = Grid(8, 8, 1, T(1e-3), T(1e-3), T(5e-3))
nSam, nDyn, nCha = 32, 3, 2
times = repeat(reshape(collect(range(zero(T), T(3e-3); length=nSam)), :, 1), 1, nDyn)
kspha = zeros(T, 9, nSam, nDyn)
for i in 1:nDyn
    kspha[2, :, i] .= range(T(-200), T(200); length=nSam)
    kspha[3, :, i] .= T(40 * (i - 2))
end
fieldmap = reshape(collect(range(T(-30), T(30); length=64)), 8, 8)
csm = ones(Complex{T}, 8, 8, nCha)
mask = trues(8, 8)

CUDA.device!(0)
op = HighOrderLowRankOp(grid, kspha, times;
    fieldmap, csm, mask,
    arrayType=CuArray,
    gpus=[0, 1],
    basis_layout=:shared,
    shared_basis_method=:rsvd,
    L_rank=4,
    rsvd_oversample=2,
    rsvd_backend=:kernel,
    rsvd_finalize=:gram,
    rsvd_distribution=:voxel,
    normal_distribution=:rank,
    normal_sample_batch_size=32,
    shared_noadd_check=false,
)
image = try
    x = CUDA.ones(Complex{T}, prod(grid.matrixSize))
    data = CuArray(reshape(op * x, nSam * nDyn, nCha))
    weights = CUDA.ones(Complex{T}, nSam * nDyn)
    rec_params = Dict{Symbol,Any}(
        :reconSize => grid.matrixSize,
        :solver => CGNR,
        :reg => L2Regularization(1f-6),
        :iterations => 3,
    )
    recon_HOOp(op, data, weights, rec_params)  # CPU Array
finally
    close(op)
end
```

The batch size 32 splits 96 sample/dynamic points into three equal batches and
avoids the default backend's output-shape limitation. Choose a batch size for
your problem's memory and transfer costs. Replace the synthetic inputs and
solver settings for real data. To enable the optional setup check, change only
`shared_noadd_check=true`; its default remains false.

## Iterative reconstruction: readout-distributed normal operator

For independent factors, select `basis_layout=:per_readout` and
`normal_distribution=:readout`. Each worker evaluates all coils for its
contiguous readout range. Image contributions are summed, retaining all
cross-rank terms within each readout. This evaluates the same weighted normal
as channel distribution for the same independent factors, up to summation order.

With `per_readout_storage=:cpu`, each assigned spatial block is streamed once
across the worker group per normal application. Channel distribution streams
the bank on every worker. Readout distribution stores all coil maps on each GPU
and uses at most one worker per readout. `:regenerate` replaces uncached factor
transfers with generation on demand. It remains a per-readout representation;
`:readout` is not available with the shared layout.

## Resource lifecycle

`recon_HOOp` accepts raw sample weights, typically square-root density weights,
and forms the corresponding weighted encoding. The normal term is
$A^H W^H W A$. When calling `normalOperator(op, D)` directly, `D` must already
contain the real, nonnegative diagonal of $W^H W$. Low-rank distributed modes
require identical sample weights for every coil.

`recon_HOOp` releases the low-rank distributed backend in a `finally` block by
default, including on a reconstruction exception. With
`recon_HOOp(...; release_backend=false)`, the backend remains available for reuse
with the same operator and weights. After direct forward/adjoint use in rank
mode, or after direct normal use, release resources explicitly:

```julia
close(op)
# To release only the distributed execution backend:
release_highorder_normal_backend!(op)
```

The release stops workers and releases their device arrays and plans. Host
factor masters remain owned by a rank operator; dropping the operator and other
references makes those arrays eligible for garbage collection. A later use of
the parent operator recreates its backend lazily. For distributed modes, release the backend before
requesting a normal with different weights, and create a fresh normal view after
release. Closing a derived low-rank normal view alone does not release the
parent's backend. The `:single` path has no owned distributed backend.

For independent factors, `close(op)` also closes the source streaming or
regeneration queue. Resident explicit operators require their own explicit
`close(op)` after their final use. Operators and their derived normals share
mutable workspaces and must not be applied concurrently.

Before replacing a large operator, release it and remove references before
starting the next construction:

```julia
close(op)
op = nothing
GC.gc()
```

`@rebuild_HOOp` performs this replacement sequence. CUDA.jl maintains reusable
memory pools, so releasing live arrays does not necessarily reduce the reserved
memory reported by `nvidia-smi` immediately.

## Implementation sources

- [Constructor defaults and execution restrictions](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/d141e7dc48e8d18380050aaa970ee4ae364d916d/src/EncodingOperator/HighOrderLowRankOp.jl).
- [Rank partitioning, batching, host communication, and cleanup](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/d141e7dc48e8d18380050aaa970ee4ae364d916d/src/EncodingOperator/LowRankApproximation/rank_sharded_shared.jl).
- [Channel/readout evaluation and normal ownership](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/d141e7dc48e8d18380050aaa970ee4ae364d916d/src/EncodingOperator/LowRankApproximation/multi_gpu_normal_operator.jl).
- [FP64 no-add check and guarded fallback](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/d141e7dc48e8d18380050aaa970ee4ae364d916d/src/EncodingOperator/LowRankApproximation/distributed_shared_spatial_basis.jl).

See [Low-rank shared subspace](/theory/low-rank) and
[References and source notes](/references) for the approximation algorithms
and their scientific background.
