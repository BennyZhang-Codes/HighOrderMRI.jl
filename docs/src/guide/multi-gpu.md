# Multi-GPU execution

Multi-GPU execution in HighOrderMRI.jl is stage specific. Legacy explicit kernels and distributed low-rank setup partition masked voxels; resident explicit backends partition samples. The low-rank normal partitions receive channels or, for independent factors, readouts. These decompositions alter data placement, communication, and summation order but preserve the selected signal model.

Symbols and the weighting convention follow [Symbols and notation](/theory/symbols).

## Runtime requirements

CUDA device IDs are zero based. A multi-GPU Julia process can be started with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 julia --threads=auto --project=.
```

Effective concurrent submission requires at least one default Julia thread per GPU worker; an additional coordinator thread is useful. With fewer threads, the numerical result is unchanged, but kernel submission, transfers, and synchronization can occur in sequential waves.

Verify the visible devices before constructing a large operator:

```julia
using CUDA

CUDA.functional() || error("CUDA is not functional")
collect(CUDA.devices())
```

## Legacy explicit encoding: voxel decomposition

```julia
op = HighOrderKernelOp(
    grid,
    kspha,
    times;
    fieldmap,
    csm,
    mask,
    arrayType=CuArray,
    gpus=[0, 1, 2, 3],
)
```

With `backend=:legacy` (the default), `HighOrderKernelOp` partitions the masked voxel set into contiguous ranges. Each device stores the corresponding field-map values, spatial basis rows, coil-map rows, and workspaces. During the forward operation, each device evaluates the contribution from its voxel subset and the partial signals are summed on the host. During the adjoint, the disjoint voxel results are gathered into their original masked positions.

For the explicit operator, this is a decomposition of the same discrete encoding model. Only data placement and finite-precision reduction order change.

## Resident explicit encoding: sample decomposition

```julia
op = HighOrderKernelOp(grid, kspha, times;
    fieldmap, csm, mask,
    gpus=[0, 1],
    backend=:blocked,
    block_size=1024,
    phase_backend=:gemm,
    gemm_backend=:cublas,
    phase_cache_bytes=8 * 2^30,
)
```

The same public constructor selects the internal multi-GPU implementation from `gpus`. Each persistent worker retains the complete masked spatial model and all coils, but only one contiguous sample range. The primary device `first(gpus)` owns resident CG vectors.

Let $A_g$ be the rows of the complete encoding operator assigned to GPU $g$, and let $D_g$ be the corresponding already squared diagonal weights, preserving sample-by-coil order. The normal is

$$
A^H D A x = \sum_g A_g^H D_g A_g x.
$$

Each worker reuses its phase blocks for both products. The image is broadcast through reusable host buffers, local image contributions are summed, and the result is copied to the primary GPU. The implementation does not require GPU peer access. Forward outputs are assembled in sample order; adjoint and normal image contributions are summed. Normalization uses the same global masked voxel count on every worker.

The `phase_cache_bytes` budget applies per GPU. A complete cached matrix, partial cache, model, and scratch must fit the corresponding device individually; total memory summed over all GPUs is insufficient evidence. With partial caching, worker-local sample prefixes are retained and the remaining rows regenerated. Construction time includes these caches.

IDs are relative to CUDA's visible list. Without `CUDA_VISIBLE_DEVICES`, `gpus=[1,2,3,4,5]` selects those CUDA device numbers. With `CUDA_VISIBLE_DEVICES=4,5`, use `gpus=[0,1]` to select physical GPUs 4 and 5. See the [public API](/reference/highorderkernelop) for backend settings and lifecycle.

## Low-rank setup: voxel-distributed rSVD

```julia
op = HighOrderLowRankOp(
    grid,
    kspha_dynamic,
    times_dynamic;
    fieldmap,
    csm,
    mask,
    arrayType=CuArray,
    gpus=[0, 1, 2, 3],
    L_rank=10,
    rsvd_oversample=5,
    rsvd_backend=:kernel,
    rsvd_finalize=:gram,
    rsvd_distribution=:voxel,
)
```

Let $\mathcal V_g$ denote the masked-voxel subset assigned to GPU $g$. Partitioning $H_d$ by voxel columns and $\Omega_d$ by the corresponding rows gives

$$
H_d\Omega_d
=
\sum_g H_{d,g}\Omega_{d,g}.
$$

Because

$$
B_d=H_d^H Q_d
$$

is partitioned by voxel rows, its small Gram matrix is obtained from

$$
B_d^H B_d
=
\sum_g B_{d,g}^H B_{d,g}.
$$

Thus, only the sample-domain randomized sketch and the small Gram contributions require reduction across devices. The large spatial factors and the shared spatial basis remain voxel distributed during setup, and the completed shared basis is gathered once to the primary operator device.

Voxel distribution requires:

- `arrayType=CuArray`;
- `rsvd_backend=:kernel` or explicit `:chunked`;
- `rsvd_finalize=:gram`;
- at least two unique GPU IDs.

With `rsvd_distribution=:auto`, this mode is selected for the fused kernel
when the remaining conditions are satisfied. Chunked voxel distribution
requires explicit `rsvd_distribution=:voxel`.

### Shared setup kernels

For shared Float32 setup with at least 256 samples per readout, voxel-tiled
forward sketches use at most 64 MiB of temporary partial sketches per worker;
small voxel shards retain the original mapping. Paired adjoint lanes reuse
temporal loads across two voxels. These mappings keep all samples, voxels,
sketch columns, and seeds, while changing the summation order. Float64,
single-GPU, and per-readout setup retain their existing mappings. The
[shared residual Gram](/theory/low-rank#incremental-shared-spatial-basis)
continues to use Float64 accumulation.

For voxel-distributed per-readout setup, `per_readout_gather=:pipeline`
overlaps downloads and host writes through two pinned buffers per GPU.
Collectors and compute workers need Julia threads to submit concurrently;
an additional coordinator thread is useful. The full materialized factor
bank is still required unless regeneration is selected. See
[setup aggregation](/theory/low-rank#pipelined-setup-aggregation).

### Setup backend selection

The fused backend supports wide sketches by batching 16 columns per launch.
It preserves the full Gram matrix, including cross-block terms. Explicit
`rsvd_distribution=:voxel, rsvd_backend=:chunked` is supported
with `rsvd_finalize=:gram`. Its phase and encoding workspaces are bounded by
`rsvd_chunk` and reused across dynamics.

Automatic voxel distribution still selects only the fused kernel path on
CUDA with multiple GPUs and Gram finalization. The chunked path requires
explicit selection: it was slower on the real EPI setup despite promising
small-matrix timings. See [backend tradeoffs](/guide/performance#backend-and-precision-tradeoffs).

With `shared_basis_method=:joint`, setup runs on the primary GPU
(`first(gpus)`), or on the CPU for `arrayType=Array`. Set
`rsvd_distribution=:auto` or `:single`; `:voxel` is rejected.
Additional GPUs can still participate in channel-distributed normal evaluation
without changing its algebra or lifecycle.

## Iterative reconstruction: channel-distributed normal operator

A channel-distributed normal operator is selected with

```julia
normal_distribution=:channel
```

when constructing a CUDA `HighOrderLowRankOp` with at least two GPUs. Let $W$ denote the diagonal reconstruction weighting operator and let $\mathcal C_g$ contain the receive coils assigned to GPU $g$. The local normal contribution is

$$
N_gx
=
\sum_{c\in\mathcal C_g}
A_c^H W^H W A_c x,
$$

and the complete weighted normal operation is

$$
A^H W^H W A x
=
\sum_g N_gx.
$$

For the square-root density weights used by `recon_HOOp`, $W$ is diagonal and the current implementation stores `abs2.(weights)` on each channel shard. When the supplied weights are real and non-negative, $W^H W=W^2$.

Channel partitioning is therefore a decomposition of the normal operator associated with the already constructed low-rank encoding operator. It does not add another low-rank approximation.

The distributed normal backend is constructed lazily when `normalOperator(W ∘ op)` is first requested. Setup distribution and normal-operator distribution are independent: the same `HighOrderLowRankOp` can use voxel decomposition during rSVD construction and channel decomposition during iterative reconstruction.

Once the shared normal backend exists, adjoints also reuse its channel
workers, including the solver RHS. Workers evaluate the supplied adjoint
input directly and sum image contributions on the host; the normal backend's
squared weights are not applied again. Before backend initialization,
standalone adjoints use the primary GPU.

On the primary worker, `q`, the resident spatial `basis`, the relevant `csm`
columns, and `mask_idx` are views of the source operator's storage. Other
devices receive their own copies. This avoids duplicate primary-device
storage; it does not change the factors or their precision. Streamed
per-readout factors instead use bounded queues.

## Iterative reconstruction: readout-distributed normal operator

For independent factors, select `basis_layout=:per_readout` and
`normal_distribution=:readout`. Each worker evaluates all coils for a
contiguous readout range $mathcal D_g$:

$$
N_gx = sum_{dinmathcal D_g}sum_c A_{d,c}^H W_{d,c}^H W_{d,c} A_{d,c}x.
$$

The image contributions are summed across workers. All cross-rank terms
within each readout are retained, so channel and readout distribution
evaluate the same weighted normal up to summation order.

With `per_readout_storage=:cpu`, each factor block is transferred once across
the worker group per normal application. Channel distribution traverses the
bank on every worker. Readout distribution stores every CSM on each GPU and
uses at most one worker per readout; uneven readout counts are supported.
Regeneration replaces uncached factor transfers with matrix-free generation.
When the distributed backend exists, streamed adjoints also reuse its queues
and distribute readouts while evaluating all coils together.

`:readout` requires the per-readout layout. The default remains `:single`;
shared factors can use `:single` or `:channel`. See the
[execution keywords](/reference/highorderlowrankop#execution-keywords).

## Communication model

The communication pattern depends on the stage and should be reported separately when evaluating multi-GPU performance:

| Stage | Partitioned dimension | Cross-device / host communication |
| --- | --- | --- |
| Legacy explicit `HighOrderKernelOp` | Masked voxels | Host reduction of partial forward signals; adjoint voxel shards are gathered |
| Resident explicit `HighOrderKernelOp` | Samples | Input image broadcast; forward samples assembled; adjoint/normal image contributions summed |
| Distributed low-rank setup | Masked voxels | Reduction of sample-domain sketches and small Gram matrices; completed shared basis gathered once |
| Low-rank normal, `:channel` | Receive channels | Input image broadcast; local image contributions summed; streamed factor bank traversed by every worker |
| Low-rank normal, `:readout` | Independent readouts | Input image broadcast; factors transferred or generated for assigned readouts; local image contributions summed |

Choose the explicit decomposition through `backend` on `HighOrderKernelOp`: legacy voxel sharding or resident sample sharding. Selecting more GPUs does not change the mathematical encoding model.

## Resource lifecycle

For resident `HighOrderKernelOp`, call `close(op)` explicitly after reconstruction and after all derived normals are no longer used. `recon_HOOp` does not close these resident explicit workspaces.

For `HighOrderLowRankOp`, `recon_HOOp` releases the distributed normal backend by default. A per-readout source queue remains available for forward/adjoint reuse; close the operator after its final use. After direct use of the normal operator, or after reconstruction with `release_backend=false`, release the associated resources explicitly:

```julia
close(op)
# or
release_highorder_normal_backend!(op)
```

Before constructing a replacement for a large operator, the previous object can be released explicitly:

```julia
close(op)
op = nothing
GC.gc()
```

During an assignment such as `op = HighOrderLowRankOp(...)`, Julia retains the previous right-hand-side value until evaluation of the replacement expression is complete. Constructing the replacement directly into the same binding can therefore make the old and new NFFT plans and spatial bases coexist temporarily.

CUDA.jl uses reusable memory pools. Releasing explicit workspaces does not necessarily produce an immediate decrease in the reserved device memory reported by `nvidia-smi`.

Environment-specific failures, including the observed PyCall/CPython finalizer crash after CUDA reconstruction, are documented in [Troubleshooting](/guide/troubleshooting).

## Reporting and benchmarking

For a multi-GPU experiment, record the device IDs and UUIDs, GPU model, driver, CUDA and Julia versions, Julia thread count, partition mode, primary device, setup and reduction times, steady-state operator time, and peak device memory. Runs affected by competing GPU workloads, device-loss events, or other system-level failures should be identified separately from valid timing measurements.

Runtime measurements should be accompanied by the operator-level numerical checks in [Scientific validation strategy](/guide/validation), with reconstruction conventions fixed according to the [reconstruction protocol](/guide/reconstruction-protocol).
