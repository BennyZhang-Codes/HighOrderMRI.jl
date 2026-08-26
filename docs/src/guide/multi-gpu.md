# Multi-GPU execution

Multi-GPU execution in HighOrderMRI.jl is stage specific. The explicit CUDA operator and distributed low-rank setup partition masked voxels, whereas the optional low-rank normal operator partitions receive channels. These decompositions alter data placement, communication, and summation order but do not define a different physical signal model.

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

## Explicit encoding: voxel decomposition

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

`HighOrderKernelOp` partitions the masked voxel set into contiguous ranges. Each device stores the corresponding field-map values, spatial basis rows, coil-map rows, and workspaces. During the forward operation, each device evaluates the contribution from its voxel subset and the partial signals are summed on the host. During the adjoint, the disjoint voxel results are gathered into their original masked positions.

For the explicit operator, this is a decomposition of the same discrete encoding model. Only data placement and finite-precision reduction order change.

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

Voxel-distributed setup currently requires:

- `arrayType=CuArray`;
- `rsvd_backend=:kernel`;
- `rsvd_finalize=:gram`;
- at least two distinct GPU IDs;
- $L+p\leq 32$, where $L$ is `L_rank` and $p$ is `rsvd_oversample`.

With `rsvd_distribution=:auto`, voxel distribution is selected when these conditions are satisfied.

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

## Communication model

The communication pattern depends on the stage and should be reported separately when evaluating multi-GPU performance:

| Stage | Partitioned dimension | Cross-device / host communication |
| --- | --- | --- |
| Explicit `HighOrderKernelOp` | Masked voxels | Host reduction of partial forward signals; adjoint voxel shards are gathered |
| Distributed low-rank setup | Masked voxels | Reduction of sample-domain sketches and small Gram matrices; completed shared basis gathered once |
| Low-rank normal operator | Receive channels | Input image distributed to workers; local normal contributions summed on the primary path |

The current explicit implementation is voxel sharded; forward samples are not the partition dimension.

## Resource lifecycle

`recon_HOOp` releases the distributed normal backend by default. After direct use of the normal operator, or after reconstruction with `release_backend=false`, release the associated resources explicitly:

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
