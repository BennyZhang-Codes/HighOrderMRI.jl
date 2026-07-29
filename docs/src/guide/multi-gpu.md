# Multi-GPU execution

HighOrderMRI uses different decompositions for different phases of the
workload:

- the explicit CUDA operator shards masked voxels;
- distributed low-rank setup shards masked voxels;
- the optional low-rank normal operator shards receive channels.

These choices are complementary. Multi-GPU partitioning changes execution and
communication, not the intended numerical approximation.

## Runtime requirements

Use zero-based device IDs and start Julia with enough default threads:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 julia --threads=auto --project=.
```

At least one default Julia thread per GPU worker is needed for effective
overlap; one additional coordinator thread is recommended. With fewer
threads, results remain mathematically valid, but kernel submission,
transfers, and synchronization may occur in waves.

Check the visible devices before allocating a large operator:

```julia
using CUDA

CUDA.functional() || error("CUDA is not functional")
collect(CUDA.devices())
```

## Explicit operator

```julia
op = HighOrderOp_Kernel(
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

The operator partitions `nVox` into contiguous ranges. Each device owns its
field map, basis rows, coil-map rows, and workspaces. Forward partial signals
are reduced on the host; adjoint voxel results are gathered into their
original mask positions.

## Voxel-distributed rSVD setup

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

For voxel shards ``\mathcal V_g``:

```math
H_d\Omega_d
=
\sum_g H_{d,g}\Omega_{d,g},
\qquad
B_d^HB_d
=
\sum_g B_{d,g}^HB_{d,g}.
```

Only small sketches and Gram contributions are reduced on the host. The
large spatial factors and the shared basis remain voxel-sharded during setup,
then the completed shared basis is gathered once to the primary operator
device.

Voxel distribution currently requires:

- `arrayType=CuArray`;
- `rsvd_backend=:kernel`;
- `rsvd_finalize=:gram`;
- at least two unique GPU IDs;
- `L_rank + rsvd_oversample <= 32`.

With `rsvd_distribution=:auto`, this mode is selected when these conditions
are satisfied.

## Channel-distributed normal operator

Set:

```julia
normal_distribution=:channel
```

when constructing a CUDA `HighOrderLowRankOp` with at least two GPUs. The
normal operator partitions coils and computes local
``A_g^H W^2 A_g x`` contributions. Those image-space contributions are
reduced to the primary GPU.

The backend is constructed lazily when `normalOperator(W ∘ op)` is first
requested. Setup distribution and normal distribution are independent; the
same operator may use voxel shards during rSVD setup and coil shards during
iterative reconstruction.

## Resource lifecycle

`recon_HOOp` releases the distributed normal backend by default. After direct
normal-operator use, or after reconstruction with `release_backend=false`,
release resources explicitly:

```julia
close(op)
# or
release_highorder_normal_backend!(op)
```

Before replacing a large operator:

```julia
close(op)
op = nothing
GC.gc()
```

Julia retains the old right-hand-side value until a replacement assignment
finishes. Constructing a new operator directly into the same variable can
therefore make old and new NFFT plans and bases coexist temporarily.

CUDA.jl uses reusable memory pools. Releasing workspaces does not guarantee an
immediate decrease in the reserved-memory value shown by `nvidia-smi`.

## What to report

Record device IDs and UUIDs, GPU model, driver, CUDA and Julia versions,
thread count, partition mode, primary device, setup time, reduction time,
steady-state operator time, peak memory, and failures such as Xid/device-loss
events. Exclude runs affected by competing GPU processes from a final
benchmark.
