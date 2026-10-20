# Multi-GPU execution

HighOrderMRI uses different decompositions for different phases of the
workload:

- the explicit CUDA operator shards masked voxels;
- `HighOrderExplicitMultiGPUOp` shards samples and retains solver vectors on
  the primary GPU;
- distributed low-rank setup shards masked voxels;
- the optional low-rank normal operator shards receive channels, or independent
  readouts with `basis_layout=:per_readout`.

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

The operator partitions `nVox` into contiguous ranges. Each device owns its
field map, basis rows, coil-map rows, and workspaces. Forward partial signals
are reduced on the host; adjoint voxel results are gathered into their
original mask positions.

### GPU-resident sample-distributed explicit operator

```julia
op = HighOrderExplicitMultiGPUOp(grid, kspha, times;
    fieldmap, csm, mask,
    gpus=[0, 1, 2, 3, 4],
    backend=:blocked,
    phase_backend=:gemm,
    gemm_backend=:cublas,
    block_size=1024,
)
```

Device IDs are relative to `CUDA_VISIBLE_DEVICES`. The first ID owns the CG
vectors. With no visibility filter, `gpus=[1, 2, 4, 5, 6]` directly selects
those CUDA device IDs. With `CUDA_VISIBLE_DEVICES=2,1,4,5,6`, use
`gpus=[0, 1, 2, 3, 4]` to select the same devices with physical GPU 2 first.
For a single GPU, `HighOrderExplicitGPUOp(...; gpus=[2])` selects device 2
when no filter is set. Each persistent worker receives a contiguous sample range, the same
masked voxels, and all coils. The weighted normal is the sum of the workers'
local normal operators, with the same global voxel normalization as the
single-GPU operator. Each complex phase block is generated once per local
normal application and reused for the forward and adjoint products. Real and
complex matrix products explicitly call cuBLAS.

Reusable host buffers broadcast the image and reduce partial images. Forward
and adjoint also assemble or split the sample-by-coil data in the original
order. Communication does not transfer phase matrices and does not require
peer access. Its measured cost belongs in reconstruction timing.

`backend=:direct` uses the existing direct kernels on each sample shard.
`backend=:cached` retains every local phase row, allowing a complete phase
matrix to be distributed over several GPUs. `backend=:cached_full` retains the
same phase matrix and uses a full sample-by-coil workspace to apply it with one
GEMM per forward or adjoint, instead of several sample-block products. Matrix
initialization remains blocked in both modes. Check each device's phase matrix,
workspace, model, solver, and library memory before constructing this mode;
aggregate memory alone does not establish feasibility.

Pass GPU data and weights to `recon_HOOp`. After all reconstruction and normal
applications finish, call `close(op)` to release the workers and device
workspaces. Operators and their derived normal operators share mutable
scratch space and must not be applied concurrently.

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

For shared-basis setup with `Float32` and at least 256 samples per readout,
the forward kernel partitions the masked voxels into smaller tiles and sums
their sketch contributions on the GPU. Each worker retains at most 64 MiB of
temporary partial sketches; small voxel shards use the original forward kernel.
The adjoint kernel uses two cooperating lanes per voxel and reuses each temporal
load for a pair of voxels, keeping spatial coefficients in registers. These mappings retain
all samples, voxels, sketch columns, and random seeds, while changing floating
point summation order. `Float64`, single-GPU setup, and per-readout setup retain
their previous mappings. Shared-basis Gram precision is unchanged.

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

For the shared spatial basis, subsequent adjoint applications also reuse these
channel workers, including the initial right-hand side in CG. Each worker
computes its coils' adjoint contribution, and the image contributions are summed
on the host. This evaluates the adjoint input directly; the normal backend's
squared weights are not applied again. Standalone adjoints before backend setup
retain single-GPU execution.

## Readout-distributed normal operator

For independent per-readout factors, select:

```julia
basis_layout=:per_readout
normal_distribution=:readout
```

Each worker evaluates all coils for its assigned contiguous readout range.
The contributions are summed in image space, including all cross-rank terms
within each readout. This is the same weighted normal operator as channel
distribution, with a different summation order.

With `per_readout_storage=:cpu`, each spatial factor block is streamed once
across the worker group per normal application. Channel distribution instead
streams the whole bank to every worker. Readout distribution keeps all coil
sensitivity maps on each GPU, which increases device storage. It uses at most
one worker per readout and supports uneven readout counts. The shared layout
continues to use `:channel`; `:readout` requires `basis_layout=:per_readout`.
The default `normal_distribution=:single` and existing `:channel` behavior
are unchanged.

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

## PyCall finalizer crash after CUDA reconstruction

HighOrderMRI currently loads PyPlot through PyCall. A compatibility failure
has been observed with Julia 1.12, CUDA.jl 6.2, PyCall 1.96.4, and CPython
3.13: reconstruction finishes successfully and the Julia prompt returns, but
the process may terminate with signal 11 several minutes later or during CUDA
memory-pool cleanup. A representative end of the native stack is:

```text
CUDACore.pool_cleanup
  -> CUDA.reclaim
  -> GC.gc
  -> PyCall.pydecref
  -> Python PyObject_Free
  -> signal 11 (segmentation fault)
```

This stack does not by itself indicate an error in `HighOrderOp`,
`HighOrderKernelOp`, coil compression, or the reconstructed result. CUDA
memory management initiates a Julia garbage collection; that collection then
runs a PyCall finalizer, and the native failure occurs while CPython releases
an object.

### Validated workaround

The following combination completed repeated reconstructions, explicit
garbage collection, and `CUDA.reclaim()` without the delayed process exit:

| Component | Validated version |
|:--|:--|
| Julia | 1.12.6 |
| CUDA.jl / CUDACore | 6.2.1 |
| PyCall | 1.96.4 |
| PyPlot | 2.11.6 |
| Python | 3.10.13 |
| Matplotlib | 3.10.6 |

The PyCall and PyPlot versions were unchanged from the failing configuration;
the effective change was from CPython 3.13 to CPython 3.10. This is empirical
compatibility evidence rather than proof that every Python 3.13 configuration
will fail. Until the finalizer interaction is resolved upstream, use a Python
3.10 environment for long-lived CUDA reconstruction sessions that also load
PyPlot.

Create or select a Python 3.10 environment containing NumPy and Matplotlib,
then rebuild PyCall against that interpreter:

```bash
conda create -n highordermri-py310 python=3.10 numpy matplotlib
conda activate highordermri-py310
PYTHON="$(command -v python)" julia --project=. -e '
using Pkg
Pkg.build("PyCall")'
```

Restart Julia after rebuilding PyCall and verify the interpreter and shared
library before reconstruction:

```julia
using PyCall

PyCall.python
PyCall.pyversion
PyCall.libpython
```

The reported executable and `libpython` must both belong to the intended
Python 3.10 installation. A basic cleanup check is:

```julia
using CUDA, PyCall, PyPlot

fig = PyPlot.figure()
PyPlot.close(fig)
fig = nothing

buffer = CUDA.zeros(Float32, 1024)
buffer = nothing
GC.gc(true)
CUDA.reclaim()
GC.gc(true)
```

The decisive validation is still the real workload: repeat the reconstruction
several times and leave the interactive Julia process alive after cleanup.

### Reproducibility limits

`Project.toml` compatibility entries can constrain Julia, CUDA.jl, PyCall,
and PyPlot versions, but neither `Project.toml` nor `Manifest.toml` records the
external Python executable or `libpython` selected when PyCall is built. The
selection is stored in the Julia depot and is shared by environments using
the same installed PyCall source. Rebuilding PyCall in a shared depot can
therefore change the Python runtime used by more than one Julia project.

Record `PyCall.python`, `PyCall.pyversion`, and `PyCall.libpython` together
with the Julia and CUDA versions in a reproducibility report. If a different
Python runtime is required, set `PYTHON` explicitly and rebuild PyCall again;
do not treat `CUDA.reclaim()` or REPL keep-alive settings as a fix for a native
finalizer crash.

## What to report

Record device IDs and UUIDs, GPU model, driver, CUDA and Julia versions,
thread count, partition mode, primary device, setup time, reduction time,
steady-state operator time, peak memory, and failures such as Xid/device-loss
events. Exclude runs affected by competing GPU processes from a final
benchmark.
