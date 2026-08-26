# Troubleshooting

This page collects environment and runtime issues that are useful for operating HighOrderMRI but are not part of the scientific encoding model.

## PyCall finalizer crash after CUDA reconstruction

HighOrderMRI currently loads PyPlot through PyCall. A compatibility failure has been observed with Julia 1.12, CUDA.jl 6.2, PyCall 1.96.4, and CPython 3.13: reconstruction finishes successfully and the Julia prompt returns, but the process may terminate with signal 11 several minutes later or during CUDA memory-pool cleanup.

A representative end of the native stack is:

```text
CUDACore.pool_cleanup
  -> CUDA.reclaim
  -> GC.gc
  -> PyCall.pydecref
  -> Python PyObject_Free
  -> signal 11 (segmentation fault)
```

This stack does not by itself indicate an error in `HighOrderOp`, `HighOrderKernelOp`, coil compression, or the reconstructed result. CUDA memory management initiates Julia garbage collection; that collection can then run a PyCall finalizer, with the native failure occurring while CPython releases an object.

### Validated workaround

The following combination completed repeated reconstructions, explicit garbage collection, and `CUDA.reclaim()` without the delayed process exit:

| Component | Validated version |
|:--|:--|
| Julia | 1.12.6 |
| CUDA.jl / CUDACore | 6.2.1 |
| PyCall | 1.96.4 |
| PyPlot | 2.11.6 |
| Python | 3.10.13 |
| Matplotlib | 3.10.6 |

The PyCall and PyPlot versions were unchanged from the failing configuration; the effective change was from CPython 3.13 to CPython 3.10. This is empirical compatibility evidence rather than proof that every Python 3.13 configuration will fail.

Create or select a Python 3.10 environment containing NumPy and Matplotlib, then rebuild PyCall against that interpreter:

```bash
conda create -n highordermri-py310 python=3.10 numpy matplotlib
conda activate highordermri-py310
PYTHON="$(command -v python)" julia --project=. -e '
using Pkg
Pkg.build("PyCall")'
```

Restart Julia after rebuilding PyCall and verify the interpreter and shared library before reconstruction:

```julia
using PyCall

PyCall.python
PyCall.pyversion
PyCall.libpython
```

The reported executable and `libpython` must both belong to the intended Python 3.10 installation.

A basic cleanup check is:

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

The decisive validation is still the real workload: repeat the reconstruction several times and leave the interactive Julia process alive after cleanup.

### Reproducibility limits

`Project.toml` compatibility entries can constrain Julia, CUDA.jl, PyCall, and PyPlot versions, but neither `Project.toml` nor `Manifest.toml` records the external Python executable or `libpython` selected when PyCall is built. The selection is stored in the Julia depot and can be shared by environments using the same installed PyCall source.

For a reproducibility record, store `PyCall.python`, `PyCall.pyversion`, and `PyCall.libpython` together with the Julia and CUDA versions. If a different Python runtime is required, set `PYTHON` explicitly and rebuild PyCall again; do not treat `CUDA.reclaim()` or REPL keep-alive settings as a fix for a native finalizer crash.

## Large-operator replacement

Before replacing a large `HighOrderLowRankOp`, release its distributed backend and remove references to the old operator before constructing the new one:

```julia
close(op)
op = nothing
GC.gc()
```

The convenience macro `@rebuild_HOOp` performs this sequence before evaluating the replacement constructor. This matters because an assignment such as `op = HighOrderLowRankOp(...)` can keep the old value alive until the new constructor finishes, temporarily making old and new NFFT plans and spatial bases coexist.

## CUDA memory reporting

CUDA.jl uses reusable memory pools. Releasing explicit workspaces or calling `close(op)` does not guarantee an immediate decrease in the reserved-memory value reported by `nvidia-smi`. Distinguish allocated memory used by live objects from memory reserved by the CUDA allocator when diagnosing an apparent leak.
