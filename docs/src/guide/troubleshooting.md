# Troubleshooting

## Check the active environment

Start Julia with the intended project and record `VERSION` and `Pkg.status()`. The package loads plotting dependencies even for reconstruction use. For a headless session set `MPLBACKEND=Agg` before starting Julia.

```julia
using Pkg, PyCall
Pkg.status()
println((python=PyCall.python, version=PyCall.pyversion, library=PyCall.libpython))
```

The Julia Manifest does not by itself identify the external Python executable and shared library selected by PyCall. If a Python change is needed, use an isolated Julia depot, select the interpreter explicitly, rebuild PyCall there, and restart Julia. A package version combination from one machine is not a universal compatibility guarantee.

## Check GPU availability and placement

```julia
using CUDA
CUDA.functional()
CUDA.devices()
```

Device numbers are local to the process and can be remapped by `CUDA_VISIBLE_DEVICES`. Match operator, data and weight placement as described in the [reconstruction guide](/guide/reconstruction). The current interfaces do not all perform automatic host/device adaptation.

For a memory error, account for factors, coil maps, input data, NFFT plans, solver vectors and temporary buffers. Rank sharding and sample batching reduce particular parts of that footprint; they do not bound every allocation. See [multi-GPU execution](/guide/multi-gpu) and [LowRank limits](/reference/highorderlowrankop).

## Large-operator replacement

Close a distributed LowRank backend before replacing its parent, then remove references to the old operator:

```julia
close(op)
op = nothing
GC.gc()
```

`@rebuild_HOOp` provides this replacement sequence. A direct assignment can keep the previous value alive while the new constructor runs. Closing a normal view is not a substitute for closing the parent that owns workers; see [resource lifecycle](/reference/resources).

## CUDA memory reporting

Allocator-reserved memory can remain visible after live workspaces are released. Separate active allocations from pool reservation when investigating memory use. Calling `CUDA.reclaim()` is not a substitute for fixing object ownership, nor a general remedy for native Python/CUDA failures.

For a reproducible failure report include a small input, the code revision, package and Python versions, GPU identities, selected backend, full exception and whether it occurs during construction, evaluation or cleanup. Check the relevant API page for known input and backend restrictions.
