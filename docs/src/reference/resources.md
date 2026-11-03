# Resource lifecycle

Resident explicit operators own CUDA workspaces and optional phase caches; CUDA low-rank operators can own a persistent channel-, readout-, or rank-distributed execution backend. Release these resources explicitly when the operator is no longer required outside `recon_HOOp`.

## `close`

```julia
close(op)
```

For resident `HighOrderKernelOp`, `close` synchronizes and releases its workspaces, phase caches, and multi-GPU workers. Derived normals share those resources and must no longer be used. Call it explicitly after `recon_HOOp`.

For `HighOrderLowRankOp`, this releases the distributed execution backend and
the source operator's streaming/regeneration queue when present. Stored
factors and other operator arrays remain owned until their references are
dropped. To release only the distributed normal backend, use:

```julia
release_highorder_normal_backend!(op)
```

`recon_HOOp` releases only the LowRank distributed normal backend by default,
unless `release_backend=false` is requested. Call `close(op)` after the final
use to also close a per-readout source queue. The wrapper does not close
resident explicit operators.

## Backend ownership and reuse

Rank mode initializes its backend on the first forward, adjoint, or normal
request. Channel/readout modes initialize on the first normal request.
The parent operator owns these resources; closing a derived low-rank normal
view alone does not stop the parent's workers.

Backend release retains the parent's factor arrays. In rank mode the complete
factor masters remain in host RAM while referenced. A later use of the parent
can recreate its backend and required streaming/regeneration queue lazily;
obtain a fresh normal view after release.

An owned distributed normal backend (`:channel`, `:readout`, or `:rank`) caches its weights. The `:single` path has no such cache. Before requesting different
normal weights, release it and construct a new normal view. A direct
`normalOperator(op, D)` call expects the already squared, real nonnegative
diagonal of $W^H W$; `recon_HOOp` accepts raw sample weights and applies them
to data and encoding. See [Weighting and cleanup](/guide/multi-gpu#resource-lifecycle).

## Replace an operator

`@rebuild_HOOp` closes the previous value, clears its binding and runs garbage
collection before evaluating the new constructor. Other references to that
operator can still keep arrays alive. If the constructor fails, the old
operator has already been released; the macro is not a rollback mechanism.

<!--@include: ../../.generated/api/rebuild-hoop.md-->

## Release only the distributed backend

<!--@include: ../../.generated/api/release-highorder-normal-backend.md-->

CUDA.jl may retain released capacity in reusable memory pools. A high reserved
memory value alone does not prove that an operator still owns live arrays.
Construction-time allocation failures are a separate cleanup boundary: the
current code does not establish complete recovery at every allocation site.
See [multi-GPU lifecycle](/guide/multi-gpu#resource-lifecycle).
