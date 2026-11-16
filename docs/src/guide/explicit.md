# Explicit reconstruction

`HighOrderExplicitOp` is the Explicit family entry. It returns an existing
concrete operator; it adds neither a new numerical model nor a workspace layer.

| Implementation | Selection | Solver storage | Detail |
|:--|:--|:--|:--|
| Array | `implementation=:array` | Selected by `arrayType` | [Array](/guide/explicit-array) |
| Direct kernel (host vectors) | `implementation=:kernel, backend=:legacy` | Host | [Direct kernel](/guide/explicit-kernel) |
| Direct kernel (GPU resident) | `implementation=:kernel, backend=:direct` | Primary GPU | [Direct kernel](/guide/explicit-kernel#resident-direct-evaluation) |
| GEMM and phase-reuse (blocked) | `implementation=:kernel, backend=:blocked` | Primary GPU | [Phase reuse](/guide/explicit-gemm) |
| GEMM and phase-reuse (cached) | `implementation=:kernel, backend=:cached` or `:cached_full` | Primary GPU | [Caching](/guide/explicit-gemm) |

The method tree distinguishes Array from CUDA acceleration. The
CUDA acceleration branch uses the existing selector `implementation=:kernel`; this name
and its behavior are unchanged. Its computation paths are Direct kernel and GEMM and phase-reuse. `backend` chooses their execution modes.
`phase_backend` chooses how matrix phase blocks are generated; `gemm_backend` chooses their matrix multiplication
implementation. These choices do not introduce low-rank truncation.

The default family selection is Array with the original `arrayType=CuArray`.
For a CPU reference explicitly pass `arrayType=Array`. The CUDA
implementation uses `backend=:legacy` by default. All Explicit operators
are constructed through `HighOrderExplicitOp`.

## Readout layout

Explicit implementations accept `(nTerm,N)` coefficients and `N` times.
For several readouts of one image, flatten the sample/readout axes with samples
varying fastest. Flatten data in the same order, retaining coils as the final
axis. This combines all measurements without deleting samples or changing the
physical model. LowRank also accepts a separate readout axis directly.

## Common contracts

All paths use the [same physical model](/theory/encoding-model), mask and
symmetric voxel normalization. Pass raw square-root sample weights to
`recon_HOOp`; `normalOperator(op,D)` instead receives the already squared
diagonal. Derivative `kspha_dt` modes have a forward-only limitation described
in the Array and Direct kernel pages and must not be used for
ordinary CG normals.

Use the [Explicit API](/reference/highorderexplicitop) for the family wrapper,
the [reconstruction workflow](/guide/reconstruction) for solving, and the
[resource reference](/reference/resources) for CUDA lifecycle management.
