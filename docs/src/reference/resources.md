# Resource lifecycle

Large CUDA low-rank operators can own persistent workspaces and an optional channel-distributed normal backend. Release these resources explicitly when the operator is no longer required outside `recon_HOOp`.

## `close`

```julia
close(op)
```

For `HighOrderLowRankOp`, this releases the distributed normal backend when present.

Equivalent explicit call:

```julia
release_highorder_normal_backend!(op)
```

`recon_HOOp` performs this release by default after reconstruction unless `release_backend=false` is requested.

## `@rebuild_HOOp`

Safely replace a large operator stored in one variable.

```julia
@rebuild_HOOp HOOp HighOrderLowRankOp(
    grid,
    kspha,
    times;
    arrayType=CuArray,
    gpus=[0, 1, 2],
)
```

The macro closes an existing compatible operator, clears the old binding, performs a full Julia garbage collection, and only then evaluates the replacement constructor. This prevents the old and new large operators from coexisting during construction.

::: info CUDA memory pools
Releasing explicit workspaces does not guarantee that the reserved-memory value reported by `nvidia-smi` decreases immediately because CUDA.jl maintains reusable memory pools.
:::

See [Multi-GPU execution](/guide/multi-gpu#resource-lifecycle) and [Troubleshooting](/guide/troubleshooting) for lifecycle and environment-specific details.

[Source: `HighOrderLowRankOp.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/EncodingOperator/HighOrderLowRankOp.jl)
