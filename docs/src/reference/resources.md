# Resource lifecycle

Resident explicit operators own CUDA workspaces and optional phase caches; CUDA low-rank operators can own persistent workspaces and an optional distributed normal backend. Release these resources explicitly when the operator is no longer required outside `recon_HOOp`.

## `close`

```julia
close(op)
```

For resident `HighOrderKernelOp`, `close` synchronizes and releases its workspaces, phase caches, and multi-GPU workers. Derived normals share those resources and must no longer be used. Call it explicitly after `recon_HOOp`.

For `HighOrderLowRankOp`, this releases the distributed normal backend and
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

[Source: `multi_gpu_normal_operator.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/main/src/EncodingOperator/LowRankApproximation/multi_gpu_normal_operator.jl)
