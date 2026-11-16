# Explicit: Direct kernel {#explicit-kernel}

CUDA acceleration provides a Direct kernel path and a
[GEMM and phase-reuse](/guide/explicit-gemm) path. The unchanged API selector
is `implementation=:kernel`. The Direct kernel path supports both host-vector
and GPU-resident execution. Vector residency determines where solver vectors
are stored and when transfers occur; it does not change the selected phase
model.

Direct kernel evaluates each encoding contribution and accumulates it by
direct summation, without storing a sample-by-voxel complex phase matrix.
In a normal application, forward and adjoint evaluate the phase separately.

## Host-vector execution

```julia
using CUDA

reference_op = HighOrderExplicitOp(
    grid,
    kspha,
    times;
    fieldmap,
    csm,
    mask,
    implementation=:kernel,
    arrayType=CuArray,
    gpus=[0, 1],
)
```

With the default `backend=:legacy`, masked voxels are divided into approximately equal contiguous ranges. Each GPU evaluates a partial forward signal; these partial signals are summed on the host. For the adjoint, disjoint voxel results are gathered into their original masked positions. Resident backends instead partition samples across GPUs, as described below. Both decompositions preserve the explicit phase model.

In this mode, solver vectors remain on the CPU. Each operator application
transfers its inputs to the GPU workspaces and gathers the result back to the
CPU. The encoding calculation runs in CUDA kernels. This remains the default
CUDA backend; choosing the unified entry does not enable GPU residency.

For a controlled comparison with another implementation, the grid, mask, field coefficients, coil maps, normalization, and reconstruction settings should be identical.

When `kspha_dt` is supplied, the current derivative forward path falls back to the array-based CPU derivative implementation. Its adjoint remains the ordinary kernel encoding adjoint. Use `HighOrderExplicitOp(...; arrayType=CuArray, kspha_dt=...)` when the derivative forward product itself must execute on CUDA. Derivative mode requires `backend=:legacy` and ignores `concomitant`; ordinary legacy and resident encoding support supplied concomitant phase.

The legacy Float64 adjoint can exceed its fixed CUDA launch register budget.
The resident direct backend caps Float64 adjoint threads separately.


## GPU-resident direct summation {#resident-direct-evaluation}

```julia
op = HighOrderExplicitOp(grid, kspha, times;
    implementation=:kernel, backend=:direct,
    fieldmap, csm, mask, gpus=[0],
)
```

`backend=:direct` evaluates the explicit forward and adjoint with resident
CUDA kernels; CG vectors remain on the primary GPU. It does not form a
complex phase matrix. For matrix products that reuse a phase block, select
[GEMM and phase-reuse](/guide/explicit-gemm) instead. No backend is selected from a
performance heuristic: the default remains the legacy host-vector path.

Call `close(op)` after the final CUDA-operator use. Weighting and resource
ownership follow the common Explicit interface. See the [kernel API](/reference/explicit-cuda).
