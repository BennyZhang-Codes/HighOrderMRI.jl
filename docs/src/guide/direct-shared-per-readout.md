# Direct Shared: Per-readout strategy

Choose a shared spatial snapshot basis, test prefixes of that basis using
sampled residual errors for every readout, then fit all actual time rows for
the selected rank. A separate post-selection audit checks the completed
factors. This is the original `method=:direct_shared, strategy=:per_readout` path.

```julia
report = Ref{Any}()
op = HighOrderLowRankOp(grid, kspha, times;
    method=:direct_shared, strategy=:per_readout,
    fieldmap, csm, mask, arrayType=Array,
    shared_basis_tol=1f-2, shared_rank_max=128,
    joint_snapshots=256, joint_samples=512,
    rsvd_seed=1234, rsvd_chunk=4096,
    joint_basis_report=report,
)
```

`shared_basis_tol` is applied to sampled random-voxel residual phase errors
for each readout. Selection uses 90% of this tolerance before a separate
post-selection audit at the requested tolerance. The algorithm may increase the fitting sample count
within its existing two-level search, but does not use the Global strategy's
parent/snapshot expansion policy. Rank/sample exhaustion or audit failure
throws and fills `joint_basis_report` rather than returning accepted factors.

High-B0-region diagnostics are reported separately. `joint_roi_tol=nothing`
(the original default) leaves the ROI gate disabled, so a random-voxel pass
can still produce a high-B0 warning. Set an explicit positive `joint_roi_tol`
when that sampled ROI must also meet a threshold. Neither check certifies the
entire phase matrix or reconstruction image error.

Validation and audit voxels are drawn outside the fitting pool, but can
overlap each other. Snapshot construction still uses all masked voxels.
Validation and audit times are disjoint when enough unused times exist;
tiny inputs can reuse times, as recorded by `disjoint_times`. This audit
checks the completed factors on a second sample set; it is not an independent
statistical certificate. See the [Per-readout derivation](/theory/joint-adaptive-low-rank#per-readout-strategy).

The fitting pool, seeds, snapshot prefix order, CPU/CUDA factor ownership and
report fields retain their existing behavior. The historical `joint_*` names
continue to tune this strategy. `epsilon_H` and `adaptive_options` belong to
the [Global strategy](/guide/direct-shared-global), not this one.

Use an [Explicit reference](/guide/explicit) and matched reconstruction settings
for image validation. The fixed snapshot control on the
[Direct Shared overview](/guide/direct-shared#fixed-rank-controls) separates
rank acceptance overhead from the corresponding fixed-rank construction.
