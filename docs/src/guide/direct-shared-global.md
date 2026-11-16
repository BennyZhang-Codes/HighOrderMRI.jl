# Direct Shared: Global strategy

Construct a snapshot parent, fit coefficients for all actual time rows, then
rotate and recompress the coefficients using blocked FP64 QR and a small SVD.
Rank selection evaluates residuals against the original pure residual phase,
not coefficient-tail energy alone. Select it with
`method=:direct_shared, strategy=:global`.

```julia
report = Ref{Any}()
op = HighOrderLowRankOp(grid, kspha, times;
    method=:direct_shared, strategy=:global,
    fieldmap, csm, mask, arrayType=Array,
    epsilon_H=0.01, shared_rank_max=128,
    adaptive_options=(; snapshots=128, sample_count=512, initial_parent=32),
    rsvd_seed=1234, rsvd_chunk=4096,
    joint_basis_report=report,
)
```

`epsilon_H` targets global relative Frobenius error of the residual phase
matrix over all samples/readouts and masked voxels. First-order Fourier terms,
zeroth-order temporal phase, center corrections, CSM, normalization and
reconstruction weights are excluded from this target. It is not a worst-readout,
ROI or image-error target.

Both selection and acceptance samples participate in selecting the final rank.
The rule `sqrt(mean(loss)+4SE)` is empirical; it is not a probability bound.
`empirical_status=:passed` keeps `certification_status` and
`external_audit_status` at `:not_requested`. A fresh saved-factor audit requires
a separately declared evaluation, not reinterpretation of the selection data.

`shared_rank_max` caps the parent width; the selected final rank cannot exceed
that width. It does not request a particular final rank. On failed
checks, diagnostics may expand fitting samples, parent width or snapshots
within `adaptive_options` budgets. Failure fills the report and the operator
constructor throws. `max_seconds` is checked between attempts, not inside a
CUDA call. Construction-memory estimates are approximate.

`joint_snapshots` and `joint_samples` configure the Per-readout strategy.
For Global, set `snapshots` and `sample_count` in `adaptive_options` instead.
`joint_roi_tol` is unsupported by Global and a non-`nothing` value raises an
error; it cannot be moved into `adaptive_options`. Global uses the global
`epsilon_H` target rather than an ROI gate. The Per-readout strategy and its
`shared_basis_tol` retain their original meanings.

See the [derivation and limitations](/theory/joint-adaptive-low-rank). The
internal coefficient fixed-rank control holds the parent and retained ranks
fixed, separating the Global rank-selection policy from its completed factors.
