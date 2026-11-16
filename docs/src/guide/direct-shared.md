# Direct Shared LowRank

Direct Shared constructs a common spatial parent directly from representative
residual phase snapshots across readouts, then fits temporal coefficients.
It does not first construct and merge local per-readout rSVD factors.
The historical implementation name is Joint.

**Direct refers to shared-basis construction.** It does not mean an explicit
phase summation, the Explicit CUDA `backend=:direct`, or a full-matrix exact SVD.

| Strategy | Ordering and construction | Rank acceptance |
|:--|:--|:--|
| [Per-readout](/guide/direct-shared-per-readout) | Snapshot prefixes; all-time fitting after rank selection | Sampled per-readout checks, then post-selection audit |
| [Global](/guide/direct-shared-global) | Fit all time rows to a parent; blocked FP64 QR/SVD rotates and recompresses its coefficients | Paired global residual checks and bounded expansion |

The strategies share phase features, phase-geometry snapshot selection,
spatial snapshot construction, fitting-point selection, FP64 QR coefficient
maps and coefficient evaluation. Their fitting pools and rank orderings remain
distinct. They differ in construction as well as acceptance.

```julia
op = HighOrderLowRankOp(grid, kspha, times;
    method=:direct_shared, strategy=:per_readout,
    fieldmap, csm, mask, arrayType=Array,
)
```

Omitting `strategy` here preserves the Per-readout strategy. A Global strategy
must explicitly supply `epsilon_H`. Both return the same public low-rank
operator representation and use the existing NFFT execution machinery.
Post-validation Incremental Shared recompression options are rejected for
Direct Shared; they must not silently invalidate the reported checks.

## Fixed-rank controls

The internal `HighOrderMRI.joint_fixed_spatial_basis` control supports
`ordering=:snapshot` and `:coefficient`. The coefficient control fixes both
parent rank and retained rank. It returns host residual factors and performs
no rank search, acceptance test or expansion. Its
`empirical_status=:not_evaluated` is not a tolerance pass.

Factor-level controls are internal tools for testing construction and rank
selection. Configure operators through `method` and `strategy`. See the
[Global derivation](/theory/joint-adaptive-low-rank),
[API](/reference/highorderlowrankop) and [LowRank method guide](/guide/low-rank).
