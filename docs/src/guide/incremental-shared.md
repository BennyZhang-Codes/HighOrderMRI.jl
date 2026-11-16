# Incremental Shared LowRank

Incremental Shared first builds local per-readout rSVD factors, then merges
and compresses their spatial directions into a common basis. This is the
existing default LowRank method, previously labelled Shared low-rank.

```julia
op = HighOrderLowRankOp(grid, kspha, times;
    method=:incremental_shared,
    fieldmap, csm, mask, arrayType=Array,
    L_rank=8, rsvd_oversample=5, rsvd_seed=1234,
    shared_basis_tol=1f-2, shared_rank_max=128,
)
```

`L_rank` is a local retained rank, not the completed shared rank.
`shared_basis_tol` controls incremental compression of retained local factors;
it is not an original full-phase or reconstructed-image error bound. Earlier
coefficient blocks are zero-padded when the basis grows; the default path does
not refit them onto the completed basis. See the
[derivation](/theory/low-rank).

## Optional sampled construction and refitting

```julia
op = HighOrderLowRankOp(grid, kspha, times;
    method=:incremental_shared,
    fieldmap, csm, mask, arrayType=CuArray, gpus=[0,1],
    rsvd_distribution=:voxel, rsvd_backend=:kernel, rsvd_finalize=:gram,
    shared_readout_count=64, shared_fit_samples=512,
    shared_basis_report=Ref{Any}(),
)
```

These experiment budgets are not general accuracy recommendations. The basis
is trained using representative readouts; coefficients are fitted for every
original time sample. Defaults `shared_readout_count=nothing` and
`shared_fit_samples=nothing` keep the original all-readout construction.
The [refitting equations](/theory/low-rank#sampled-construction-and-coefficient-refitting)
explain why this option has different error conditions from the original
local-projection construction.

## Optional final recompression

`global_basis_rank` fixes the final shared rank; `global_basis_tol` instead
controls truncation relative to the existing shared factors. They are mutually
exclusive. `global_basis_weights=D` uses already squared nonnegative sample
weights for compression; `global_basis_coil_weighting=true` also uses CSM
energy. These compression weights do not replace reconstruction weights.
See [weighted recompression](/theory/low-rank#fixed-rank-and-weighted-recompression)
for the metric and factor transformations.

Sampled fitting, fixed final rank, weighted compression and the no-add check
are opt-in. They do not automatically enable each other. Inspect phase checks
and complete reconstruction errors rather than interpreting a factor-tail
bound as an image-error guarantee. These are distinct from Direct Shared's
Global strategy, which rotates coefficients of a snapshot parent and checks
residuals against the original phase model.

The completed shared factors support the existing standard, batched and
opt-in pruned CUDA execution subject to documented restrictions. See
[setup/execution parameters](/reference/highorderlowrankop) and
[multi-GPU execution](/guide/multi-gpu).
