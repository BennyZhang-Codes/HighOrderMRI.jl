# LowRank reconstruction

`HighOrderLowRankOp` retains the first-order Fourier terms in an NFFT and
approximates the residual phase using temporal and spatial factors. Choose how
the factors are constructed with `method`; choose their execution separately.

| Method | Construction | Strategy | Detail |
|:--|:--|:--|:--|
| `:per_readout` | Independent local rSVD factors | — | [Per-readout](/guide/per-readout) |
| `:incremental_shared` | Local rSVD followed by incremental spatial compression | — | [Incremental Shared](/guide/incremental-shared) |
| `:direct_shared` | Representative phase snapshots directly form a shared parent | `:per_readout` or `:global` | [Direct Shared](/guide/direct-shared) |

The default `method=:incremental_shared` selects Incremental Shared. Omitting `strategy`
with `method=:direct_shared` selects its Per-readout strategy. `strategy` is
otherwise unset and only valid with Direct Shared.

`method=:per_readout` keeps independent factors.
`method=:direct_shared, strategy=:per_readout` still uses one shared spatial
basis; Per-readout here specifies its rank-selection and checking strategy.

## Input and numerical representation

Coefficients are `(nTerm,nSam,nDyn)` and times are `(nSam,nDyn)`. The convenience
constructor accepts one coefficient matrix and time vector. Every readout
encodes the same unknown full-grid image. Samples vary fastest, then readouts,
then coils. For `recon_HOOp`, reshape data to `(nSam*nDyn,nCha)`.

The pure residual factors exclude the zeroth-order temporal phase, first-order
NFFT, CSM, weights and voxel normalization. The operator folds the required
temporal corrections and normalization once. In 2D, `kz*z` remains a residual
term. See the [encoding model](/theory/encoding-model) for coordinate scope.

## Construction and execution are separate

| Setting | Role |
|:--|:--|
| `method`, `strategy` | Construct and select factors |
| `rsvd_distribution` | Execute local rSVD setup; Direct Shared uses the primary device |
| `per_readout_storage` | Resident, CPU-streamed or regenerated independent factors |
| `normal_distribution` | Assign solver work to one device, coil groups, readout groups or rank groups |
| `normal_method` | Standard or supported batched CUDA execution |
| `normal_options` | Existing opt-in execution resources, including pruned FFT |

Defaults remain CPU arrays, single-device standard execution and unchanged
NUFFT settings. Batched/pruned execution does not become enabled by selecting
a new method name. Per-readout, Incremental Shared and Direct Shared can have
different storage and distribution restrictions; use the
[execution reference](/reference/highorderlowrankop#choose-setup-and-reconstruction-execution-separately)
and [multi-GPU guide](/guide/multi-gpu).

## Construction controls

Use `method` and `strategy` to select the algorithm. `joint_*` options tune
Direct Shared's Per-readout strategy; `adaptive_options` configures Global.
These controls retain their numerical meanings. See the
[complete API](/reference/highorderlowrankop) and [validation guide](/guide/validation).
