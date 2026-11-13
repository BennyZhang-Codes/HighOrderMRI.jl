# Reconstruction benchmarks

Compare explicit `HighOrderKernelOp` encoding with `HighOrderLowRankOp` on
fixed inputs, weights, GPUs, regularization, and CG iterations. Rank sweeps
use shared low-rank/local rSVD, rather than joint adaptive H-rank selection.

```text
benchmark/
├── Project.toml, Manifest.toml   Independent tool environment
├── common/                      Timing, input, provenance, and CSV utilities
├── run/                         Executable benchmark entry points
├── analysis/                    Offline plots
├── test/                        CPU regression tests
└── results/                     Generated outputs (not versioned)
```

## Environment and tests

Run from the repository root with Julia 1.12:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg \
    julia --startup-file=no --project=benchmark benchmark/test/runtests.jl
```

The benchmark environment declares MAT, CSV, NIfTI, and plotting dependencies
without adding them to the package's runtime dependencies. Its source entries
load HighOrderMRI and MRIGeometry from this checkout. The manifest freezes the
resolved tool environment; root dependency files are unchanged.

## Experiments

Datasets and saved references are external inputs. Select them using
`HIGHORDER_BENCHMARK_DATA`, `HIGHORDER_2D_CC_DATA`, or
`HIGHORDER_3D_DATA`/`HIGHORDER_3D_REFERENCE`; local default paths need not exist
on another machine.

| Experiment | Run script | Analysis script |
|---|---|---|
| Fixed 2D Explicit/LowRank | `run/run_2d_kernel_vs_lowrank.jl` | Images and CSVs saved by the run |
| 2D local rank/seed | `run/sweep_2d_lowrank_parameters.jl` | `analysis/plot_2d_lowrank_sweep.jl` |
| 0.5-mm fixed comparison | `run/run_2d_highres_kernel_vs_lowrank.jl` | Images and CSVs saved by the run |
| 0.5-mm local rank/seed | `run/sweep_2d_highres_lowrank_parameters.jl` | `analysis/plot_2d_highres_lowrank_sweep.jl` |
| 2D GPU scaling | `run/run_2d_kernel_vs_lowrank_gpus.jl` | `analysis/plot_2d_kernel_vs_lowrank_gpus.jl` |
| 2D coil compression | `run/sweep_2d_coil_compression.jl` | `analysis/plot_2d_coil_compression.jl` |
| 3D rank/tolerance | `run/sweep_3d_lowrank_parameters.jl` | `analysis/plot_3d_lowrank_sweep.jl`, `analysis/plot_3d_rank_mid_slices.jl` |
| 3D coil compression | `run/sweep_3d_coil_compression.jl` | `analysis/plot_3d_coil_compression.jl` |

For example:

```bash
HIGHORDER_HIGHRES_SEQUENCE=spiral HIGHORDER_BENCHMARK_GPUS=2,4,5,6 \
    julia --project=benchmark --threads=4 benchmark/run/sweep_2d_highres_lowrank_parameters.jl

HIGHORDER_2D_CC_GPUS=2,4,5,6 \
    julia --project=benchmark --threads=4 benchmark/run/sweep_2d_coil_compression.jl

HIGHORDER_3D_GPUS=2,4,5,6 \
    julia --project=benchmark --threads=4 benchmark/run/sweep_3d_lowrank_parameters.jl

julia --project=benchmark benchmark/analysis/plot_3d_lowrank_sweep.jl /path/to/completed/run
```

Select `HIGHORDER_HIGHRES_SEQUENCE=spiral` or `epi`. Override the high-resolution
input with `HIGHORDER_HIGHRES_SOURCE`; its 200×200 CSM/B0/mask calibration is
resized to the acquisition's 400×400 grid. Preparation uses a separate temporary
MAT file and preserves the raw input. Both high-resolution run scripts reuse
the ordinary 2D constructors, sweep, metrics, and plotting logic.

The 2D loader keeps its established image-array transform separate from Grid.
The 3D loader owns its DCS-to-RPS trajectory conversion. These dataset-specific
conventions are not interchangeable. The 3D scripts use a saved explicit image;
they do not recompute an explicit reference with current code.

## Recovery and result selection

New runs use timestamped directories. The two coil-compression scripts and the
3D rank sweep save a `run_contract.toml` before exporting results. It records
SHA256 fingerprints of inputs, local source, environment manifests, and
reconstruction/timing configuration, including GPUs, seed, lambda, CG
iterations, rank settings, and the fixed NFFT backend.

To resume, set `HIGHORDER_2D_CC_RUN_DIR` or `HIGHORDER_3D_RUN_DIR` to that run
and retain the same numerical/timing configuration. Mismatches stop before
replacing configuration, images, or checkpoints. Legacy directories without
a contract are rejected; inspect them read-only or start a new directory.
The 2D rank sweep writes progress CSVs but does not implement resume.

For the 3D rank sweep, `HIGHORDER_3D_RERUN_CONFIGS=L06_tol5e-2_seed1234`
replaces that configuration within a matching run. `HIGHORDER_3D_RETRY_ERRORS`
controls retrying failed configurations. Presentation-only difference scaling
may change without changing the reconstruction contract.

Analysis accepts an explicit result directory. Automatic selection requires
the corresponding final CSVs; the middle-slice plot also checks all requested
ranks and image files. The high-resolution plotting entry selects names
containing `_0p5_`; use an explicit path for custom dataset names.

## Error and timing definitions

Raw complex NRMSE, magnitude NRMSE, and SSIM are computed without a fitted
scaling. Default reconstruction and difference images also use the raw
reconstruction. Explicitly named `aligned` outputs/diagnostics in the fixed
2D comparison are supplemental and must not replace raw error evidence.
Magnitude differences are `scale * abs(abs(x) - abs(reference))` on the
reference display range. The 3D middle-slice plot converts the recorded
saved scale to its labeled 20× scale.

NFFT is fixed to the NFFT.jl backend, with the package's default accuracy
settings. There is no environment-dependent backend fallback. Input loading,
SHA256 validation, metric evaluation, image export, and file I/O are outside
reconstruction timing. GPU stages synchronize all selected devices before
and after measurement. Free-memory snapshots are not monitored peaks.

| Measurement | Included cost |
|---|---|
| 2D fixed/rank and 3D rank `total_s` | Fresh operator setup + complete `recon_HOOp` call |
| 2D coil `operator_total_s` | Fresh operator setup + reconstruction |
| 2D coil `end_to_end_s` | Compression application + input preparation/transfer + operator total |
| 2D coil `total_including_shared_fit_s` | Separately measured coil SVD fit + end-to-end stages |
| 3D coil `total_s` | Coil update + reconstruction using the common phase operator |
| 3D coil `total_with_compression_s` | Compression application + transfer + reused-operator total |
| 3D coil `accounted_total_s` | The above + one common phase setup, coil fit, and original device-to-host transfer |

Totals are sums of measured stages, rather than a timer around the whole
script. Input loading, contract validation, and export are excluded as stated
above. Reconstruction timing includes solver creation and lazy normal-backend setup.
The fixed 2D comparison includes the explicit operator's GPU-to-host input
conversion in reconstruction time; the 2D coil sweep starts with host data
and reports LowRank input transfer separately.
The 3D coil phase basis is built once and reused across coil counts; its
accounted total is a sum of measured stages, not repeated fresh-build wall
time. Coil-fit/common-setup measurements may include their first JIT.
Repeated timings do not provide independent rSVD seeds. Keep first-call,
warm fresh, and reused-phase results separate when reporting speedups.
