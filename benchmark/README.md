# Reconstruction benchmarks

This directory compares `HighOrderOp_Kernel` with `HighOrderLowRankOp` under a
fixed dataset, operator convention, GPU set, and CG iteration count.

```text
benchmark/
├── common/       Dimension-independent timing, CUDA, CSV, and error utilities
├── run/          Dataset preparation and executable 2D/3D benchmark entry points
├── analysis/     Offline result plotting and table analysis
└── results/      Generated CSV, figures, and checkpoints (not versioned)
```

## Current 2D entries

- `run/run_2d_kernel_vs_lowrank.jl`: fixed-configuration Kernel vs LowRank comparison.
- `run/sweep_2d_lowrank_parameters.jl`: single-shot local-LowRank rank/seed sweep, using one Kernel reference. It fixes `shared_basis_tol=0` and oversampling to `5`.
- `analysis/plot_2d_lowrank_sweep.jl`: plots rank–error, rank–time, Pareto, and seed-stability figures for one completed local-rank sweep.

Run the fixed comparison from the repository root:

```bash
julia --project=. --threads=4 benchmark/run/run_2d_kernel_vs_lowrank.jl
```

The 2D run script owns the MAT-file preprocessing and its image-coordinate
convention. In particular, `Grid` remains in its standard orientation while
`csm`, `fieldmap`, and `mask` receive the established array-layout transform.
Do not move this convention into `common/`: a 3D dataset may need a different
layout without changing generic timing or metric code.

## 3D benchmark

`run/sweep_3d_lowrank_parameters.jl` benchmarks the saved 3D Kernel reference
against eleven unique LowRank configurations:

- ranks `2, 4, 6, 8, 10, 15, 20, 25` at `shared_basis_tol=1e-2`;
- tolerances `5e-2, 2e-2, 1e-2, 5e-3` at rank `6`.

It never constructs `HighOrderOp_Kernel`; instead, it reads
`20260728_HighOrderOp_Kernel.mat`. A full rank-2 warmup precedes the timed
configurations. Timings report LowRank operator setup separately from the
complete fixed-iteration `recon_HOOp` call.

Every successful configuration saves:

- a magnitude PNG using the Kernel reference's fixed 99.9th-percentile `vmax`;
- magnitude and phase NIfTI volumes;
- a `10 ×` magnitude absolute-difference PNG and NIfTI using the same display
  range. Override the scale with `HIGHORDER_3D_DIFFERENCE_SCALE`.

The reference PNG and magnitude/phase NIfTI volumes are saved once. CSV and
serialized checkpoints allow an interrupted sweep to resume without repeating
completed configurations.

Run from the repository root:

```bash
julia --project=. --threads=5 benchmark/run/sweep_3d_lowrank_parameters.jl
```

Override the GPU list or resume into a specific run directory with:

```bash
HIGHORDER_3D_GPUS=2,3,4,5,6 \
HIGHORDER_3D_RUN_DIR=/path/to/3d_lowrank_sweep_run \
julia --project=. --threads=5 benchmark/run/sweep_3d_lowrank_parameters.jl
```

Rerun and replace one completed configuration without touching the others:

```bash
HIGHORDER_3D_RUN_DIR=/path/to/3d_lowrank_sweep_run \
HIGHORDER_3D_WARMUP=false \
HIGHORDER_3D_RERUN_CONFIGS=L06_tol5e-2_seed1234 \
julia --project=. --threads=5 benchmark/run/sweep_3d_lowrank_parameters.jl
```

Generate rank/tolerance summary figures without rerunning reconstruction:

```bash
julia --project=. benchmark/analysis/plot_3d_lowrank_sweep.jl \
    benchmark/results/3d_lowrank_sweep_<timestamp>
```

The 3D run owns its dataset layout and DCS-to-RPS trajectory conversion.
Dimension-independent timing, CUDA, CSV, and error helpers remain in
`common/benchmark_utils.jl`.
