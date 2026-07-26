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

## Adding a 3D benchmark

Add `run/run_3d_kernel_vs_lowrank.jl`. It should load and preprocess its own
dataset, construct its 3D `Grid`, and call the helpers in
`common/benchmark_utils.jl`. Those helpers are independent of matrix size and
number of spatial dimensions. Keep 3D display/slice export in the 3D run file
or a 3D-specific helper, not in `common/`.
