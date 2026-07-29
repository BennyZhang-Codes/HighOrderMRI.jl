# HighOrderMRI.jl

HighOrderMRI.jl is a Julia toolbox for non-Cartesian MRI reconstruction with
measured or predicted dynamic higher-order fields. It provides explicit
encoding operators for numerical reference and a matrix-free low-rank
operator for large 2D and 3D problems.

The package supports:

- real solid-harmonic field models through third order;
- static off-resonance, coil sensitivities, and reconstruction masks;
- CPU and CUDA execution;
- an explicit multi-GPU CUDA operator;
- per-dynamic randomized SVD followed by an adaptive global shared spatial
  basis;
- voxel-distributed multi-GPU setup and channel-distributed multi-GPU normal
  operators;
- model-based field/data synchronization and GIRF-based field prediction;
- reconstruction metrics that preserve the raw complex-valued convention.

!!! warning "Research software"
    HighOrderMRI.jl is under active development. Confirm the phase, coordinate,
    normalization, and data-layout conventions against the
    [reconstruction protocol](guide/reconstruction-protocol.md) before using
    it for a scientific comparison.

## Where to begin

1. Follow [Getting started](getting-started.md) to install the package and
   construct a small CPU operator.
2. Read the [expanded encoding model](theory/encoding-model.md) before
   preparing field coefficients.
3. Use [Choose an operator](guide/operators.md) to select an exact or low-rank
   implementation.
4. Follow the [reconstruction workflow](guide/reconstruction.md) to create an
   operator, density weights, solver configuration, and accuracy report.

## Operator overview

| Operator | Approximation | Backend | Intended use |
|---|---|---|---|
| [`HighOrderOp`](@ref) | None | CPU or one CUDA device | Small problems, debugging, derivative operator |
| [`HighOrderOp_Kernel`](@ref) | None | One or more CUDA devices | Explicit numerical reference and GPU validation |
| [`HighOrderLowRankOp`](@ref) | Local rSVD + shared spatial compression | CPU or CUDA; optional multi-GPU stages | Repeated forward/adjoint evaluations and large reconstruction |

All three implement the same positive-phase signal convention and symmetric
normalization:

```math
y_{jdc}
=
\frac{1}{\sqrt{N_v}}
\sum_{v=1}^{N_v}
m_v C_{vc}
\exp\!\left(i2\pi\Phi_{jdv}\right).
```

The operators differ in how they evaluate the phase, not in the target signal
model.

## Documentation design

The documentation follows the same separation used by mature Julia MRI
packages: a short entry path, task-oriented guides, explanation of the
underlying model, generated API pages, and a GitHub Actions deployment through
Documenter.jl. The mathematical pages are synthesized from the project's
Notion design notes and checked against the current implementation and
encoding-convention tests. See [References and source
notes](references.md) for provenance and limitations.
