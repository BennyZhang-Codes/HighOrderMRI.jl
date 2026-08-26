---
layout: home

title: HighOrderMRI.jl
titleTemplate: false

description: GPU-accelerated Cartesian and non-Cartesian MRI reconstruction with dynamic high-order field encoding.

hero:
  name: HighOrderMRI.jl
  text: High-order field-aware MRI reconstruction
  tagline: A Julia framework for Cartesian and non-Cartesian MRI reconstruction using measured or predicted dynamic magnetic fields, with explicit, CUDA-kernel, and low-rank implementations of a common encoding model.
  actions:
    - theme: brand
      text: Get Started
      link: /getting-started
    - theme: alt
      text: Explore Theory
      link: /theory/encoding-model
    - theme: alt
      text: Validation Strategy
      link: /guide/validation

features:
  - title: Dynamic field encoding
    details: Represent measured or predicted field evolution with real solid-harmonic terms through third order.
    link: /theory/encoding-model
  - title: Explicit GPU encoding
    details: Evaluate the expanded signal model with array-based or fused CUDA implementations, including supported multi-GPU execution.
    link: /guide/operators
  - title: Shared spatial subspace
    details: Approximate dynamic residual encoding using per-dynamic rSVD and an incrementally constructed spatial basis shared across dynamics.
    link: /theory/low-rank
  - title: Field-monitoring workflow
    details: Combine dynamic field measurements or predictions with synchronization, preprocessing, and iterative reconstruction.
    link: /guide/field-preprocessing
---

::: warning Research software
HighOrderMRI.jl is under active development. Regression tests establish implementation consistency but do not constitute independent physical validation. Use the [validation strategy](/guide/validation) to distinguish numerical from physical evidence and the [reconstruction protocol](/guide/reconstruction-protocol) to keep comparison conventions fixed.
:::

## Common signal model, alternative numerical implementations

HighOrderMRI.jl keeps the target MRI signal model fixed while changing how the encoding operator is evaluated. `HighOrderOp` and `HighOrderKernelOp` are explicit implementations of the expanded model. `HighOrderLowRankOp` introduces a controlled approximation only in the residual spatial phase representation.

| Operator | Numerical representation | Backend | Typical use |
| --- | --- | --- | --- |
| `HighOrderOp` | Explicit expanded encoding | CPU or array-based CUDA | Small reference problems, debugging, derivative products |
| `HighOrderKernelOp` | Explicit expanded encoding with fused kernels | CUDA, including supported multi-GPU execution | Explicit GPU calculations and implementation-consistency testing |
| `HighOrderLowRankOp` | Per-dynamic rSVD + incremental shared spatial basis | CPU/CUDA with optional multi-GPU stages | Repeated large-scale forward/adjoint evaluations and iterative reconstruction |

All three use the same positive-phase signal convention:

$$
y_{jdc}
=
\frac{1}{\sqrt{N_v}}
\sum_{v=1}^{N_v}
m_v C_{vc}\exp\!\left(i2\pi\Phi_{jdv}\right).
$$

A controlled comparison should therefore proceed hierarchically: establish agreement between the explicit implementations, quantify the low-rank approximation against a fixed explicit reference, and then evaluate the complete signal model against an independent numerical or physical reference.

## Suggested reading order

Start with **[Getting Started](/getting-started)** for installation and a minimal operator example. Continue with the **[architecture overview](/concepts-overview)** and **[expanded encoding model](/theory/encoding-model)** before preparing dynamic field coefficients. For reconstruction studies, use **[Choose an operator](/guide/operators)** together with the **[reconstruction workflow](/guide/reconstruction)**. Accuracy and performance results should be interpreted using the **[scientific validation strategy](/guide/validation)** and the fixed conventions in the **[reconstruction protocol](/guide/reconstruction-protocol)**.
