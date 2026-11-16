---
layout: home

title: HighOrderMRI.jl
titleTemplate: false

description: GPU-accelerated Cartesian and non-Cartesian MRI reconstruction with dynamic high-order field encoding.

hero:
  name: HighOrderMRI.jl
  text: Higher-order MRI reconstruction
  tagline: Explicit and LowRank encoding operators for Cartesian and non-Cartesian MRI, with off-resonance and higher-order phase correction.
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
    link: /guide/explicit
  - title: LowRank construction methods
    details: Choose Per-readout, Incremental Shared, or Direct Shared with Per-readout and Global strategies.
    link: /guide/low-rank
  - title: Field-monitoring workflow
    details: Combine dynamic field measurements or predictions with synchronization, preprocessing, and iterative reconstruction.
    link: /guide/field-preprocessing
---

::: warning Research software
HighOrderMRI.jl is under active development. Regression tests establish implementation consistency but do not constitute independent physical validation. Use the [validation strategy](/guide/validation) to distinguish numerical from physical evidence and the [reconstruction protocol](/guide/reconstruction-protocol) to keep comparison conventions fixed.
:::

## Common signal model, alternative numerical implementations

HighOrderMRI.jl keeps the target signal model fixed while selecting either
[Explicit](/guide/explicit) or [LowRank](/guide/low-rank) execution.

| Family entry | Numerical representation | Choices |
|:--|:--|:--|
| `HighOrderExplicitOp` | Explicit expanded phase | Array; CUDA acceleration with Direct kernel or GEMM and phase-reuse |
| `HighOrderLowRankOp` | NFFT plus low-rank residual phase | Per-readout, Incremental Shared, Direct Shared |

Direct Shared has Per-readout and Global strategies. Explicit uses Array by
default; LowRank uses Incremental Shared. LowRank factorization and NFFT
interpolation introduce separate numerical errors.

The common signal model is:

$$
y_{jic}
=
\frac{1}{\sqrt{N_v}}
\sum_{v=1}^{N_v}
m_v C_{vc}\exp\!\left(\mathrm{i}2\pi\Phi_{jiv}\right).
$$

A controlled comparison should therefore proceed hierarchically: establish agreement between the explicit implementations, quantify the low-rank approximation against a fixed explicit reference, and then evaluate the complete signal model against an independent numerical or physical reference.

## Suggested reading order

Start with **[Getting Started](/getting-started)** for installation and a minimal operator example. Continue with the **[architecture overview](/concepts-overview)** and **[expanded encoding model](/theory/encoding-model)** before preparing dynamic field coefficients. For reconstruction studies, use **[Choose an operator](/guide/operators)** together with the **[reconstruction workflow](/guide/reconstruction)**. Accuracy and performance results should be interpreted using the **[scientific validation strategy](/guide/validation)** and the fixed conventions in the **[reconstruction protocol](/guide/reconstruction-protocol)**.
