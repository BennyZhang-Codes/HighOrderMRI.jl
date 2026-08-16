# Low-rank shared subspace

`HighOrderLowRankOp` accelerates repeated applications of the expanded
encoding model with two compression stages:

1. a matrix-free randomized SVD of each dynamic's residual phase matrix;
2. an adaptive shared spatial basis across the retained local factors.

It does not approximate the image, coil sensitivities, or first-order Fourier
trajectory.

## Residual encoding matrix

For dynamic ``d``, define

```math
H_d(j,v)
=
\exp\!\left\{
i2\pi\left[
t_{jd}\Delta f_{0,v}
+\sum_{\ell\in\mathcal R}
k_{\ell,jd}b_{\ell,v}
\right]\right\},
\qquad
H_d\in\mathbb C^{N_s\times N_v}.
```

The zeroth-order temporal phase and the first-order Fourier terms are kept
outside ``H_d``. The explicit operators evaluate this matrix element by
element. The low-rank operator never materializes it.

## Per-dynamic randomized SVD

Let ``L`` be `L_rank`, ``p`` be `rsvd_oversample`, and
``\ell=L+p``. For each dynamic, the setup computes:

```math
\begin{aligned}
\Omega_d &\in \mathbb C^{N_v\times\ell}, \\
Y_d &= H_d\Omega_d, \\
Q_d &= \operatorname{orth}(Y_d), \\
B_d &= H_d^H Q_d.
\end{aligned}
```

Both multiplications by ``H_d`` and ``H_d^H`` are evaluated matrix-free. The
seed schedule is deterministic for a fixed configuration:

```math
s_d=s_0+d-1.
```

### SVD finalization

For `rsvd_finalize=:svd`,

```math
B_d=P_d\Sigma_d Z_d^H,
```

and the retained factors are

```math
U_d=Q_d Z_{d,L},
\qquad
\widetilde V_d=P_{d,L}\Sigma_{d,L}.
```

Thus

```math
H_d\approx U_d\widetilde V_d^H.
```

The singular values are absorbed into the spatial factor so that the shared
compression sees the physical energy of each retained mode.

### Small-Gram finalization

For `rsvd_finalize=:gram`, the setup diagonalizes the small Hermitian matrix

```math
G_d=B_d^HB_d\in\mathbb C^{\ell\times\ell}.
```

If ``G_d=Z_d\Lambda_d Z_d^H``, then

```math
U_d=Q_d Z_{d,L},
\qquad
\widetilde V_d=B_d Z_{d,L}.
```

This avoids a direct SVD of the tall ``N_v\times\ell`` matrix and is required
by the voxel-distributed setup. It also squares the condition number, so the
implementation checks for non-finite values, significant negative
eigenvalues, and degeneracy. The single-device path can fall back to the
conventional SVD when the Gram result is not trustworthy.

## Global shared spatial basis

Independent local factors would require ``N_dL`` spatial modes. Instead,
HighOrderMRI builds an orthonormal basis

```math
S\in\mathbb C^{N_v\times R},
\qquad
S^HS=I,
```

such that

```math
\widetilde V_d\approx S C_d,
\qquad
C_d=S^H\widetilde V_d.
```

For each new dynamic, the algorithm twice reorthogonalizes the unexplained
residual

```math
R_d=\widetilde V_d-SC_d
```

and diagonalizes the small residual Gram matrix ``R_d^H R_d``. It appends the
minimum number of new directions needed to satisfy

```math
\sqrt{
\frac{\sum_{i=r_d+1}^{L}\lambda^{(R)}_{d,i}}
     {\lVert\widetilde V_d\rVert_F^2}
}
\leq \tau,
```

where ``\tau`` is `shared_basis_tol`. If the tolerance requires more than
`shared_rank_max` directions, setup fails instead of silently violating the
requested bound.

The local rank ``L`` and shared rank ``R`` answer different questions:

- ``L`` controls the approximation of each individual ``H_d``;
- ``R`` is the joint dimension needed by all retained spatial factors.

Consequently, ``R`` may be smaller than, equal to, or larger than ``L``.

## Final operator

Define

```math
q_d=U_d C_d^H\in\mathbb C^{N_s\times R}.
```

Then

```math
H_d\approx q_d S^H.
```

All ``q_d`` matrices are concatenated over samples and dynamics. The
zeroth-order phase, NFFT centre correction, and ``1/\sqrt{N_v}``
normalization are folded into `q` once during setup.

For shared basis column ``s_r`` and sample-domain coefficient ``q_r``, one
coil's forward operation has the form

```math
A_c m
\approx
\sum_{r=1}^{R}
\operatorname{diag}(q_r)
\mathcal F
\left(m\odot C_c\odot s_r^*\right).
```

The implementation uses one global NFFT plan containing the trajectories of
all dynamics. The transform count per forward or adjoint evaluation is

```math
R N_c,
```

instead of approximately ``N_d L N_c`` for independent per-dynamic factors.
This is a transform-count statement, not a promise that runtime is independent
of the number of samples or dynamics.

## Two-stage error budget

The approximation is hierarchical:

```math
H_d
\longrightarrow
U_d\widetilde V_d^H
\longrightarrow
q_dS^H.
```

A useful bound is

```math
\lVert H_d-q_dS^H\rVert_F
\leq
\underbrace{\lVert H_d-U_d\widetilde V_d^H\rVert_F}_{\text{local rSVD}}
+
\underbrace{\lVert\widetilde V_d-SC_d\rVert_F}_{\text{shared compression}}.
```

The incremental shared basis is scalable and memory bounded, but it is not
claimed to be the globally optimal rank-``R`` basis of all dynamics. Its
finite-precision result can also depend slightly on dynamic order near the
tolerance threshold.

## Parameter interpretation

| Parameter | Role | Practical check |
|---|---|---|
| `L_rank` | Local truncation rank | Sweep against an explicit operator or independent dense reference |
| `rsvd_oversample` | Extra random sketch directions | Verify `L_rank + rsvd_oversample ≤ min(nSam,nVox)` |
| `rsvd_seed` | Reproducible sketch schedule | Repeat with several seeds for a final study |
| `rsvd_finalize` | `:svd` or memory-saving `:gram` | Compare spectra and operator errors on a tractable problem |
| `shared_basis_tol` | Second-stage relative residual tolerance | Report it together with final `shared_rank` |
| `shared_rank_max` | Hard cap on shared rank | Treat a cap error as a configuration failure |
| `rsvd_backend` | `:chunked` or fused CUDA `:kernel` | The fused kernel requires ``L+p\leq32`` |

Do not select rank solely from one reconstruction image. At minimum, validate
forward error, adjointness, normal-operator error, reconstruction error, seed
sensitivity, and the final solver residual. The [reconstruction
protocol](../guide/reconstruction-protocol.md) defines the primary metrics and
timing boundaries.

## Evidence boundary

Current regression tests establish internal agreement of coordinate, phase,
centering, normalization, masking, data layout, and adjoint conventions on
small problems. Project research notes also contain large 3D comparisons
against the explicit CUDA operator. Those comparisons support the engineering
case for the shared representation, but an explicit operator is not an
independent physical gold standard. Claims of absolute physical accuracy still
require independent simulation or measured-reference validation.
