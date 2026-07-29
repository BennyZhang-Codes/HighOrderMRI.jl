# Expanded encoding model

This page defines the mathematical and array conventions implemented by
`HighOrderOp`, `HighOrderOp_Kernel`, and `HighOrderLowRankOp`. These
conventions are part of the operator definition; changing them changes the
scientific model.

## Dynamic field representation

Let ``\mathbf r=(x,y,z)^T`` be a physical voxel coordinate and let
``t_{jd}`` be sample ``j`` in dynamic ``d``. The accumulated phase in cycles
is

```math
\Phi_{jd}(\mathbf r)
=
k_{0,jd}
+ t_{jd}\Delta f_0(\mathbf r)
+ \sum_{\ell=1}^{N_\mathrm{term}-1}
k_{\ell,jd} b_\ell(\mathbf r).
```

Here:

- ``\Delta f_0`` is the static off-resonance map;
- ``b_\ell`` is an unnormalized real solid-harmonic polynomial;
- ``k_{\ell,jd}`` is the time-integrated coefficient for that basis;
- ``k_{0,jd}`` is a spatially uniform dynamic phase.

The implementation evaluates

```math
\exp\!\left(+i2\pi\Phi_{jd}(\mathbf r)\right).
```

Everything inside the exponential must therefore be in **cycles**.
`fieldmap` in hertz multiplied by `times` in seconds already satisfies this
rule. If field-camera phase is supplied in radians, convert it to cycles before
constructing an operator.

## Solid-harmonic basis order

Julia uses one-based row indices in `kspha`. The first 16 implemented terms
are:

| Julia row | Order | Spatial basis |
|---:|---:|---|
| 1 | 0 | ``1`` |
| 2 | 1 | ``x`` |
| 3 | 1 | ``y`` |
| 4 | 1 | ``z`` |
| 5 | 2 | ``xy`` |
| 6 | 2 | ``zy`` |
| 7 | 2 | ``3z^2-(x^2+y^2+z^2)`` |
| 8 | 2 | ``xz`` |
| 9 | 2 | ``x^2-y^2`` |
| 10 | 3 | ``3yx^2-y^3`` |
| 11 | 3 | ``xyz`` |
| 12 | 3 | ``[5z^2-(x^2+y^2+z^2)]y`` |
| 13 | 3 | ``5z^3-3z(x^2+y^2+z^2)`` |
| 14 | 3 | ``[5z^2-(x^2+y^2+z^2)]x`` |
| 15 | 3 | ``x^2z-y^2z`` |
| 16 | 3 | ``x^3-3xy^2`` |

These polynomials are not normalized spherical harmonics. Coefficients
defined for another normalization or coordinate system cannot be inserted
without conversion. When coordinates are in metres, typical accumulated
coefficient units are cycles, cycles/m, cycles/m², and cycles/m³ for orders
zero through three.

Use [`basisfunc_spha`](@ref) when an implementation-consistent basis matrix is
needed.

## Discrete multi-coil model

For masked voxel ``v`` and receive coil ``c``, the signal is

```math
y_{jdc}
=
\frac{1}{\sqrt{N_v}}
\sum_{v=1}^{N_v}
m_v C_{vc}
\exp\!\left\{
i2\pi\left[
k_{0,jd}
+t_{jd}\Delta f_{0,v}
+\sum_{\ell=1}^{N_\mathrm{term}-1}
k_{\ell,jd}b_{\ell,v}
\right]\right\}
+\varepsilon_{jdc}.
```

The factor ``1/\sqrt{N_v}`` is symmetric between the forward and adjoint
operators. Do not fit an extra global intensity or phase merely to make two
operators agree; use the raw complex comparison first.

## Fourier and residual separation

For one dynamic, define:

```math
\begin{aligned}
D_0(j,j) &= \exp(i2\pi k_{0,j}), \\
F(j,v) &= \exp\!\left(i2\pi\mathbf k_j^{(1)T}\mathbf r_v\right), \\
H(j,v) &= \exp\!\left\{i2\pi\left[
t_j\Delta f_{0,v}
+\sum_{\ell\in\mathcal R} k_{\ell,j}b_{\ell,v}
\right]\right\}.
\end{aligned}
```

The total encoding matrix is

```math
E = D_0(F\odot H),
```

where ``\odot`` is the Hadamard product.

`HighOrderOp` and `HighOrderOp_Kernel` evaluate the complete phase directly.
`HighOrderLowRankOp` evaluates ``F`` with an NFFT and approximates only the
residual matrix ``H``.

### 2D versus 3D

- In 3D, rows 2–4 (`x`, `y`, and `z`) define the 3D NFFT trajectory. The
  residual begins with row 5.
- In 2D, rows 2–3 (`x` and `y`) define the 2D NFFT trajectory. Row 4 remains
  in the residual so that ``k_z(t)z`` is retained for an off-centre slice.

This 2D assignment is required for agreement with the explicit model.

## NFFT sign and centring

AbstractNFFTs uses a negative Fourier exponent on an integer-centred grid.
The explicit HighOrderMRI model uses a positive physical phase. For active
axis ``i``, the NFFT node is therefore

```math
\xi_{i,jd}=-k_{i,jd}\Delta_i,
```

with a separate voxel spacing ``\Delta_i`` for each axis.

`nfft_center_correction=true` also applies the parity-dependent temporal
phase that aligns the integer-centred NFFT grid with the physical centres
created by [`Grid`](@ref):

```math
r_i[n] = \left[n-\frac{N_i+1}{2}\right]\Delta_i.
```

Even axes require a half-voxel correction; odd axes do not.

## Array layout

| Quantity | Explicit operators | Low-rank operator |
|---|---|---|
| `kspha` | `(nTerm, nSam)` | `(nTerm, nSam, nDyn)` |
| `times` | `(nSam,)` | `(nSam, nDyn)` |
| `fieldmap` | `(nX,nY[,nZ])` | same |
| `csm` | `(nX,nY[,nZ],nCha)` | same |
| `mask` | `(nX,nY[,nZ])` | same |
| logical output | `(nSam,nCha)` | `(nSam,nDyn,nCha)` |

Julia column-major vectorization makes samples the fastest-changing index,
then dynamics, then channels.

## Selecting field orders

`recon_terms` is a binary string:

- 9 terms: three digits for zeroth, first, and second order;
- 16 terms: four digits for zeroth through third order.

Disabling zeroth, second, or third order zeros those coefficients. Disabling
first order replaces rows 2–4 with `k_nominal`; it does not set the trajectory
to zero. The input `kspha` is copied before selection and is not mutated.

Use the same `recon_terms`, `k_nominal`, mask, coordinate transformations, and
normalization when comparing implementations. The full frozen convention is
recorded in the [reconstruction
protocol](../guide/reconstruction-protocol.md).
