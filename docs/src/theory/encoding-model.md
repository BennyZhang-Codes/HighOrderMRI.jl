# Expanded encoding model

This page defines the signal model and discrete conventions implemented by `HighOrderOp`, `HighOrderKernelOp`, and `HighOrderLowRankOp`. All three operators represent the same field-aware encoding model; they differ only in numerical implementation and, for `HighOrderLowRankOp`, in the low-rank approximation applied to the residual encoding matrix.

Notation is defined centrally in [Symbols and notation](/theory/symbols). Phase sign, spatial coordinates, solid-harmonic normalization, vector ordering, and operator normalization are treated as part of the model definition and are held fixed when implementations are compared.

## Signal model

Let $m(\mathbf r)$ denote the complex transverse magnetization at physical location $\mathbf r=(x,y,z)^T$, and let $C_c(\mathbf r)$ denote the receive sensitivity of coil $c$. For dynamic or interleave $d$, the receive signal is represented as

$$
s_{c,d}(t)
=
\int_{\Omega}
m(\mathbf r)C_c(\mathbf r)
\exp\!\left[i2\pi\Phi_d(\mathbf r,t)\right]
\,d\mathbf r
+
\varepsilon_{c,d}(t),
$$

where $\Phi_d(\mathbf r,t)$ denotes the accumulated encoding phase in cycles before multiplication by $2\pi$. This formulation follows the expanded-encoding treatment used for reconstruction in the presence of spatiotemporally varying higher-order fields. [[1]](/references#ref-1 "Wilm BJ, Barmet C, Pavan M, Pruessmann KP. Higher order reconstruction for MRI in the presence of spatiotemporal field perturbations. Magn Reson Med. 2011;65:1690-1701.")

For an instantaneous dynamic field coefficient $a_{\ell,d}(t)$, define the accumulated coefficient

$$
k_{\ell,d}(t)
=
\int_{t_{\mathrm{ref}}}^{t}
a_{\ell,d}(\tau)\,d\tau.
$$

If $a_{\ell,d}$ is expressed in Hz per basis unit, then $k_{\ell,d}$ is expressed in cycles per basis unit. HighOrderMRI therefore expects the time-integrated coefficients $k_{\ell,d}(t)$ in `kspha`.

Separating the spatially uniform term, the first-order Fourier terms, static off-resonance, and the remaining non-Fourier terms gives

$$
\Phi_d(\mathbf r,t)
=
k_{0,d}(t)
+t\,\Delta f_0(\mathbf r)
+\mathbf k_d^{(1)}(t)^T\mathbf r
+\sum_{\ell\in\mathcal R}
k_{\ell,d}(t)b_\ell(\mathbf r).
$$

Here $k_{0,d}(t)$ is the zeroth-order accumulated phase, $\Delta f_0(\mathbf r)$ is the static off-resonance map, $\mathbf k_d^{(1)}(t)$ contains the first-order spatial coefficients, and $b_\ell(\mathbf r)$ is a real solid-harmonic basis function. The residual set $\mathcal R$ depends on whether the reconstruction is two- or three-dimensional, as specified below.

HighOrderMRI uses the positive forward-phase convention

$$
\exp\!\left(+i2\pi\Phi_d(\mathbf r,t)\right).
$$

Accordingly, all terms inside $\Phi_d$ must be expressed in cycles. Field-camera phase supplied in radians must be divided by $2\pi$ before entering this convention.

## Discrete dynamic formulation

Let $t_{jd}$ denote sample $j$ in dynamic $d$, $m_v$ the magnetization at masked voxel $v$, and $C_{vc}$ the corresponding receive sensitivity. The discrete operator implemented in HighOrderMRI is

$$
y_{jdc}
=
\frac{1}{\sqrt{N_v}}
\sum_{v=1}^{N_v}
m_v C_{vc}
\exp\!\left\{
i2\pi\left[
k_{0,jd}
+t_{jd}\Delta f_{0,v}
+\mathbf k_{jd}^{(1)T}\mathbf r_v
+\sum_{\ell\in\mathcal R}
k_{\ell,jd}b_{\ell,v}
\right]
\right\}
+
\varepsilon_{jdc}.
$$

The factor $1/\sqrt{N_v}$ is a symmetric numerical normalization used by the forward and adjoint operators; it is not introduced as a spatial quadrature weight for the continuous signal integral. The adjoint conjugates the complete encoding phase and the receive sensitivity.

In the explicit implementation, the phase contribution is evaluated as

```julia
phase = times .* fieldmap + bf * kspha
```

where the first row of `bf` is identically one. Thus, the spatially uniform dynamic term is included once through the first row of `bf * kspha`.

## Matrix formulation

For dynamic $d$, define the zeroth-order temporal modulation

$$
D_{0,d}(j,j)
=
\exp\!\left(i2\pi k_{0,jd}\right),
$$

the first-order Fourier matrix

$$
F_d(j,v)
=
\exp\!\left(i2\pi\mathbf k_{jd}^{(1)T}\mathbf r_v\right),
$$

and the residual phase matrix

$$
H_d(j,v)
=
\exp\!\left\{
i2\pi\left[
t_{jd}\Delta f_{0,v}
+\sum_{\ell\in\mathcal R}
k_{\ell,jd}b_{\ell,v}
\right]
\right\}.
$$

With $D_{0,d}\in\mathbb C^{N_s\times N_s}$ and $F_d,H_d\in\mathbb C^{N_s\times N_v}$, the single-dynamic spatial encoding matrix is

$$
E_d
=
D_{0,d}\left(F_d\odot H_d\right),
$$

where $\odot$ denotes the Hadamard product. For coil $c$,

$$
\mathbf y_{d,c}
=
\frac{1}{\sqrt{N_v}}
E_d C_c\mathbf m
+
\boldsymbol\varepsilon_{d,c},
$$

with $C_c=\operatorname{diag}(C_{1c},\ldots,C_{N_vc})$.

This factorization provides the common mathematical reference for the explicit and low-rank implementations.

## Explicit implementation

`HighOrderOp` and `HighOrderKernelOp` evaluate the complete phase model directly and therefore do not introduce a low-rank approximation.

`HighOrderOp` is the array-based reference implementation and can execute on CPU or CUDA arrays. Sample-voxel phase blocks are evaluated explicitly, with `nBlock` controlling the temporary-memory/runtime trade-off.

`HighOrderKernelOp` evaluates the same signal model using fused CUDA kernels. Its single- and multi-GPU paths change data placement and execution strategy, not the encoding equation. In the multi-GPU implementation, masked voxels are partitioned across devices; forward partial signals are reduced and adjoint voxel shards are gathered.

Agreement between the two explicit implementations is therefore an implementation-consistency test rather than an independent physical validation of the signal model. The validation hierarchy is described in [Scientific validation strategy](/guide/validation).

## Low-rank factorization

`HighOrderLowRankOp` retains $D_{0,d}$ and the first-order Fourier encoding while approximating only the residual matrix $H_d$. This separation is related to singular-vector representations previously used to accelerate higher-order MRI reconstruction and to subsequent expanded-model reconstructions such as MaxGIRF. [[2]](/references#ref-2 "Wilm BJ, Barmet C, Pruessmann KP. Fast higher-order MR image reconstruction using singular-vector separation. IEEE Trans Med Imaging. 2012;31:1396-1403.") [[3]](/references#ref-3 "Lee NG, Ramasawmy R, Lim Y, Campbell-Washburn AE, Nayak KS. MaxGIRF: Image reconstruction incorporating concomitant field and gradient impulse response function effects. Magn Reson Med. 2022;88:691-710.")

For each dynamic,

$$
H_d
\approx
U_d\widetilde V_d^H,
$$

where the local factors are obtained using a matrix-free randomized SVD. The randomized range-finding step follows standard randomized matrix-decomposition methods. [[4]](/references#ref-4 "Halko N, Martinsson PG, Tropp JA. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Rev. 2011;53:217-288.")

The retained spatial factors are then recompressed incrementally into a basis shared across dynamics,

$$
\widetilde V_d
\approx
S\bar C_d,
$$

which yields

$$
H_d
\approx
\widehat q_dS^H,
\qquad
\widehat q_d
=
U_d\bar C_d^H.
$$

The stored sample-domain coefficient matrix additionally incorporates the zeroth-order temporal phase, the parity-dependent NFFT centre correction, and the $1/\sqrt{N_v}$ normalization. The first-order Fourier phase remains in the NFFT.

The shared basis is constructed incrementally across dynamics. For an earlier dynamic, the final stored coefficient block is generally not equal to the post-hoc projection $S^H\widetilde V_d$, because previously processed dynamics are not reprojected when later basis columns are appended. The complete derivation is given in [Low-rank shared subspace](/theory/low-rank).

## Solid-harmonic basis order

Julia uses one-based rows in `kspha`. The operator-supported terms through third order are:

| Julia row | Order | Spatial basis |
|---:|---:|---|
| 1 | 0 | $1$ |
| 2 | 1 | $x$ |
| 3 | 1 | $y$ |
| 4 | 1 | $z$ |
| 5 | 2 | $xy$ |
| 6 | 2 | $zy$ |
| 7 | 2 | $3z^2-(x^2+y^2+z^2)$ |
| 8 | 2 | $xz$ |
| 9 | 2 | $x^2-y^2$ |
| 10 | 3 | $3yx^2-y^3$ |
| 11 | 3 | $xyz$ |
| 12 | 3 | $[5z^2-(x^2+y^2+z^2)]y$ |
| 13 | 3 | $5z^3-3z(x^2+y^2+z^2)$ |
| 14 | 3 | $[5z^2-(x^2+y^2+z^2)]x$ |
| 15 | 3 | $x^2z-y^2z$ |
| 16 | 3 | $x^3-3xy^2$ |

These are the unnormalized real polynomial basis functions implemented by `basisfunc_spha`; they should not be interpreted as normalized spherical harmonics. Coefficients defined for a different normalization or coordinate system require an explicit conversion. When coordinates are expressed in metres, typical accumulated-coefficient units are cycles, cycles/m, cycles/m², and cycles/m³ for orders zero through three.

`basisfunc_spha` contains additional higher-order polynomial terms, whereas the current high-order encoding constructors accept 9 or 16 terms, corresponding to second- or third-order encoding, respectively.

## 2D and 3D Fourier-residual separation

The Fourier/residual split depends on reconstruction dimensionality and follows the implemented operator:

- **3D:** rows 2-4 ($x$, $y$, $z$) define the 3D NFFT trajectory; the residual set $\mathcal R$ begins at row 5.
- **2D:** rows 2-3 ($x$, $y$) define the 2D NFFT trajectory; row 4 remains in the residual model. With the default centred `Grid(..., nZ=1, ...)`, the single $z$ coordinate is $z=0$, so $k_z(t)z=0$. A nonzero row-4 contribution requires geometry in which the $z$ coordinate represents a nonzero physical slice offset.

The zeroth-order row is included in neither the NFFT trajectory nor $H_d$; it is applied once through $D_{0,d}$ or, in the low-rank implementation, through the equivalent temporal modulation incorporated into `q`.

## NFFT sign and centring

AbstractNFFTs uses a negative Fourier exponent on an integer-centred grid, whereas the explicit HighOrderMRI convention uses the positive physical phase defined above. For active axis $i$, the NFFT node is therefore

$$
\xi_{i,jd}
=
-k_{i,jd}\Delta_i.
$$

The physical voxel spacing $\Delta_i$ is applied independently along each active axis; use of one common spacing is incorrect for an anisotropic grid.

`Grid` places voxel centres at

$$
r_i[n]
=
\left[n-\frac{N_i+1}{2}\right]\Delta_i,
\qquad
n=1,\ldots,N_i.
$$

For odd $N_i$, the physical and integer-centred NFFT grids have the same centre. For even $N_i$, the physical voxel centres are shifted by half a voxel. With $\xi_i=-k_i\Delta_i$, the low-rank implementation incorporates the corresponding temporal correction

$$
c_{\mathrm{ctr},i}(j,d)
=
\exp\!\left[i2\pi\left(-\frac{1}{2}\xi_{i,jd}\right)\right]
=
\exp\!\left(i\pi k_{i,jd}\Delta_i\right)
$$

for each even active axis; no correction is required for odd axes. The total centre correction is the product over active axes. This convention is used when `nfft_center_correction=true`.

## Array layout

| Quantity | Explicit operators | Low-rank operator |
|---|---|---|
| `kspha` | `(nTerm,nSam)` | `(nTerm,nSam,nDyn)` |
| `times` | `(nSam,)` | `(nSam,nDyn)` |
| `fieldmap` | `(nX,nY[,nZ])` | same |
| `csm` | `(nX,nY[,nZ],nCha)` | same |
| `mask` | `(nX,nY[,nZ])` | same |
| logical output | `(nSam,nCha)` | `(nSam,nDyn,nCha)` |

Julia column-major vectorization makes samples the fastest-changing index, followed by dynamics and then channels. The same convention is summarized in the [unified symbol table](/theory/symbols#implementation-array-names).

## Selecting field orders

`recon_terms` is a binary string:

- 9 terms: three digits for zeroth, first, and second order;
- 16 terms: four digits for zeroth through third order.

Disabling zeroth-, second-, or third-order encoding sets the corresponding coefficients to zero. Disabling first-order measured encoding replaces rows 2-4 with `k_nominal`; it does not set the trajectory to zero. `prep_kspha` copies the input before applying this selection, so the caller's `kspha` is not modified.

Use identical `recon_terms`, `k_nominal`, mask, coordinate transformations, normalization, and data ordering when comparing implementations. The fixed comparison convention is recorded in the [reconstruction protocol](/guide/reconstruction-protocol).
