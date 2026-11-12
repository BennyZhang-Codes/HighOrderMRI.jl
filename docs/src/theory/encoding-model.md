# Expanded encoding model

This page explains how physical field information becomes the forward encoding operator used in reconstruction. `HighOrderOp` and `HighOrderKernelOp` evaluate the phase explicitly; `HighOrderLowRankOp` separates Fourier encoding from a low-rank residual. For the ordinary forward model (`kspha_dt=nothing`), they target the same equation under the coordinate and array conventions below. LowRank also introduces low-rank truncation and NFFT interpolation error.

Read the [signal model](#signal-model) and [array layout](#array-layout) first to prepare inputs, then the [Fourier/residual split](#2d-and-3d-fourier-residual-separation) to understand the algorithm. A [complete CPU example](#minimal-cpu-example) connects the equation to the constructor. Detailed low-rank construction belongs in [Low-rank shared subspace](/theory/low-rank).

Notation is defined centrally in [Symbols and notation](/theory/symbols). Phase sign, spatial coordinates, solid-harmonic normalization, vector ordering, and operator normalization are treated as part of the model definition and are held fixed when implementations are compared.

## Signal model

Let $m(\mathbf r)$ denote the complex transverse magnetization at physical location $\mathbf r=(x,y,z)^T$, and let $C_c(\mathbf r)$ denote the receive sensitivity of coil $c$. For encoding profile or interleave $i$, the receive signal is represented as

$$
y_{i,c}(t)
=
\int_{\Omega}
m(\mathbf r)C_c(\mathbf r)
\exp\!\left[\mathrm{i}2\pi\Phi_i(\mathbf r,t)\right]
\,\mathrm{d}\mathbf r
+
\epsilon_{i,c}(t),
$$

Here all profiles encode the same unknown image $m(\mathbf r)$; $i$ does not introduce a separate reconstructed image frame. $\Phi_i(\mathbf r,t)$ denotes the accumulated encoding phase in cycles before multiplication by $2\pi$. This formulation follows the expanded-encoding treatment used for reconstruction in the presence of spatiotemporally varying higher-order fields. [[1]](/references#ref-1 "Wilm BJ, Barmet C, Pavan M, Pruessmann KP. Higher order reconstruction for MRI in the presence of spatiotemporal field perturbations. Magn Reson Med. 2011;65:1690-1701.")

For an instantaneous dynamic field coefficient $a_{\ell,i}(t)$, define the accumulated coefficient

$$
k_{\ell,i}(t)
=
\int_{t_{\mathrm{ref}}}^{t}
a_{\ell,i}(\tau)\,\mathrm{d}\tau.
$$

If $a_{\ell,i}$ is expressed in Hz per basis unit, then $k_{\ell,i}$ is expressed in cycles per basis unit. HighOrderMRI therefore expects the time-integrated coefficients $k_{\ell,i}(t)$ in `kspha`; the encoding constructors do not integrate instantaneous fields or convert radians automatically. `times` contains the time used in the static off-resonance product (seconds for a field map in Hz). Choose consistent time references for that phase and the supplied accumulated coefficients.

Readout $i$ contains $N_t$ samples; there are $N_{\mathrm{ro}}$ readouts (`nSam` and `nDyn` in Julia). The physical first-order vector $\mathbf k_i(t)$ contains x/y/z coefficients, and $\mathcal H=\{5,\ldots,nTerm\}$ contains the modeled higher-order rows. For NFFT factorization, active axes $\mathcal I$ are x/y when `grid.nZ == 1` and x/y/z otherwise; the residual row set is $\mathcal R=\{4\}\cup\mathcal H$ in 2D and $\mathcal R=\mathcal H$ in 3D. The z term is counted once.

Separating the spatially uniform term, full first-order phase, static off-resonance, and higher-order terms gives

$$
\Phi_i(\mathbf r,t)
=
k_{0,i}(t)
+t\,\Delta f_0(\mathbf r)
+\mathbf k_i(t)^T\mathbf r
+\sum_{\ell\in\mathcal H}
k_{\ell,i}(t)b_\ell(\mathbf r)
+\phi_{\mathrm{con},i}(\mathbf r,t).
$$

Here $k_{0,i}(t)$ is the zeroth-order accumulated phase, $\Delta f_0(\mathbf r)$ is the static off-resonance map, $\mathbf k_i(t)$ contains the first-order spatial coefficients, and $b_\ell(\mathbf r)$ is a real solid-harmonic basis function. The residual set $\mathcal R$ depends on whether the reconstruction is two- or three-dimensional, as specified below.

HighOrderMRI uses the positive forward-phase convention

$$
\exp\!\left(+\mathrm{i}2\pi\Phi_i(\mathbf r,t)\right).
$$

Accordingly, all terms inside $\Phi_i$ must be expressed in cycles. Field-camera phase supplied in radians must be divided by $2\pi$ before entering this convention.

## Optional concomitant phase

The `concomitant` keyword supplies four additional time-integrated coefficients,

$$
\phi_{\mathrm{con},i}(\mathbf r,t)=c_{1,i}(t)z^2+c_{2,i}(t)(x^2+y^2)
+c_{3,i}(t)xz+c_{4,i}(t)yz.
$$

Supply coefficients in cycles/m², with coordinates in metres in the same frame and physical origin as the coefficients. Use `(4,nSam)` for `HighOrderOp`, `HighOrderKernelOp`, or a single-dynamic LowRank problem; use `(4,nSam,nDyn)` for dynamic LowRank. Their row order is exactly `z²`, `x²+y²`, `xz`, `yz`.

With `concomitant=nothing`, $\phi_{\mathrm{con},i}=0$. Otherwise add this phase to $\Phi_i$ and to the residual exponent in $H_i$ below. The coefficients are already integrated at the supplied sample times: they are neither inferred from gradients nor multiplied by `times` again. Divide coefficients expressed in rad/m² by $2\pi$ before supplying them.

The original `kspha` retains its 9 or 16 spherical-harmonic rows. Concomitant coefficients are appended after `recon_terms` selection, so this string does not disable them. This augments the physical model, independently of choosing an explicit or low-rank representation. The explicit `kspha_dt` delay-estimation path ignores `concomitant` and keeps its existing derivative model.

## Discrete dynamic formulation

Let $t_{ji}$ denote sample $j$ in dynamic $i$, $m_v$ the magnetization at masked voxel $v$, and $C_{vc}$ the corresponding receive sensitivity. The discrete operator implemented in HighOrderMRI is

$$
y_{jic}
=
\frac{1}{\sqrt{N_v}}
\sum_{v=1}^{N_v}
m_v C_{vc}
\exp\!\left\{
\mathrm{i}2\pi\left[
k_{0,ji}
+t_{ji}\Delta f_{0,v}
+\mathbf k_{ji}^{T}\mathbf r_v
+\sum_{\ell\in\mathcal H}
k_{\ell,ji}b_{\ell,v}
+\phi_{\mathrm{con},i}(\mathbf r_v,t_{ji})
\right]
\right\}
+
\varepsilon_{jic}.
$$

The factor $1/\sqrt{N_v}$ is a symmetric numerical normalization used by the forward and adjoint operators; it is not introduced as a spatial quadrature weight for the continuous signal integral. $N_v=\operatorname{count}(\mathtt{mask})$, while the public image vector has length `prod(grid.matrixSize)`. The forward ignores values outside the mask; the adjoint returns zeros there.

If $\varphi_{jiv}$ is the complete bracketed phase above, the adjoint is

$$
(A^H\mathbf y)_v
=
\frac{1}{\sqrt{N_v}}
\sum_{i=1}^{N_{\mathrm{ro}}}\sum_{c=1}^{N_c}\sum_{j=1}^{N_t}
\overline{C_{vc}}\,
\exp(-\mathrm{i}2\pi\varphi_{jiv})\,y_{jic}.
$$

It conjugates both the encoding phase and receive sensitivity and sums all profiles into the same image. Applying this adjoint alone is a backprojection; iterative reconstruction solves the inverse problem.

For one dynamic and the masked voxels, the explicit phase can be written as

```julia
fieldmap_masked = vec(fieldmap)[vec(mask)]
phase = fieldmap_masked .* transpose(times) .+ bf * kspha
```

where `times` is a vector for this single profile, `bf` has shape `(nVox,nTerm)`, and its first column is identically one. Thus, the spatially uniform dynamic term is included once through the first coefficient row of `kspha`. Optional concomitant basis columns and coefficient rows are appended before evaluating the same product.

## Matrix formulation

For dynamic $i$, define the zeroth-order temporal modulation

$$
D_{0,i}(j,j)
=
\exp\!\left(\mathrm{i}2\pi k_{0,ji}\right),
$$

the first-order Fourier matrix

$$
F_i(j,v)
=
\exp\!\left(\mathrm{i}2\pi\sum_{a\in\mathcal I}k_{a,ji}r_{a,v}\right),
$$

and the residual phase matrix

$$
H_i(j,v)
=
\exp\!\left\{
\mathrm{i}2\pi\left[
t_{ji}\Delta f_{0,v}
+\sum_{\ell\in\mathcal R}
k_{\ell,ji}b_{\ell,v}
+\phi_{\mathrm{con},i}(\mathbf r_v,t_{ji})
\right]
\right\}.
$$

With $D_{0,i}\in\mathbb C^{N_t\times N_t}$ and $F_i,H_i\in\mathbb C^{N_t\times N_v}$, the single-dynamic spatial encoding matrix is

$$
E_i
=
D_{0,i}\left(F_i\odot H_i\right),
$$

where $\odot$ denotes the Hadamard product. For coil $c$,

$$
\mathbf y_{i,c}
=
\frac{1}{\sqrt{N_v}}
E_i C_c\mathbf m
+
\boldsymbol\varepsilon_{i,c},
$$

with $C_c=\operatorname{diag}(C_{1c},\ldots,C_{N_vc})$.

This factorization provides the common mathematical reference for the explicit and low-rank implementations.

## Explicit implementation

`HighOrderOp` and `HighOrderKernelOp` evaluate the complete phase model directly and therefore do not introduce a low-rank approximation.

`HighOrderOp` is the array-based reference implementation and can execute on CPU or CUDA arrays. Sample-voxel phase blocks are evaluated explicitly, with `nBlock` controlling the temporary-memory/runtime trade-off.

`HighOrderKernelOp` evaluates the same signal model with legacy/direct kernels or resident cuBLAS backends. Legacy multi-GPU execution partitions masked voxels; resident backends partition samples. Blocked and cached representations retain the complete phase model, including supplied concomitant terms. Data placement and finite-precision reduction order change; the encoding equation does not.

For a single profile, the operator represents $\mathbf y=A\mathbf m$. The reconstruction wrapper applies data weights and a CG-based solver; the encoding operator itself does not reconstruct an image. See [Reconstruction workflow](/guide/reconstruction) for solver use. Explicit implementations avoid low-rank truncation, while finite-precision phase evaluation and reduction remain numerical error sources.

## Low-rank factorization

`HighOrderLowRankOp` retains $D_{0,i}$ and the first-order Fourier encoding while approximating only the residual matrix $H_i$. This separation is related to singular-vector representations previously used to accelerate higher-order MRI reconstruction and to subsequent expanded-model reconstructions such as MaxGIRF. [[2]](/references#ref-2 "Wilm BJ, Barmet C, Pruessmann KP. Fast higher-order MR image reconstruction using singular-vector separation. IEEE Trans Med Imaging. 2012;31:1396-1403.") [[3]](/references#ref-3 "Lee NG, Ramasawmy R, Lim Y, Campbell-Washburn AE, Nayak KS. MaxGIRF: Image reconstruction incorporating concomitant field and gradient impulse response function effects. Magn Reson Med. 2022;88:691-710.")

For each dynamic,

$$
H_i
\approx
U_{i,L}\widetilde V_i^H,
$$

where the local factors are obtained using a matrix-free randomized SVD. The randomized range-finding step follows standard randomized matrix-decomposition methods. [[4]](/references#ref-4 "Halko N, Martinsson PG, Tropp JA. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Rev. 2011;53:217-288.")

With the default `basis_layout=:shared` and `shared_basis_method=:rsvd`, the retained spatial factors are compressed incrementally into a shared basis. Denote the completed incremental basis by $S^{(0)}\in\mathbb C^{N_v\times R_0}$ and the saved coefficient blocks by $\bar C_i\in\mathbb C^{R_0\times L}$:

$$
\widetilde V_i
\approx
S^{(0)}\bar C_i,
$$

which yields

$$
H_i
\approx
\widehat q_i^{(0)}(S^{(0)})^H,
\qquad
\widehat q_i^{(0)}
=
U_{i,L}\bar C_i^H.
$$

By default `global_basis_tol=nothing`, so the final $S,\widehat q_i$ are these incremental factors. A supplied nonnegative tolerance enables [optional final recompression](/theory/low-rank#optional-final-global-recompression) of the completed representation; this can reduce its rank and adds a separate approximation. It occurs before temporal phase folding.

The stored sample-domain coefficient matrix is

$$
q_i(j,r)=\frac{1}{\sqrt{N_v}}
\exp(\mathrm{i}2\pi k_{0,ji})\,c_{\mathrm{ctr}}(j,i)\,\widehat q_i(j,r).
$$

Here $c_{\mathrm{ctr}}$ is the parity-dependent correction defined [below](#nfft-sign-and-centring); it is one when `nfft_center_correction=false`. The first-order Fourier phase remains in the NFFT. Forward multiplication uses $q_i$ and the conjugated spatial basis, while the adjoint uses their conjugates in reverse order.

The shared basis is constructed incrementally across profiles. For an earlier profile, $\bar C_i$ generally differs from $(S^{(0)})^H\widetilde V_i$: earlier coefficient blocks are zero-padded when the basis grows, rather than reprojected. This statement refers to the incremental representation before optional final recompression.

`basis_layout=:per_readout` instead retains each profile's local factors. It avoids incremental sharing and requires `global_basis_tol=nothing`; it is a storage/execution choice, distinct from enabling physical field terms. The complete derivation and parameter meanings are given in [Low-rank shared subspace](/theory/low-rank) and the [LowRank API](/reference/highorderlowrankop).

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

- **3D (`nZ > 1`):** rows 2-4 ($x$, $y$, $z$) define the 3D NFFT trajectory; $\mathcal R=\{5,\ldots,nTerm\}$.
- **2D (`nZ == 1`):** rows 2-3 ($x$, $y$) define the 2D NFFT trajectory; $\mathcal R=\{4,\ldots,nTerm\}$, so row 4 remains in the residual model. With the default centred `Grid(..., nZ=1, ...)`, the single $z$ coordinate is $z=0$, so $k_z(t)z=0$. A nonzero row-4 contribution requires a supplied nonzero `grid.z` slice coordinate and coefficients in the same frame. This residual contribution does not add a third NFFT axis.

The zeroth-order row is included in neither the NFFT trajectory nor $H_i$; it is applied once through $D_{0,i}$ or, in the low-rank implementation, through the equivalent temporal modulation incorporated into `q`.

## NFFT sign and centring

AbstractNFFTs uses a negative Fourier exponent on an integer-centred grid, whereas the explicit HighOrderMRI convention uses the positive physical phase defined above. For active axis $a$, the NFFT node is therefore

$$
\xi_{a,ji}
=
-k_{a,ji}\Delta_a.
$$

The physical voxel spacing $\Delta_a$ is applied independently along each active axis; use of one common spacing is incorrect for an anisotropic grid.

For the default untransformed Cartesian grid, `Grid` places voxel centres at

$$
r_a[n]
=
\left[n-\frac{N_a+1}{2}\right]\Delta_a,
\qquad
n=1,\ldots,N_a.
$$

For odd $N_a$, the physical and integer-centred NFFT grids have the same centre. For even $N_a$, the physical voxel centres are shifted by half a voxel. With $\xi_a=-k_a\Delta_a$, the low-rank implementation incorporates the corresponding temporal correction

$$
c_{\mathrm{ctr},a}(j,i)
=
\exp\!\left[\mathrm{i}2\pi\left(-\frac{1}{2}\xi_{a,ji}\right)\right]
=
\exp\!\left(\mathrm{i}\pi k_{a,ji}\Delta_a\right)
$$

for each even active axis; no correction is required for odd axes. The total centre correction is the product over active axes. This convention is enabled by default (`nfft_center_correction=true`). Turning it off removes this parity correction; it does not select a different forward phase sign or disable zeroth-order phase.

### Coordinate scope

The explicit operators evaluate the supplied `grid.x`, `grid.y`, and `grid.z` coordinates directly. LowRank evaluates its residual basis at those coordinates, but constructs its Fourier nodes from `kspha`, the axis spacings, and `MatrixSize`. It does not infer an active-axis permutation, reversal, translation, or rotation from edited coordinate vectors.

For the Fourier/residual equivalence above, use the default centred Cartesian coordinates on the active NFFT axes. A `Grid` coordinate change such as `exchange_xy`, `reverse_x`, or an added active-axis offset is not automatically carried through the LowRank Fourier encoding. Matching the intended transformed geometry requires an explicitly consistent coefficient/image/phase conversion; changing the coordinate vectors alone is insufficient. In a 2D problem, a nonzero `grid.z` remains in the residual model as described above.

## Array layout

| Quantity | Explicit operators | Low-rank operator |
|---|---|---|
| `kspha` | `(nTerm,nSam)` | `(nTerm,nSam,nDyn)` |
| `times` | `(nSam,)` | `(nSam,nDyn)` |
| image vector | `(prod(grid.matrixSize),)` | same unknown image for all profiles |
| `fieldmap` | `(nX,nY[,nZ])` | same |
| `csm` | `(nX,nY[,nZ],nCha)` | same |
| `mask` | `(nX,nY[,nZ])` | same |
| logical output | `(nSam,nCha)` | `(nSam,nDyn,nCha)` |

The single-profile LowRank overload also accepts `(nTerm,nSam)` and `(nSam,)`, adding a length-one profile dimension internally. This does not change the coil ordering.

Julia column-major vectorization makes samples the fastest-changing index, followed by profiles and then channels. Thus the one-based vector index for $y_{jic}$ is

$$
j+(i-1)N_t+(c-1)N_tN_{\mathrm{ro}}.
$$

Vectorize the logical output with `vec`, and pass `vec(image)` to an encoding operator. The same convention is summarized in the [unified symbol table](/theory/symbols#implementation-array-names).

## Selecting field orders

`recon_terms` is a binary string:

- 9 terms: three digits for zeroth, first, and second order;
- 16 terms: four digits for zeroth through third order.

With `recon_terms=nothing`, all supported orders are selected (`"111"` or `"1111"`). A zero in the zeroth-, second-, or third-order position zeros the corresponding coefficient rows. A zero in the first-order position replaces all of rows 2-4 with `k_nominal`; it does not set the trajectory to zero. The default `k_nominal` is the supplied `kspha[2:4,...]`, so a different nominal trajectory must be supplied for that replacement to change those rows.

This flag does not disable the `fieldmap * times` contribution, and optional concomitant terms are appended after order selection. `prep_kspha` copies the input before applying this selection, so the caller's `kspha` is not modified.

## Minimal CPU example

This example evaluates the forward equation and its adjoint for one profile. All lengths are in metres, `times` is in seconds, and the coefficient rows use cycles per corresponding basis unit.

```julia
using HighOrderMRI, LinearAlgebra

T = Float64
nX, nY, nSam = 6, 5, 8
grid = Grid(nX, nY, 1, T(0.002), T(0.003), T(0.004))
times = collect(range(zero(T), T(0.002); length=nSam))
kspha = zeros(T, 9, nSam)
kspha[1, :] .= T(0.05)                         # cycles
kspha[2, :] .= range(T(-35), T(35); length=nSam) # cycles/m
kspha[3, :] .= T(12)                           # cycles/m
kspha[5, :] .= range(T(-40), T(40); length=nSam) # cycles/m²
fieldmap = fill(T(15), nX, nY)                  # Hz
csm = ones(Complex{T}, nX, nY, 1)
mask = trues(nX, nY)
mask[1, 1] = false

op = HighOrderOp(grid, kspha, times;
    fieldmap, csm, mask, arrayType=Array, nBlock=2)
image = ones(Complex{T}, nX, nY)
y = op * vec(image)
backprojection = reshape(adjoint(op) * y, nX, nY)
@assert length(y) == nSam
@assert iszero(backprojection[1, 1])
```

The explicit CPU operator makes every phase term visible without choosing a low-rank truncation. For LowRank setup and multi-profile or multi-GPU configuration, continue with the [LowRank API](/reference/highorderlowrankop) and [multi-GPU guide](/guide/multi-gpu). Use [Reconstruction workflow](/guide/reconstruction) to solve for an image from measured data rather than applying only the adjoint.
