# Symbols and notation

This page maps the mathematical notation to the arrays used by HighOrderMRI. Use it alongside the [encoding model](/theory/encoding-model) for physical phase conventions and the [low-rank derivation](/theory/low-rank) for the algorithm. Unless a page explicitly states a local convention, the same symbol has the same meaning throughout the documentation.

The readouts indexed by $i$ encode one unknown image. The shared rSVD notation below distinguishes the completed incremental representation (superscript $(0)$ and rank $R_0$) from the final representation after optional recompression (rank $R$).

## Indices and sizes

| Symbol | Meaning |
| --- | --- |
| $i$ | Readout (encoding profile): dynamic, interleave, shot, or repeated encoding of the same unknown image |
| $j$ | ADC sample index within dynamic $i$ |
| $v$ | Masked reconstruction-voxel index |
| $c$ | Receive-coil index |
| $\ell$ | Solid-harmonic basis-term index |
| $r$ | Shared-mode index; $q_r$ and $s_r$ are columns of the stacked temporal matrix and spatial basis |
| $N_t$ | Number of ADC samples per readout (`nSam`) |
| $N_{\mathrm{ro}}$ | Number of readouts (`nDyn`) |
| $N_v$ | Number of voxels inside the reconstruction mask, `count(mask)` |
| $a$ | Spatial-axis index, x/y/z; distinct from readout $i$ |
| $N_a$ | Number of grid points along spatial axis $a$ |
| $N_{\mathrm{grid}}$ | Full image-grid size, `prod(grid.matrixSize)`; includes masked-out voxels |
| $N_c$ | Number of receive channels after any coil compression |
| $L$ | Retained local per-profile rank (`L_rank`) |
| $p$ | Additional sketch directions (`rsvd_oversample`); sketch width is $L+p$ |
| $R_0$ | Completed incremental shared rank before optional global recompression |
| $R$ | Final shared spatial rank; $R=R_0$ when global recompression is disabled |

## Physical and image-domain quantities

| Symbol | Meaning | Typical units / dimensions |
| --- | --- | --- |
| $\mathbf r=(x,y,z)^T$ | Physical voxel coordinate | m when the grid is specified in metres |
| $\mathbf r_v$ | Coordinate of masked voxel $v$ | m |
| $m(\mathbf r)$, $m_v$ | Complex transverse magnetization / masked image value; shared by all profiles | arbitrary signal units |
| $C_c(\mathbf r)$, $C_{vc}$ | Receive sensitivity of coil $c$ | complex relative sensitivity |
| $\Delta f_0(\mathbf r)$, $\Delta f_{0,v}$ | Static off-resonance map | Hz when time is in s |
| $b_\ell(\mathbf r)$, $b_{\ell,v}$ | Real solid-harmonic spatial basis function | depends on harmonic order and coordinate units |
| $\Delta_a$ | Physical voxel spacing along axis $a$ | m or another consistently used length unit |

## Dynamic field and phase quantities

| Symbol | Meaning | Typical units |
| --- | --- | --- |
| $a_{\ell,i}(t)$ | Instantaneous coefficient of solid-harmonic term $\ell$ | Hz per basis unit |
| $k_{\ell,i}(t)$ | Time-integrated coefficient $\int a_{\ell,i}(\tau)\mathrm{d}\tau$ | cycles per basis unit |
| $k_{0,ji}$ | Zeroth-order spatially uniform accumulated phase | cycles |
| $\mathbf k_{ji}$ | First-order accumulated coefficients along x/y/z; Fourier encoding uses the active subset | reciprocal physical length |
| $\mathcal I$ | Active Fourier axes: x/y for `nZ == 1`, x/y/z otherwise | axis set |
| $\mathcal H$ | Modeled higher-order solid-harmonic rows, 5 through `nTerm` | row set |
| $\mathcal R$ | Residual rows: $\mathcal H$ in 3D; $\{4\}\cup\mathcal H$ in 2D | row set |
| $\phi_{\mathrm{con},i}$ | Additional accumulated concomitant phase | cycles |
| $\Phi_i(\mathbf r,t)$ | Total accumulated encoding phase before multiplication by $2\pi$ | cycles |
| $t_{ji}$ | ADC sampling time of sample $j$ in dynamic $i$ | s when $\Delta f_0$ is in Hz |
| $\xi_{a,ji}$ | NFFT node $-k_{a,ji}\Delta_a$ for active axis $a$ | dimensionless |
| $d_{0,ji}$ | Zeroth-order modulation $\exp(\mathrm{i}2\pi k_{0,ji})$ | complex unit modulus |
| $c_{\mathrm{ctr},ji}$ | Parity-dependent NFFT centre correction, or one when disabled | complex unit modulus |

In the implementation, the time-integrated coefficients $k_{\ell,i}(t)$ are stored in `kspha`. Convert field-camera phase from radians to cycles before supplying it; the encoding constructors do not integrate instantaneous fields or perform this conversion. The physical first-order dot product uses x/y/z; the NFFT uses only axes in $\mathcal I$. In 2D, the z term stays in $\mathcal R$ and is counted once. See the [coordinate scope](/theory/encoding-model#coordinate-scope) for the Cartesian-grid contract.

## Encoding matrices and operators

| Symbol | Meaning | Dimensions |
| --- | --- | --- |
| $D_{0,i}$ | Diagonal zeroth-order temporal modulation | $N_t\times N_t$ |
| $F_i$ | First-order Fourier encoding matrix | $N_t\times N_v$ |
| $\mathcal F$ | First-order Fourier/NFFT encoding stacked over readouts, with full-grid embedding of masked inputs | sample space $\leftarrow$ image grid |
| $H_i$ | Residual phase matrix containing static off-resonance and non-NFFT field terms | $N_t\times N_v$ |
| $E_i$ | Complete single-dynamic spatial encoding matrix, excluding coil multiplication and the symmetric normalization | $N_t\times N_v$ |
| $C_c$ | Diagonal matrix of receive sensitivities for coil $c$ | $N_v\times N_v$ |
| $A_c$ | Encoding operator for coil $c$, including mask support and normalization | $(N_tN_{\mathrm{ro}})\times N_{\mathrm{grid}}$ |
| $A$ | Multi-coil encoding operator, with profile blocks stacked within each coil | $(N_tN_{\mathrm{ro}}N_c)\times N_{\mathrm{grid}}$ |
| $W$ | Diagonal operator of raw reconstruction weights, repeated over coils by `recon_HOOp` | $(N_tN_{\mathrm{ro}}N_c)\times(N_tN_{\mathrm{ro}}N_c)$ |
| $D=W^H W$ | Nonnegative diagonal normal weights, with entries $|w|^2$ for raw weight $w$ | same as $W$ |

The weighted normal operator is written

$$
A^H W^H W A.
$$

For real non-negative square-root density weights, $W^H W=W^2$. For complex raw weights use $W^H W$, whose diagonal is `abs2.(weight)`.

`recon_HOOp` applies the raw weights to both the data and $A$; pass the raw weights once. The lower-level `normalOperator(op, weights2)` convention instead expects the already squared real nonnegative diagonal entries of $D$. See [reconstruction weights](/reference/recon-hoop) and the [weighted-normal API](/reference/highorderkernelop#weighted-normal) for the relevant call conventions. The data normal excludes any regularization term chosen by the solver.

## Low-rank quantities

The local rSVD sketch width is $L_{\mathrm{sk}}=L+p\leq\min(N_t,N_v)$; $\ell$ is reserved for the solid-harmonic row index. $B_i$ below is $H_i^H Q_i$, the projected **adjoint** matrix.

| Symbol | Meaning | Dimensions |
| --- | --- | --- |
| $\Omega_i$ | Random test matrix used by rSVD | $N_v\times L_{\mathrm{sk}}$ |
| $Y_i=H_i\Omega_i$ | Randomized range sketch | $N_t\times L_{\mathrm{sk}}$ |
| $Q_i$ | Orthonormal basis for the randomized range | $N_t\times L_{\mathrm{sk}}$ |
| $Q_i^{\mathrm{add}}$ | Appended residual modes, equal to $S_{i,\mathrm{new}}$; distinct from the range basis $Q_i$ | $N_v\times r_i$ |
| $B_i=H_i^H Q_i$ | Projected adjoint matrix | $N_v\times L_{\mathrm{sk}}$ |
| $U_{i,L}$ | Retained local sample-domain factor | $N_t\times L$ |
| $\widetilde V_i=V_{i,L}\Sigma_{i,L}$ | Retained spatial singular vectors with singular values absorbed | $N_v\times L$ |
| $Z_i$ | Spatial-factor residual after projection onto the existing shared basis | $N_v\times L$ |
| $Z_i^{\mathrm{svd}}$ | Right rotation from the local projected SVD/Gram eigensystem | $L_{\mathrm{sk}}\times L_{\mathrm{sk}}$ |
| $S_i$ | Incremental shared spatial basis after profile $i$; $R_i$ here is its column count | $N_v\times R_i$ |
| $S^{(0)}=S_{N_{\mathrm{ro}}}$ | Completed incremental shared basis | $N_v\times R_0$ |
| $\bar C_i$ | Saved incremental coefficient block, zero-padded to rank $R_0$ | $R_0\times L$ |
| $\widehat q_i^{(0)}=U_{i,L}\bar C_i^H$ | Unscaled temporal factor before optional global recompression | $N_t\times R_0$ |
| $S$ | Final shared spatial basis, orthonormal up to numerical roundoff | $N_v\times R$ |
| $\widehat q_i$ | Final unscaled temporal factor, after optional global recompression | $N_t\times R$ |
| $q_i$ | Stored sample-domain coefficient matrix after zeroth-order phase, NFFT-centre correction, and $1/\sqrt{N_v}$ normalization | $N_t\times R$ |
| $\eta_i$ | Retained local rank-$L$ energy, $\|\widetilde V_i\|_F^2$ | non-negative scalar |
| $\tau$ | Incremental shared-basis tolerance (`shared_basis_tol`) | dimensionless |
| $\epsilon_g$ | Additional global recompression tolerance (`global_basis_tol`, when supplied) | dimensionless |

The shared projection residual is $Z_i=\widetilde V_i-S_{i-1}C_i^{\mathrm{old}}$, with shape $N_v\times L$. In contrast, $R_i$ is the scalar column count of $S_i$. The local rSVD rotation $Z_i^{\mathrm{svd}}$ is a separate matrix; the superscript distinguishes it from this residual.

For incremental rSVD construction, a central implementation detail is that, for an earlier profile, the saved coefficient block is generally **not** the post-hoc projection of $\widetilde V_i$ onto the completed incremental basis:

$$
\bar C_i\neq (S^{(0)})^H\widetilde V_i.
$$

The incremental shared builder does not retain earlier $\widetilde V_i$ matrices for reprojection when later profiles expand the basis. Its representation is

$$
H_i\approx\widehat q_i^{(0)}(S^{(0)})^H.
$$

Optional final global recompression maps the completed factors to $\widehat q_i,S$ with $R\leq R_0$; without it these factors are unchanged. The final residual model and stored temporal rows are

$$
H_i\approx\widehat q_iS^H,
\qquad
q_i(j,:)=\frac{d_{0,ji}\,c_{\mathrm{ctr},ji}}{\sqrt{N_v}}
\widehat q_i(j,:).
$$

$\tau$ refers to compression of retained local factors. $\epsilon_g$ refers to additional error relative to the already shared representation. Neither is an image-error tolerance. See the [algorithm and error derivation](/theory/low-rank#two-stage-approximation-error).

## Implementation array names

The `q` and `basis` rows in this table describe `basis_layout=:shared` after optional final recompression. Dimensions describe array layout, independently of CPU/GPU placement.

| Julia name | Mathematical role | Explicit shape | Low-rank shape |
| --- | --- | --- | --- |
| `kspha` | $k_{\ell,i}(t_j)$ | `(nTerm,nSam)` | `(nTerm,nSam,nDyn)` |
| `times` | $t_{ji}$ | `(nSam,)` | `(nSam,nDyn)` |
| `fieldmap` | $\Delta f_0(\mathbf r)$ | `(nX,nY[,nZ])` | same |
| `csm` | $C_c(\mathbf r)$ | `(nX,nY[,nZ],nCha)` | same |
| `mask` | Reconstruction mask | `(nX,nY[,nZ])` | same |
| `concomitant` | Four integrated concomitant coefficients | `(4,nSam)` | `(4,nSam,nDyn)` |
| image vector | Full image-grid input | `(prod(grid.matrixSize),)` | same image for all profiles |
| logical output | Samples before `vec` | `(nSam,nCha)` | `(nSam,nDyn,nCha)` |
| `q` | Concatenated stored $q_i$ | — | `(nSam*nDyn,R)` |
| `basis` | Final shared spatial basis $S$ on masked voxels | — | `(nVox,R)` |

The single-profile LowRank overload also accepts a coefficient matrix `(nTerm,nSam)` and a time vector `(nSam,)`. For one profile, `concomitant` may be `(4,nSam)`.

With `basis_layout=:per_readout`, `q` instead has shape `(nSam*nDyn,L)` and the logical spatial-factor bank has shape `(nVox,L*nDyn)`. Regenerated storage need not materialize this whole bank; see [per-readout storage](/reference/highorderlowrankop#per-readout-parameters).

Julia column-major vectorization makes samples the fastest-changing index, followed by profiles and then receive channels. For a logical output $y_{jic}$, its one-based vector index is

$$
j+(i-1)N_t+(c-1)N_tN_{\mathrm{ro}}.
$$

Use `vec(image)` for the full-grid input and `vec(data)` for output shaped `(nSam,nDyn,nCha)`; masked-out image values do not contribute to the signal.
