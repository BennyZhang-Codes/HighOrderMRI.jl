# Symbols and notation

This page defines the notation used across the HighOrderMRI theory, reconstruction, validation, and performance documentation. Unless a page explicitly states otherwise, the same symbol has the same meaning everywhere.

## Indices and sizes

| Symbol | Meaning |
| --- | --- |
| $d$ | Dynamic, interleave, shot, or repeated encoding index, depending on the acquisition |
| $j$ | ADC sample index within dynamic $d$ |
| $v$ | Masked reconstruction-voxel index |
| $c$ | Receive-coil index |
| $\ell$ | Solid-harmonic basis-term index |
| $r$ | Shared spatial-basis index |
| $N_s$ | Number of ADC samples per dynamic |
| $N_d$ | Number of dynamics/interleaves represented by the low-rank operator |
| $N_v$ | Number of voxels inside the reconstruction mask |
| $N_c$ | Number of receive channels after any coil compression |
| $L$ | Local per-dynamic low-rank truncation rank (`L_rank`) |
| $R$ | Final shared spatial rank |

## Physical and image-domain quantities

| Symbol | Meaning | Typical units / dimensions |
| --- | --- | --- |
| $\mathbf r=(x,y,z)^T$ | Physical voxel coordinate | m when the grid is specified in metres |
| $\mathbf r_v$ | Coordinate of masked voxel $v$ | m |
| $m(\mathbf r)$, $m_v$ | Complex transverse magnetization / reconstructed image value | arbitrary signal units |
| $C_c(\mathbf r)$, $C_{vc}$ | Receive sensitivity of coil $c$ | complex relative sensitivity |
| $\Delta f_0(\mathbf r)$, $\Delta f_{0,v}$ | Static off-resonance map | Hz when time is in s |
| $b_\ell(\mathbf r)$, $b_{\ell,v}$ | Real solid-harmonic spatial basis function | depends on harmonic order and coordinate units |
| $\Delta_i$ | Physical voxel spacing along axis $i$ | m or another consistently used length unit |

## Dynamic field and phase quantities

| Symbol | Meaning | Typical units |
| --- | --- | --- |
| $a_{\ell,d}(t)$ | Instantaneous coefficient of solid-harmonic term $\ell$ | Hz per basis unit |
| $k_{\ell,d}(t)$ | Time-integrated coefficient $\int a_{\ell,d}(\tau)d\tau$ | cycles per basis unit |
| $k_{0,jd}$ | Zeroth-order spatially uniform accumulated phase | cycles |
| $\mathbf k^{(1)}_{jd}$ | First-order accumulated coefficients used for spatial Fourier encoding | reciprocal physical length |
| $\Phi_d(\mathbf r,t)$ | Total accumulated encoding phase before multiplication by $2\pi$ | cycles |
| $t_{jd}$ | ADC sampling time of sample $j$ in dynamic $d$ | s when $\Delta f_0$ is in Hz |
| $\xi_{i,jd}$ | Dimensionless AbstractNFFTs node for active axis $i$ | dimensionless |

In the implementation, the time-integrated coefficients $k_{\ell,d}(t)$ are stored in `kspha`. If field-camera phase is supplied in radians, it must be converted to cycles before entering this convention.

## Encoding matrices and operators

| Symbol | Meaning | Dimensions |
| --- | --- | --- |
| $D_{0,d}$ | Diagonal zeroth-order temporal modulation | $N_s\times N_s$ |
| $F_d$ | First-order Fourier encoding matrix | $N_s\times N_v$ |
| $H_d$ | Residual phase matrix containing static off-resonance and non-NFFT field terms | $N_s\times N_v$ |
| $E_d$ | Complete single-dynamic spatial encoding matrix, excluding coil multiplication and the symmetric normalization | $N_s\times N_v$ |
| $C_c$ | Diagonal matrix of receive sensitivities for coil $c$ | $N_v\times N_v$ |
| $A_c$ | Complete linear encoding operator for receive coil $c$ | sample space $\leftarrow$ image space |
| $A$ | Multi-coil encoding operator | sample/coil space $\leftarrow$ image space |
| $W$ | Diagonal reconstruction weighting operator, typically containing square-root density weights | sample space $\to$ sample space |

The weighted normal operator is written

$$
A^H W^H W A.
$$

For real non-negative square-root density weights, $W^H W=W^2$, but the Hermitian form is used throughout the documentation because it is unambiguous for a general complex diagonal weighting operator.

## Low-rank quantities

| Symbol | Meaning | Dimensions |
| --- | --- | --- |
| $\Omega_d$ | Random test matrix used by rSVD | $N_v\times(L+p)$ |
| $Y_d=H_d\Omega_d$ | Randomized range sketch | $N_s\times(L+p)$ |
| $Q_d$ | Orthonormal basis for the randomized range | $N_s\times(L+p)$ |
| $B_d=H_d^H Q_d$ | Projected adjoint matrix | $N_v\times(L+p)$ |
| $U_d$ | Retained local sample-domain factor | $N_s\times L$ |
| $\widetilde V_d$ | Retained weighted local spatial factor | $N_v\times L$ |
| $S_d$ | Incremental shared spatial basis after dynamic $d$ | $N_v\times R_d$ |
| $S$ | Final shared spatial basis $S_{N_d}$ | $N_v\times R$ |
| $\bar C_d$ | Stored incremental coefficient block for dynamic $d$, zero-padded to the final rank | $R\times L$ |
| $\widehat q_d=U_d\bar C_d^H$ | Unscaled sample-domain shared-basis coefficient matrix | $N_s\times R$ |
| $q_d$ | Stored sample-domain coefficient matrix after zeroth-order phase, NFFT-centre correction, and $1/\sqrt{N_v}$ normalization | $N_s\times R$ |
| $\eta_d$ | Retained local rank-$L$ energy, $\|\widetilde V_d\|_F^2$ | non-negative scalar |
| $\tau$ | Incremental shared-basis tolerance (`shared_basis_tol`) | dimensionless |

A central implementation detail is that, for an earlier dynamic, the stored final coefficient block is generally **not** the post-hoc projection of $\widetilde V_d$ onto the completed basis:

$$
\bar C_d\neq S^H\widetilde V_d.
$$

The production implementation is streaming and does not retain earlier $\widetilde V_d$ matrices for reprojection after later basis expansion.

## Implementation array names

| Julia name | Mathematical role | Explicit shape | Low-rank shape |
| --- | --- | --- | --- |
| `kspha` | $k_{\ell,d}(t_j)$ | `(nTerm,nSam)` | `(nTerm,nSam,nDyn)` |
| `times` | $t_{jd}$ | `(nSam,)` | `(nSam,nDyn)` |
| `fieldmap` | $\Delta f_0(\mathbf r)$ | `(nX,nY[,nZ])` | same |
| `csm` | $C_c(\mathbf r)$ | `(nX,nY[,nZ],nCha)` | same |
| `mask` | Reconstruction-domain support | `(nX,nY[,nZ])` | same |
| `q` | Concatenated stored $q_d$ | — | `(nSam*nDyn,R)` |
| `basis` | Final shared spatial basis $S$ on masked voxels | — | `(nVox,R)` |

Julia column-major vectorization makes samples the fastest-changing index, followed by dynamics and then receive channels for the dynamic low-rank data layout.
