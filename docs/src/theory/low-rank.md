# Low-rank shared subspace

By default, `HighOrderLowRankOp` reduces the cost of repeated applications of the expanded encoding model using two successive approximations: a matrix-free randomized SVD (rSVD) of each dynamic-specific residual encoding matrix, followed by incremental recompression of the retained spatial factors into a basis shared across dynamics. The image, coil sensitivities, and first-order Fourier trajectory are not themselves low-rank approximated.

An alternative, experimental `shared_basis_method=:joint` construction
builds the spatial basis directly from representative phase snapshots and
fits the temporal factors by sampled least squares. Both methods use the
same final encoding and NFFT. See [joint construction](#direct-joint-shared-basis)
and the [API parameters](/reference/highorderlowrankop#shared-basis-parameters).

Symbols follow [Symbols and notation](/theory/symbols).

## Method overview

The following diagram shows the default rSVD construction.

```mermaid
flowchart TD
    A["Residual encoding H_d"] --> B["1. Matrix-free rSVD"]
    B --> C["Temporal factor U_d"]
    B --> D["Spatial factor Ṽ_d"]
    D --> E["2. Incremental shared-basis update"]
    E --> F["Shared basis S_d + coefficients C̄_d"]
    C --> G["3. Sample-domain factor q̂_d"]
    F --> G
    G --> H["Residual model H_d ≈ q̂_d Sᴴ"]
    F --> H
```

The three stages are therefore local factorization, incremental spatial recompression, and assembly of the sample-domain coefficients used by the final operator. The shared basis is constructed sequentially. After a dynamic has been processed, its retained spatial factor is not stored for later reprojection. If subsequent dynamics append columns to the shared basis, the coefficient blocks of earlier dynamics are extended with zeros.

## Relation to previous methods

Separable temporal-spatial representations of higher-order MRI encoding have previously been obtained using singular-vector separation, and related decompositions are used in expanded-model reconstruction such as MaxGIRF. [[2]](/references#ref-2 "Wilm BJ, Barmet C, Pruessmann KP. Fast higher-order MR image reconstruction using singular-vector separation. IEEE Trans Med Imaging. 2012;31:1396-1403.") [[3]](/references#ref-3 "Lee NG, Ramasawmy R, Lim Y, Campbell-Washburn AE, Nayak KS. MaxGIRF: Image reconstruction incorporating concomitant field and gradient impulse response function effects. Magn Reson Med. 2022;88:691-710.") The randomized range-finding step used here follows standard rSVD methods. [[4]](/references#ref-4 "Halko N, Martinsson PG, Tropp JA. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Rev. 2011;53:217-288.")

HighOrderMRI combines matrix-free per-dynamic factorization with an incrementally constructed shared spatial basis and a global-trajectory NFFT representation. The resulting shared basis is not an exact post-hoc global SVD of all dynamic-specific matrices. Consequently, the Eckart-Young optimality result for a truncated SVD of a fixed matrix does not apply directly to the completed incremental representation. [[5]](/references#ref-5 "Eckart C, Young G. The approximation of one matrix by another of lower rank. Psychometrika. 1936;1:211-218.")

A direct global SVD/rSVD remains a useful approximation-quality reference when memory permits because all dynamics can be considered jointly. The incremental method instead targets memory-bounded construction and scalable execution. These alternatives are distinguished explicitly in [Scientific validation strategy](/guide/validation).

Shared subspace and hierarchical POD constructions also have established mathematical foundations, including HAPOD. [[23]](/references#ref-23 "Himpe C, Leibner T, Rave S. Hierarchical approximate proper orthogonal decomposition. SIAM J Sci Comput. 2018;40:A3267-A3292.")

MRI encoding compression predates the joint builder: Compton et al. use randomized interpolative decomposition for field-corrected MRI; Cartesian MaxGIRF exploits reusable shot/readout structure; Tian and Scheffler jointly compress dynamic field and RF encoding within k-space subregion groups. [[18]](/references#ref-18 "Compton R, Osher S, Bouchard LS. Hybrid regularization for MRI reconstruction with static field inhomogeneity correction. Inverse Probl Imaging. 2013;7:1215-1233.") [[19]](/references#ref-19 "Lee NG, Cui SX, Nayak KS. Cartesian MaxGIRF: Model-based EPI reconstruction incorporating gradient nonlinearity and concomitant field effects. Magn Reson Med. 2026;95:1044-1067. First published online 2025.") [[20]](/references#ref-20 "Tian R, Scheffler K. Group-Patch Joint Compression: Compressing dynamic B0 and static RF spatial modulations across k-space subregion groups for highly accelerated MRI. Magn Reson Med. Published online 2026.")

The joint builder described below keeps the supplied coil/data dimensions and handles nonidentical phase histories. It combines weighted snapshot POD, geometric coverage, sampled least squares, and residual checks; it does not implement those papers in full or establish mathematical priority. Sampled matrix acceptance does not certify physical image accuracy.

For local fixed-precision approximation, Yu et al. give QB energy identities with orthogonality and finite-precision requirements. [[21]](/references#ref-21 "Yu W, Gu Y, Li Y. Efficient randomized algorithms for the fixed-precision low-rank matrix approximation. SIAM J Matrix Anal Appl. 2018;39:1339-1359.") Demmel et al. describe TSQR, a stable alternative to forming a tall matrix's Gram before SVD. [[22]](/references#ref-22 "Demmel J, Grigori L, Hoemmen M, Langou J. Communication-optimal parallel and sequential QR and LU factorizations. SIAM J Sci Comput. 2012;34:A206-A239.") These are numerical references, not additional constructor backends.

## Residual encoding matrix

For dynamic $d$, define

$$
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
$$

The zeroth-order temporal phase and the first-order Fourier terms are excluded from $H_d$. In 3D, $\mathcal R$ begins with the second-order terms. In 2D, the $z$ first-order term remains in $\mathcal R$ because only the $x$ and $y$ first-order terms are represented by the 2D NFFT. The full matrix $H_d$ is not explicitly materialized during low-rank setup.

## Per-dynamic randomized SVD

Let $L$ denote `L_rank`, $p$ denote `rsvd_oversample`, and $\ell=L+p$. For each dynamic, a random test matrix

$$
\Omega_d\in\mathbb C^{N_v\times\ell}
$$

is generated, and the randomized range sketch is computed as

$$
Y_d
=
H_d\Omega_d,
\qquad
Q_d
=
\operatorname{orth}(Y_d).
$$

The projected adjoint matrix is then

$$
B_d
=
H_d^H Q_d
\in
\mathbb C^{N_v\times\ell}.
$$

Products with $H_d$ and $H_d^H$ are evaluated matrix-free from `times`, `fieldmap`, the residual rows of `kspha`, and the corresponding spatial basis functions. For a fixed configuration, dynamic $d$ uses the deterministic seed

$$
s_d
=
s_0+d-1.
$$

### Direct SVD finalization

For `rsvd_finalize=:svd`, let

$$
B_d
=
P_d\Sigma_d Z_d^H.
$$

Retaining the first $L$ singular triplets gives

$$
U_d
=
Q_d Z_{d,L},
\qquad
\widetilde V_d
=
P_{d,L}\Sigma_{d,L},
$$

and therefore

$$
H_d
\approx
U_d\widetilde V_d^H.
$$

Because $Q_d$ and $Z_{d,L}$ have orthonormal columns,

$$
U_d^H U_d
=
I_L.
$$

The singular values are absorbed into the spatial factor $\widetilde V_d$. The retained local energy is therefore

$$
\eta_d
=
\lVert\widetilde V_d\rVert_F^2
=
\sum_{l=1}^{L}\sigma_{d,l}^2.
$$

### Small-Gram finalization

For `rsvd_finalize=:gram`, the implementation forms

$$
G_d
=
B_d^H B_d
\in
\mathbb C^{\ell\times\ell}
$$

and diagonalizes

$$
G_d
=
Z_d\Lambda_d Z_d^H,
\qquad
\Lambda_d
=
\Sigma_d^2.
$$

The retained factors are recovered as

$$
U_d
=
Q_d Z_{d,L},
\qquad
\widetilde V_d
=
B_d Z_{d,L}.
$$

If $B_d=P_d\Sigma_d Z_d^H$, then $B_dZ_{d,L}=P_{d,L}\Sigma_{d,L}$; thus, in exact arithmetic, the Gram and direct-SVD routes produce the same retained factorization. The Gram formulation avoids a direct SVD of the tall $N_v\times\ell$ matrix but squares its condition number. The single-device implementation checks for non-finite values, significant negative eigenvalues, and degeneracy and can fall back to the direct SVD if the Gram result fails these checks.

## Incremental shared spatial basis

Independent local factorizations would retain up to $N_dL$ spatial columns. HighOrderMRI instead constructs one orthonormal spatial basis sequentially across dynamics.

Before processing dynamic $d$, let

$$
S_{d-1}
\in
\mathbb C^{N_v\times R_{d-1}},
\qquad
S_{d-1}^H S_{d-1}
=
I.
$$

The current spatial factor is first projected onto the existing basis,

$$
C_d^{\mathrm{old}}
=
S_{d-1}^H\widetilde V_d,
$$

and the residual is

$$
R_d
=
\widetilde V_d
-
S_{d-1}C_d^{\mathrm{old}}.
$$

A second projection/subtraction is performed in the implementation and the corresponding correction is accumulated in $C_d^{\mathrm{old}}$ to reduce finite-precision loss of orthogonality.

The residual Gram matrix is decomposed as

$$
R_d^H R_d
=
T_d\Lambda_d^{(R)}T_d^H.
$$

The minimum number $r_d$ of additional basis vectors is selected such that

$$
\sqrt{
\frac{
\sum_{i=r_d+1}^{L}\lambda_{d,i}^{(R)}
}{
\eta_d
}}
\leq
\tau,
$$

where $\tau$ is `shared_basis_tol`. The denominator $\eta_d$ is the retained energy of the local rank-$L$ factor, not the energy of the untruncated matrix $H_d$.

For the selected positive residual eigenvalues, the appended basis vectors are

$$
S_{d,\mathrm{new}}
=
R_dT_{d,1:r_d}
\left[\Lambda_{d,1:r_d}^{(R)}\right]^{-1/2}.
$$

The basis is updated according to

$$
S_d
=
\begin{bmatrix}
S_{d-1}&S_{d,\mathrm{new}}
\end{bmatrix},
$$

and the coefficient block for the current dynamic is completed as

$$
C_d^{\mathrm{new}}
=
S_{d,\mathrm{new}}^H\widetilde V_d,
\qquad
\bar C_d
=
\begin{bmatrix}
C_d^{\mathrm{old}}\\
C_d^{\mathrm{new}}
\end{bmatrix}.
$$

At the time dynamic $d$ is inserted,

$$
\widetilde V_d
\approx
S_d\bar C_d.
$$

If the required accumulated rank would exceed `shared_rank_max`, setup terminates with an error rather than relaxing the requested rank bound.

### Streaming coefficient representation

The implementation is memory bounded: after dynamic $d$ has been processed, $\widetilde V_d$ is not retained. When later dynamics append basis columns, earlier coefficient blocks are extended with zeros rather than recomputed.

Let

$$
S
\equiv
S_{N_d}
\in
\mathbb C^{N_v\times R}
$$

be the completed shared basis, and let $\bar C_d\in\mathbb C^{R\times L}$ denote the stored coefficient block for dynamic $d$ after zero-padding to the final rank. The implementation satisfies the approximation

$$
\widetilde V_d
\approx
S\bar C_d.
$$

For an earlier dynamic, however, the stored coefficient block is generally not the orthogonal projection onto the completed basis:

$$
\bar C_d
\neq
S^H\widetilde V_d.
$$

Appending basis columns and padding earlier coefficient blocks with zeros leaves the previously stored approximation unchanged. The final coefficients should therefore be interpreted as streaming coefficients associated with the incremental construction, rather than as coefficients from a post-hoc global projection.

## Optional final global recompression

In the rSVD path, `global_basis_tol` optionally compresses the
completed representation once more. Write its unscaled temporal factor as
$\widehat q$ and assume $S^HS=I$. The eigenvalues of
$\widehat q^H\widehat q$ are then the squared singular values of
$\widehat qS^H$. If $Z_{R_g}$ contains the retained eigenvectors, rotate both
factors:

$$
\widehat q_{\mathrm{new}}=\widehat qZ_{R_g},
\qquad
S_{\mathrm{new}}=SZ_{R_g}.
$$

The relative squared Frobenius error is the discarded eigenvalue sum divided
by the total eigenvalue sum. For non-negligible total energy, the smallest
positive rank whose ratio is at most `global_basis_tol^2` is selected.
This measures additional error relative
to the already compressed representation. It does not include local rSVD error,
incremental merging error, or missing earlier coefficients. The default
`global_basis_tol=nothing` preserves the input representation. Finite-precision
Gram and basis-orthogonality effects still need numerical validation.

## Direct joint shared basis

The `shared_basis_method=:joint` path constructs $S$ jointly from
representative phase snapshots across dynamics, then fits $\widehat q$
directly. It bypasses the per-dynamic $U_d,\widetilde V_d$ construction above.
The final residual model remains

$$
H\approx\widehat qS^H,\qquad
H\in\mathbb C^{(N_sN_d)\times N_v},\quad
\widehat q\in\mathbb C^{(N_sN_d)\times R},\quad
S\in\mathbb C^{N_v\times R}.
$$

Here $H$ stacks all dynamics with samples varying fastest. It is an implicit
matrix, not a full allocated phase array. Within this section, row indices refer to the stacked matrix. The snapshot budget $K$, fitting
voxel count $J$, and final spatial rank $R$ are separate quantities.

### Phase geometry and representative snapshots

Let $a_j$ contain time and residual field coefficients, and let $b_v$ contain
the field map and matching spatial basis values, so that
$H(j,v)=\exp(i2\pi a_j^Tb_v)$. Compute

$$
\mu=\frac{1}{N_v}\sum_v b_v,\qquad
G_b=\frac{1}{N_v}\sum_v(b_v-\mu)(b_v-\mu)^T.
$$

The phase distance used for representative selection is

$$
d(a,a')^2=(a-a')^TG_b(a-a').
$$

For $\widetilde h_a(v)=\exp(i2\pi a^T(b_v-\mu))$,
$|e^{ix}-e^{iy}|\leq|x-y|$ gives the exact-arithmetic bound

$$
\frac{\|\widetilde h_a-\widetilde h_{a'}\|_2}{\sqrt{N_v}}
\leq 2\pi d(a,a').
$$

The removed spatially constant phase can be absorbed into $\widehat q$.
Only the selection metric is centred; the actual encoding phases are not
changed. This bound motivates geometric coverage, but it is not a proof
that a particular snapshot budget covers all dynamics or extreme-field voxels.

The current selection uses a candidate time grid with stride 16 and the last
sample. When there are at least four dynamics, approximately 10% are held out
from snapshot selection. Farthest-point selection chooses up to $K$
representatives; coincident geometries can reduce that count. Coverage counts
provide snapshot weights. All dynamics, including the held-out ones, are
subsequently checked.

### Weighted snapshot POD

For selected phase rows $h_{j_k}=H(j_k,:)$ and coverage counts $w_k$, define

$$
A_{\mathrm{snap}}=
\begin{bmatrix}
\sqrt{w_1}h_{j_1}^H & \cdots & \sqrt{w_K}h_{j_K}^H
\end{bmatrix}.
$$

The leading left singular subspace minimizes the weighted snapshot error:

$$
\min_{S^HS=I_R}\|A_{\mathrm{snap}}-SS^HA_{\mathrm{snap}}\|_F^2
=\sum_{r>R}\sigma_r(A_{\mathrm{snap}})^2.
$$

This is the POD/truncated-SVD objective, following Eckart--Young and the
low-rank approximation framework reviewed by Halko et al.
[[5]](/references#ref-5 "Eckart C, Young G. The approximation of one matrix by another of lower rank. Psychometrika. 1936;1:211-218.") [[4]](/references#ref-4 "Halko N, Martinsson PG, Tropp JA. Finding structure with randomness. SIAM Rev. 2011;53:217-288."). The optimum applies to
$A_{\mathrm{snap}}$, not automatically to the complete $H$ or to the worst dynamic.

The implementation generates conjugated phase snapshots in bounded voxel
chunks, accumulates $A_{\mathrm{snap}}^HA_{\mathrm{snap}}$, and builds $S$ in a second pass. A small
eigendecomposition and a Cholesky-based correction provide an approximately
orthonormal basis. Directions below `eps(T)` times the largest Gram eigenvalue
are dropped before normalization. The Gram is formed in the requested input
precision; converting it to Float64 for its eigendecomposition cannot recover
weak directions already lost in Float32 arithmetic.

### Oversampled coefficient fitting

For fitting voxel indices $\mathcal J$, let $S_J=S[\mathcal J,:]$. Coefficients
are obtained by QR-based least squares:

$$
\widehat q_{\mathrm{fit}}
=\arg\min_Q\|H[:,\mathcal J]-QS_J^H\|_F
=H[:,\mathcal J](S_J^\dagger)^H,
$$

where the last identity assumes $S_J$ has full column rank. The code computes
the least-squares map using QR rather than normal equations.

The fitting pool contains at most 8192 seeded random voxels and at most half
the mask. Pivoted QR of the sampled basis adjoint selects anchor points;
random points supply the remaining oversampling. This combines Q-DEIM-style
selection with gappy POD. Drmač and Gugercin provide the QR-selection
foundation; Peherstorfer et al. analyze oversampled gappy POD stability
[[16]](/references#ref-16 "Drmač Z, Gugercin S. A new selection operator for the discrete empirical interpolation method—Improved a priori error bound and extensions. SIAM J Sci Comput. 2016;38:A631-A648.") [[17]](/references#ref-17 "Peherstorfer B, Drmač Z, Gugercin S. Stability of discrete empirical interpolation and gappy proper orthogonal decomposition with randomized and deterministic sampling points. SIAM J Sci Comput. 2020."). The present pool and
oversampling policy are not the exact GappyPOD+E algorithm, so its published
probability bounds are not inherited unchanged.

For any orthonormal $S$, the error separates exactly as

$$
\boxed{
\|H-\widehat qS^H\|_F^2
=\|H-HSS^H\|_F^2+\|HS-\widehat q\|_F^2.
}
$$

The first term measures inadequate spatial span; the second measures
coefficient error. With $E_\perp=H-HSS^H$ and full-column-rank $S_J$,

$$
\|\widehat q_{\mathrm{fit}}-HS\|_F
\leq \frac{\|E_\perp[:,\mathcal J]\|_F}{\sigma_{\min}(S_J)}.
$$

Increasing $R$ and improving the fitting samples therefore address different
errors. The package joint path uses sampled fitting; the full $\widehat q=HS$
projection investigated in the EPI experiments is not this implementation.

### Rank selection and final audit

The full requested snapshot budget is used before rank selection. The search
tests $R=1,8,16,\ldots$ and the final available rank, recomputing the
least-squares coefficients for each tested prefix. If no rank passes, it can
double $J$ once, bounded by the fitting pool. It does not automatically grow
$K$ or search every integer rank.

For each dynamic separately, selection checks evaluate a sampled relative
Frobenius error against phases recomputed in Float64 from the supplied inputs.
Selection requires error at most `0.9 * shared_basis_tol`. After generating the
complete factors, a fresh time-sample audit uses `shared_basis_tol` itself.
The margin is heuristic and does not establish a confidence level.

For large inputs, the current fixed limits are 193 selection and 257 audit
time samples per dynamic, 1024 random validation voxels, and an additional
128 high-$|B_0|$ voxels. Both spatial sets exclude the fitting pool; selection
and audit voxel sets may overlap. Their time samples are disjoint and exclude
the snapshot candidate grid when enough samples exist. Tiny inputs report
`disjoint_times=false` when time separation is impossible.

`joint_roi_tol` optionally imposes a separate high-$|B_0|$ tolerance with the
same selection margin. With `nothing`, an excessive ROI error is reported and
warned about, but does not reject the operator. This option changes acceptance,
not the fitting-point selection policy.

Rank/sample exhaustion and a failed final audit throw, with diagnostics
available through `joint_basis_report`. There is no accepted partial result
or silent CPU fallback. Successful sampled checks are not a full-matrix,
uniform-voxel, minimum-rank, or reconstruction-image guarantee.

## Final residual representation

For the rSVD path, define the unscaled sample-domain factor

$$
\widehat q_d
=
U_d\bar C_d^H
\in
\mathbb C^{N_s\times R}.
$$

Then

$$
H_d
\approx
\widehat q_d S^H.
$$

All $\widehat q_d$ blocks are concatenated with samples as the fastest-changing index and dynamics as the next index. Joint construction directly supplies the same unscaled representation. Optional final global recompression applies only to the rSVD path. The stored coefficient matrix additionally incorporates three sample-domain factors:

1. the zeroth-order phase $\exp(i2\pi k_{0,jd})$;
2. the parity-dependent NFFT centre correction;
3. the symmetric normalization $1/\sqrt{N_v}$.

If $d_{0,jd}$ denotes the zeroth-order phase and $c_{\mathrm{ctr},jd}$ the centre correction, the stored rows are

$$
q_d(j,:)
=
\frac{d_{0,jd}\,c_{\mathrm{ctr},jd}}{\sqrt{N_v}}
\widehat q_d(j,:).
$$

The first-order Fourier phase remains in the NFFT and is not included in $H_d$ or $\widehat q_d$.

## Forward and adjoint operators

Let $\mathcal F$ denote the global NFFT containing the first-order trajectories of all dynamics, let $s_r$ denote column $r$ of the completed shared basis, and let $q_r$ denote column $r$ of the stored sample-domain coefficient matrix. For receive coil $c$,

$$
A_c m
\approx
\sum_{r=1}^{R}
\operatorname{diag}(q_r)
\mathcal F
\left(m\odot C_c\odot s_r^*\right).
$$

The complex conjugation of $s_r$ follows from $H_d\approx\widehat q_dS^H$.

The corresponding multi-coil adjoint is

$$
A^Hy
\approx
\sum_{r=1}^{R}
s_r\odot
\sum_{c=1}^{N_c}
C_c^*\odot
\mathcal F^H
\left(q_r^*\odot y_c\right).
$$

These expressions agree with the implemented conjugation pattern: the forward operator multiplies the image by the conjugated shared-basis vector and coil sensitivity before the NFFT, whereas the adjoint applies the conjugated temporal coefficient, the NFFT adjoint, the conjugated coil sensitivity, and the non-conjugated shared-basis vector.

The number of NFFT evaluations per forward or adjoint application is

$$
N_{\mathrm{NFFT}}
=
R N_c.
$$

For independent per-dynamic spatial factorizations, the corresponding transform count would scale approximately as $N_dLN_c$. This comparison concerns the number of NFFT evaluations and does not imply runtime independence from the number of samples or dynamics.

## Two-stage approximation error

For the rSVD path, before applying zeroth-order, centre-correction, and normalization factors, the approximation is

$$
H_d
\longrightarrow
U_d\widetilde V_d^H
\longrightarrow
U_d\bar C_d^H S^H
=
\widehat q_dS^H.
$$

The triangle inequality gives

$$
\begin{aligned}
\lVert H_d-\widehat q_dS^H\rVert_F
&\leq
\lVert H_d-U_d\widetilde V_d^H\rVert_F\\
&\quad+
\lVert U_d(\widetilde V_d-S\bar C_d)^H\rVert_F.
\end{aligned}
$$

Since $U_d^HU_d=I_L$,

$$
\lVert U_d(\widetilde V_d-S\bar C_d)^H\rVert_F
=
\lVert\widetilde V_d-S\bar C_d\rVert_F,
$$

and therefore

$$
\lVert H_d-\widehat q_dS^H\rVert_F
\leq
\lVert H_d-U_d\widetilde V_d^H\rVert_F
+
\lVert\widetilde V_d-S\bar C_d\rVert_F.
$$

The first term is the local rSVD approximation error and the second term is the incremental shared-basis approximation error. `shared_basis_tol` controls only the second stage relative to the retained local energy $\eta_d$; it does not bound the local rSVD truncation error relative to the full matrix $H_d$.

## Parameter interpretation

For the default rSVD construction:

| Parameter | Role | Recommended check |
|---|---|---|
| `L_rank` | Local truncation rank | Sweep against an explicit operator or independent dense reference |
| `rsvd_oversample` | Additional random sketch directions | Verify `L_rank + rsvd_oversample ≤ min(nSam,nVox)` |
| `rsvd_seed` | Reproducible random sketch schedule | Repeat with several seeds in a final sensitivity analysis |
| `rsvd_finalize` | `:svd` or memory-saving `:gram` | Compare retained spectra and operator errors on a tractable problem |
| `shared_basis_tol` | Incremental second-stage residual tolerance | Report together with the final `shared_rank` |
| `shared_rank_max` | Hard cap on accumulated shared rank | Treat an exceeded cap as a configuration failure |
| `rsvd_backend` | `:chunked` or fused CUDA `:kernel` | The kernel supports wide sketches in batches of at most 16 |
| `global_basis_tol` | Optional final rSVD representation compression | Additional error relative to the already shared approximation; default `nothing` |

The local rank $L$ and final shared rank $R$ describe different approximations. $L$ controls the per-dynamic rSVD truncation, whereas $R$ is the dimension accumulated by the shared spatial representation. Thus, $R$ may be smaller than, equal to, or larger than $L$.

For `:joint`, `L_rank` and `rsvd_oversample` do not select $R$. Instead,
`joint_snapshots` controls $K$, `joint_samples` controls the initial $J$, and
`shared_rank_max` caps $R$. `shared_basis_tol` then denotes a sampled
original-phase matrix target, not the incremental local-factor tolerance
above. `global_basis_tol` must be `nothing` to preserve the audited factors.
See the [joint usage example](/guide/operators#automatic-joint-shared-basis)
and the [API parameter table](/reference/highorderlowrankop#shared-basis-parameters).

Rank selection should not be based on a single reconstructed image. At minimum, forward error, adjointness, normal-operator error, reconstruction error, seed sensitivity, shared rank, and final solver residual should be examined. The [Scientific validation strategy](/guide/validation) defines the comparison hierarchy, and the [Reconstruction protocol](/guide/reconstruction-protocol) defines the fixed metrics and timing boundaries.

## Validation scope

Regression tests assess consistency of phase sign, coordinate convention, NFFT node mapping, centring, normalization, masking, data layout, and adjointness on tractable problems. These tests establish implementation consistency but do not constitute an independent physical reference. Validation of absolute physical accuracy requires independent simulation or measured reference data.
