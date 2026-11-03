<script setup>
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'

const { isDark } = useData()
const sharedBasisFigure = computed(() => withBase(
  `/figures/adaptive-shared-basis/Fig1_adaptive_shared_basis${isDark.value ? '_dark' : ''}.svg`
))
</script>

# Low-rank shared subspace

By default, `HighOrderLowRankOp` reduces repeated encoding cost with two
successive approximations: local randomized SVD (rSVD), then incremental
sharing of the retained spatial factors. An optional final recompression
introduces a third approximation stage. The image and coil maps are not
low-rank approximated. The supplied Fourier trajectory is retained, while
the numerical NFFT approximation has its own error.

Symbols follow [Symbols and notation](/theory/symbols). For optional global
recompression, this page distinguishes the accumulated rank $R_0$ and factors
$S^{(0)},\widehat q^{(0)}$ from the final rank $R$ and factors
$S,\widehat q$. With recompression disabled, these are the same. Each dynamic
is an encoding profile of the same unknown image, not a separately
reconstructed frame.

Readouts use index $i$, with $N_t$ samples each and $N_{\mathrm{ro}}$ readouts (`nSam`, `nDyn` in Julia).

## Method overview

<a :href="sharedBasisFigure">
  <img :src="sharedBasisFigure" alt="Signal model, shared-basis reconstruction, and sequential adaptive spatial compression across readouts i" />
</a>

**Figure 1. Adaptive shared-basis reconstruction.**
(a) The signal model separates zeroth-order phase, Fourier encoding, and
residual field modulation. (b) Local rSVD factors are combined into a shared
spatial basis and readout-specific temporal factors. (c) Readouts $i$ are
processed sequentially; residual spatial modes extend the basis, and earlier
coefficient blocks are zero-padded. Here
$\widetilde V_i=V_{i,L}\Sigma_{i,L}$, the local rank $L$ is fixed, and the shared
rank $R$ adapts and may exceed $L$.

The figure shows the default shared rSVD construction with
`global_basis_tol=nothing`, so $R=R_0$. The tolerance $\tau$ controls relative
spatial-factor error, not image error. Without sample batching, the indicated
NUFFT counts apply per coil to one forward or adjoint application; they are
not runtime speedup factors. <a :href="sharedBasisFigure">Open the full-size SVG</a>.

This schematic omits [optional final recompression](#optional-final-global-recompression).
The stored temporal factors also include normalization and the NFFT centre
correction, as detailed in [phase folding](#final-residual-representation).
The full residual model, including the [2D residual terms](#residual-encoding-matrix)
and [optional concomitant phase](#concomitant-phase-in-the-residual-model),
is described below.

The default path is easiest to follow in this order:

1. [Local rSVD](#per-dynamic-randomized-svd) constructs $U_{i,L},\widetilde V_i$.
2. [Incremental sharing](#incremental-shared-spatial-basis) builds a spatial
   basis and saved coefficient blocks; earlier blocks are zero-padded when
   the basis grows.
3. [Temporal assembly](#final-residual-representation) forms
   $\widehat q_i^{(0)}=U_{i,L}\bar C_i^H$.
4. [Optional recompression](#optional-final-global-recompression) rotates
   and may truncate the completed representation.
5. [Phase folding and NFFT](#forward-and-adjoint-operators) implement the
   final forward, adjoint, and weighted normal operators.

The local and incremental stages introduce separate approximation errors.
Temporal assembly and phase folding add no intended truncation. Optional
final recompression adds error relative to the already shared representation.
It cannot recover components discarded earlier.

## Relation to previous methods

Separable temporal-spatial representations of higher-order MRI encoding have previously been obtained using singular-vector separation, and related decompositions are used in expanded-model reconstruction such as MaxGIRF. [[2]](/references#ref-2 "Wilm BJ, Barmet C, Pruessmann KP. Fast higher-order MR image reconstruction using singular-vector separation. IEEE Trans Med Imaging. 2012;31:1396-1403.") [[3]](/references#ref-3 "Lee NG, Ramasawmy R, Lim Y, Campbell-Washburn AE, Nayak KS. MaxGIRF: Image reconstruction incorporating concomitant field and gradient impulse response function effects. Magn Reson Med. 2022;88:691-710.") The randomized range-finding step used here follows standard rSVD methods. [[4]](/references#ref-4 "Halko N, Martinsson PG, Tropp JA. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Rev. 2011;53:217-288.")

HighOrderMRI combines matrix-free per-dynamic factorization with an incrementally constructed shared spatial basis and a global-trajectory NFFT representation. The resulting shared basis is not an exact post-hoc global SVD of all dynamic-specific matrices. Consequently, the Eckart-Young optimality result for a truncated SVD of a fixed matrix does not apply directly to the completed incremental representation. [[5]](/references#ref-5 "Eckart C, Young G. The approximation of one matrix by another of lower rank. Psychometrika. 1936;1:211-218.")

A direct global SVD/rSVD remains a useful approximation-quality reference when memory permits because all dynamics can be considered jointly. The incremental method instead targets memory-bounded construction and scalable execution. These alternatives are distinguished explicitly in [Scientific validation strategy](/guide/validation).

Shared subspace and hierarchical POD constructions also have established mathematical foundations, including HAPOD. [[23]](/references#ref-23 "Himpe C, Leibner T, Rave S. Hierarchical approximate proper orthogonal decomposition. SIAM J Sci Comput. 2018;40:A3267-A3292.")

For local fixed-precision approximation, Yu et al. give QB energy identities with orthogonality and finite-precision requirements. [[21]](/references#ref-21 "Yu W, Gu Y, Li Y. Efficient randomized algorithms for the fixed-precision low-rank matrix approximation. SIAM J Matrix Anal Appl. 2018;39:1339-1359.") Demmel et al. describe TSQR, a stable alternative to forming a tall matrix's Gram before SVD. [[22]](/references#ref-22 "Demmel J, Grigori L, Hoemmen M, Langou J. Communication-optimal parallel and sequential QR and LU factorizations. SIAM J Sci Comput. 2012;34:A206-A239.") These are numerical references, not additional constructor backends.

## Residual encoding matrix

For dynamic $i$, define

$$
H_i(j,v)
=
\exp\!\left\{
\mathrm{i}2\pi\left[
t_{ji}\Delta f_{0,v}
+\sum_{\ell\in\mathcal R}
k_{\ell,ji}b_{\ell,v}
+\phi_c(j,v,i)
\right]\right\},
\qquad
H_i\in\mathbb C^{N_t\times N_v}.
$$

The zeroth-order temporal phase and the first-order Fourier terms are excluded from $H_i$. In 3D, $\mathcal R$ begins with the second-order terms. In 2D, the $z$ first-order term remains in $\mathcal R$ because only the $x$ and $y$ first-order terms are represented by the 2D NFFT. The setup products do not require a full $H_i$. The chunked backend does
materialize phase and encoding tiles; a chunk covering every masked voxel
can materialize the complete matrix for that dynamic. Fused kernels accumulate
the products directly. Matrix-free products still require sketch, factor,
and workspace storage.

## Concomitant phase in the residual model

The optional term in the residual exponent is zero when `concomitant=nothing`. For supplied coefficients,

$$
\phi_c(j,v,i)=c_1(j,i)z_v^2+c_2(j,i)(x_v^2+y_v^2)
+c_3(j,i)x_vz_v+c_4(j,i)y_vz_v.
$$

The four basis columns and corresponding coefficient rows are appended before residual basis construction. This applies to the selected shared or per-readout representation; the nominal first-order NFFT trajectory is unchanged. `recon_terms` selects only the original spherical-harmonic rows, and `kspha` still has 9 or 16 rows at the public interface.

```julia
op = HighOrderLowRankOp(grid, kspha_dynamic, times_dynamic;
    fieldmap, csm, mask,
    concomitant=concomitant_coefficients,  # (4,nSam,nDyn), cycles/m²
)
```

A single-dynamic call also accepts `(4,nSam)`. With `concomitant=nothing`, no extra phase buffers are appended. Match the coefficient coordinate frame and physical origin to the grid, and convert rad/m² to cycles/m² by dividing by $2\pi$. Do not multiply these integrated coefficients by `times` again. See the [encoding model](/theory/encoding-model#optional-concomitant-phase) for the common convention.

## Per-dynamic randomized SVD

Let $L$ denote `L_rank`, $p$ denote `rsvd_oversample`, and $L_{\mathrm{sk}}=L+p$. For each dynamic, a random test matrix

$$
\Omega_i\in\mathbb C^{N_v\times L_{\mathrm{sk}}}
$$

is generated, and the randomized range sketch is computed as

$$
Y_i
=
H_i\Omega_i,
\qquad
Q_i
=
\operatorname{orth}(Y_i).
$$

The projected adjoint matrix is then

$$
B_i
=
H_i^H Q_i
\in
\mathbb C^{N_v\times L_{\mathrm{sk}}}.
$$

Products with $H_i$ and $H_i^H$ are evaluated matrix-free from `times`, `fieldmap`, the residual rows of `kspha`, and the corresponding spatial basis functions. For a fixed configuration, dynamic $i$ uses the deterministic seed

$$
s_i
=
s_0+i-1.
$$

The current implementation uses one Gaussian sketch and QR range finding,
without power iteration. The retained rank $L$ is configured rather than
chosen by an original-phase error tolerance. The sketch width must satisfy
$L+p\leq\min(N_t,N_v)$.

With voxel-distributed setup, worker $g$ uses the seed
$s_{i,g}=s_i+1{,}000{,}003(g-1)$ for its sketch rows. Reproducibility therefore
requires the same backend and voxel partition, not just the same base seed.

### Direct SVD finalization

For `rsvd_finalize=:svd`, let

$$
B_i
=
P_i\Sigma_i (Z_i^{\mathrm{svd}})^H.
$$

Retaining the first $L$ singular triplets gives

$$
U_{i,L}
=
Q_i Z_{i,L}^{\mathrm{svd}},
\qquad
\widetilde V_i
=
P_{i,L}\Sigma_{i,L},
$$

and therefore

$$
H_i
\approx
U_{i,L}\widetilde V_i^H.
$$

Because $Q_i$ and $Z_{i,L}^{\mathrm{svd}}$ have orthonormal columns,

$$
U_{i,L}^H U_{i,L}
=
I_L.
$$

The singular values are absorbed into the spatial factor $\widetilde V_i=V_{i,L}\Sigma_{i,L}$, where $V_{i,L}=P_{i,L}$. The retained local energy is therefore

$$
\eta_i
=
\lVert\widetilde V_i\rVert_F^2
=
\sum_{l=1}^{L}\sigma_{i,l}^2.
$$

### Small-Gram finalization

For `rsvd_finalize=:gram`, the implementation forms

$$
G_i
=
B_i^H B_i
\in
\mathbb C^{L_{\mathrm{sk}}\times L_{\mathrm{sk}}}
$$

and diagonalizes

$$
G_i
=
Z_i^{\mathrm{svd}}\Lambda_i (Z_i^{\mathrm{svd}})^H,
\qquad
\Lambda_i
=
\Sigma_i^2.
$$

The retained factors are recovered as

$$
U_{i,L}
=
Q_i Z_{i,L}^{\mathrm{svd}},
\qquad
\widetilde V_i
=
B_i Z_{i,L}^{\mathrm{svd}}.
$$

If $B_i=P_i\Sigma_i (Z_i^{\mathrm{svd}})^H$, then $B_iZ_{i,L}^{\mathrm{svd}}=P_{i,L}\Sigma_{i,L}$; thus, in exact arithmetic, the Gram and direct-SVD routes produce the same retained factorization. The Gram formulation avoids a direct SVD of the tall $N_v\times L_{\mathrm{sk}}$ matrix but squares its condition number. The single-device implementation checks for non-finite values, significant negative eigenvalues, and degeneracy and can fall back to the direct SVD if the Gram result fails these checks.
Voxel-distributed setup requires Gram finalization and reports failure when
its eigensystem is invalid; it does not silently switch to a distributed SVD.

## Independent per-readout representation

Set `basis_layout=:per_readout` to retain each local pair
$U_{i,L},\widetilde V_i$ directly, bypassing the shared-basis construction:

```julia
op = HighOrderLowRankOp(grid, kspha, times;
    fieldmap, csm, mask,
    basis_layout=:per_readout,
    L_rank=6,
    rsvd_oversample=5,
)
```

For each readout and coil, the forward operation is

$$
A_{i,c}m \approx \sum_{l=1}^{L}
\operatorname{diag}(u_{i,l})\mathcal F_i
\left(m\odot C_c\odot\widetilde v_{i,l}^{*}\right).
$$

The same zeroth-order phase, centre correction, and normalization are folded
into the temporal factors. With resident factors, one NFFT plan is cached per readout, giving
$N_{\mathrm{ro}} L N_c$ transforms per forward or adjoint evaluation. Streaming and
regeneration instead update a plan for the current readout within each cache. The spatial
factor bank has $N_v N_{\mathrm{ro}} L$ complex entries; cached plans also consume
memory. This baseline can therefore require substantially more memory than
the shared layout.

### CPU storage and GPU lookahead

To bound GPU memory for a large independent factor bank, keep the factors in
host RAM and use a readout queue:

```julia
op = HighOrderLowRankOp(grid, kspha, times;
    fieldmap, csm, mask,
    arrayType=CuArray,
    basis_layout=:per_readout,
    per_readout_storage=:cpu,
    per_readout_cache_size=5,
    L_rank=6,
    rsvd_oversample=5,
)
```

Each normal-operator worker caches at most five readouts in this example. A
separate CUDA stream copies future factors through one pinned host staging
buffer. A single NFFT plan per worker updates its trajectory on the compute
stream. Ready/consumed events prevent a slot from being used
before its transfer completes or overwritten before its transforms finish.
Completed slots are replenished while another slot computes. The same queue
is used by the forward, adjoint, and multi-GPU normal operator. The
source operator owns a separate queue for forward and standalone adjoint
evaluations on the primary GPU.

Once the multi-GPU normal backend
has been initialized, adjoints reuse its workers and queues, distributing
readouts and evaluating all coils for each readout together. Each spatial
factor block is then transferred once across the worker group. This also
parallelizes the solver's initial adjoint without changing its weights or
independent spatial factors.

For CG itself, select `normal_distribution=:readout` to assign disjoint
readout ranges to workers, evaluating all coils for each assigned readout.
Each factor block is then streamed once across the worker group per normal
application, while all coil sensitivity maps are kept on each active GPU.
`normal_distribution=:channel` retains the existing coil partition, with each
worker traversing every readout. These compute the same weighted normal
operator up to floating-point summation order.

Only storage and scheduling change: the full independent spatial bank still
occupies $N_v N_{\mathrm{ro}} L$ complex entries in host RAM. The default `per_readout_storage=:gpu` materializes independent factors and
plans on the selected array backend: GPU for `CuArray`, CPU for `Array`.
The `:cpu` and `:regenerate` modes require CUDA and bound cache slots rather
than the complete host/device memory footprint.

### Pipelined setup aggregation

Voxel-distributed per-readout setup uses `per_readout_gather=:pipeline` by
default. Each GPU has two pinned host buffers and a separate collector worker.
Downloads and host writes can overlap later rSVD phases. A CUDA event delays
the next overwrite of the spatial factor until its download finishes; a host
buffer is reused only after its final write completes. The queue is bounded
to two outstanding readouts. `per_readout_gather=:synchronous` retains the
blocking gather for comparisons. The independent factors and their precision
are unchanged, and the full host factor bank is still required.

Provide threads for compute workers, factor collectors, and a coordinator
when using the pipeline. A smaller thread count can reduce overlap while
the queue remains bounded. See [runtime requirements](/guide/multi-gpu#runtime-requirements).

### On-demand regeneration

When the independent spatial bank exceeds host RAM, use
`per_readout_storage=:regenerate` with `rsvd_distribution=:voxel`,
`rsvd_backend=:kernel` and `rsvd_finalize=:gram`. This retains each readout's
original unmodulated $Q_i$ and Gram rotation $Z_i^{\mathrm{svd}}$ and generates

$$
\widetilde V_i = (H_i^H Q_i) Z_{i,L}^{\mathrm{svd}}
$$

with the existing matrix-free adjoint kernel. Uncached spatial blocks are generated as needed, rather than materializing
the complete spatial bank. No additional singular-value truncation is
introduced; GPU partition/GEMM geometry may change rounding, so validate the
factor and operator errors for the workload. The temporal normalization and
zeroth-order/centre phases remain in the operator's `q`, separately from Q/Z.

`per_readout_cache_size=1` generates and consumes factors serially. Values of
`2` or more enable an independent producer task/stream to prepare future
readouts while the consumer evaluates all coils of the current readout.
Each slot has ready/consumed events and is never overwritten while in use.
Generation and NUFFT share GPU compute resources; lookahead is an opportunity
to overlap work, not a guarantee that generation cost disappears.

`per_readout_host_cache_bytes` sets an optional host factor budget (default
zero). Whole factor blocks from a fixed, evenly spaced subset are materialized
during setup; all GPU workers reuse the same bank. Fixed residency avoids
evicting every cached readout during repeated sequential CG sweeps. The budget
excludes Q/Z, field maps, workspaces and pinned staging. The budget applies only to resident spatial factor blocks, not to Q/Z,
the phase model, generator workspaces, or GPU slots.
The source operator and each normal worker own bounded GPU slots and a
generator workspace; `close(op)` releases these workers/caches, which are
recreated lazily if the operator is subsequently reused.

### Comparison scope

Both representations use the same local rSVD model. Changing
`basis_layout` changes how retained factors are stored and combined.
`shared_rank_max` and `shared_basis_tol` are unused for independent factors;
`global_basis_tol` must be `nothing`, and `shared_basis_method` must be
`:rsvd`. A zero shared tolerance still constructs a shared basis and does
not select the independent representation. See the
[storage reference](/reference/highorderlowrankop#per-readout-parameters).

The independent factorization and per-readout NUFFT sums follow the
[MAXGIRF author's encoding implementation](https://github.com/usc-mrel/lowfield_maxgirf/blob/637719ce1d96e26323803c0abe7607e8d598d3c0/src/encoding_lowrank_maxgirf.m).
This switch retains HighOrderMRI's local rSVD implementation. Supply
`concomitant` separately to include its phase; the layout does not estimate
physical coefficients or promise bitwise reproduction of the MATLAB
reconstruction.

## Incremental shared spatial basis

Independent local factorizations would retain up to $N_{\mathrm{ro}}L$ spatial columns. HighOrderMRI instead constructs one orthonormal spatial basis sequentially across dynamics.

Before processing dynamic $i$, let

$$
S_{i-1}
\in
\mathbb C^{N_v\times R_{i-1}},
\qquad
S_{i-1}^H S_{i-1}
=
I.
$$

The current spatial factor is first projected onto the existing basis,

$$
C_i^{\mathrm{old}}
=
S_{i-1}^H\widetilde V_i,
$$

and the residual is

$$
Z_i
=
\widetilde V_i
-
S_{i-1}C_i^{\mathrm{old}}.
$$

A second projection/subtraction is performed in the implementation and the corresponding correction is accumulated in $C_i^{\mathrm{old}}$ to reduce finite-precision loss of orthogonality.

The residual Gram matrix is decomposed as

$$
Z_i^H Z_i
=
T_i\Lambda_i^{(R)}T_i^H.
$$

The residual Gram is accumulated in ComplexF64 on both local and
voxel-distributed paths, including for Float32 input factors. Its small
eigendecomposition and spectral rank selection use Float64; stored basis
vectors, coefficients, and NFFT products retain the requested precision.
This behavior is automatic. It applies to the incremental shared residual
Gram, separately from local rSVD finalization.

The minimum number $r_i$ of additional basis vectors is selected such that

$$
\sqrt{
\frac{
\sum_{h=r_i+1}^{L}\lambda_{i,h}^{(R)}
}{
\eta_i
}}
\leq
\tau,
$$

where $\tau$ is `shared_basis_tol`. The denominator $\eta_i$ is the retained
energy of the local rank-$L$ factor, not the untruncated $H_i$ energy.
Nonnegative residual eigenvalues below a numerical floor are set to zero
before this selection. Near-zero retained energy also returns without adding
directions. Reported shared errors are therefore numerical residual estimates,
not rigorous certificates of arbitrarily small original-phase error.

For the selected positive residual eigenvalues, the appended basis vectors are

$$
S_{i,\mathrm{new}}
=
Z_iT_{i,1:r_i}
\left[\Lambda_{i,1:r_i}^{(R)}\right]^{-1/2}.
$$

Write the appended residual modes as $Q_i^{\mathrm{add}}=S_{i,\mathrm{new}}$; these differ from the local randomized range basis $Q_i$. The basis is updated according to

$$
S_i
=
\begin{bmatrix}
S_{i-1}&S_{i,\mathrm{new}}
\end{bmatrix},
$$

and the coefficient block for the current dynamic is completed as

$$
C_i^{\mathrm{new}}
=
S_{i,\mathrm{new}}^H\widetilde V_i,
\qquad
\bar C_i
=
\begin{bmatrix}
C_i^{\mathrm{old}}\\
C_i^{\mathrm{new}}
\end{bmatrix}.
$$

At the time dynamic $i$ is inserted,

$$
\widetilde V_i
\approx
S_i\bar C_i.
$$

If the required accumulated rank would exceed `shared_rank_max`, setup terminates with an error rather than relaxing the requested rank bound.

### Streaming coefficient representation

The implementation is memory bounded: after dynamic $i$ has been processed, $\widetilde V_i$ is not retained. When later dynamics append basis columns, earlier coefficient blocks are extended with zeros rather than recomputed.

Let

$$
S^{(0)}
\equiv
S_{N_{\mathrm{ro}}}
\in
\mathbb C^{N_v\times R_0}
$$

be the completed shared basis, and let $\bar C_i\in\mathbb C^{R_0\times L}$ denote the stored coefficient block for dynamic $i$ after zero-padding to the accumulated rank $R_0$. The implementation satisfies the approximation

$$
\widetilde V_i
\approx
S^{(0)}\bar C_i.
$$

For an earlier dynamic, however, the stored coefficient block is generally not the orthogonal projection onto the completed basis:

$$
\bar C_i
\neq
(S^{(0)})^H\widetilde V_i.
$$

Appending basis columns and padding earlier coefficient blocks with zeros leaves the previously stored approximation unchanged. The final coefficients should therefore be interpreted as streaming coefficients associated with the incremental construction, rather than as coefficients from a post-hoc global projection.

### Optional no-add check

`shared_noadd_check=false` is the default. With voxel-distributed shared
rSVD, the opt-in check can avoid the residual-Gram eigendecomposition when
the existing basis is already sufficient. It keeps the original two-pass
projection, coefficient correction, and full residual computation.

For a nonempty shared basis, workers sum the residual energy in FP64:

$$
e_i=\sum_g\lVert Z_{i,g}\rVert_F^2.
$$

A conservative guard compares this energy with $\tau^2\eta_i$. Only a safely
sub-threshold result skips residual-Gram construction and its eigensolve.
The first update, near-threshold or non-finite energies, and insufficient
guard margins use the original Gram path. The diagnostic on an early return
uses the full residual norm and can differ from the original diagnostic after
tiny eigenvalues are discarded; that difference is not used to construct factors.

This is a setup optimization for the default rSVD method. It neither prunes
ranks during CG nor selects the reconstruction distribution. Conditions and
defaults are listed in the [LowRank reference](/reference/highorderlowrankop#optional-no-add-check).

## Optional final global recompression

The completed incremental representation is
$\widehat q^{(0)}(S^{(0)})^H$, with $R_0$ columns. When
`global_basis_tol=nothing`, retain it unchanged:
$\widehat q=\widehat q^{(0)}$, $S=S^{(0)}$, and $R=R_0$.
Otherwise, let $\epsilon_g$ be the nonnegative supplied tolerance and form

$$
(\widehat q^{(0)})^H\widehat q^{(0)}
=Z_{\mathrm g}\operatorname{diag}(\mu_1,\ldots,\mu_{R_0})Z_{\mathrm g}^H,
\qquad \mu_1\geq\cdots\geq\mu_{R_0}\geq0.
$$

If $(S^{(0)})^HS^{(0)}=I$, these are the squared singular values of the
completed representation. For non-negligible total energy, retain the
smallest positive rank $R$ satisfying

$$
\frac{\sum_{h=R+1}^{R_0}\mu_h}{\sum_{h=1}^{R_0}\mu_h}
\leq\epsilon_g^2,
\qquad
\widehat q=\widehat q^{(0)}Z_{\mathrm g,R},
\qquad S=S^{(0)}Z_{\mathrm g,R}.
$$

For a nonempty input representation, the implementation keeps at least one
direction. The displayed rotation is used when the rank is reduced; when
all columns are retained, the input factors are returned unchanged without
rotation. Numerically negligible total energy also leaves the factors unchanged. The reported
relative error is additional to local rSVD and incremental sharing error.
It does not recover earlier discarded components or certify image error.
Gram and basis-orthogonality roundoff remain separate numerical limitations.
Recompression occurs before the temporal phase and normalization folding.
For `global_basis_tol >= 1`, the current reported discarded-energy diagnostic
can be inconsistent with the forced minimum rank of one. Do not treat that
diagnostic as a checked error bound in this range; no numerical implementation
change is included here.

## Construction scope {#direct-joint-shared-basis}

<a id="phase-geometry-and-representative-snapshots"></a>
<a id="weighted-snapshot-pod"></a>
<a id="oversampled-coefficient-fitting"></a>
<a id="rank-selection-and-final-audit"></a>

This derivation covers regular local rSVD and incremental sharing. Joint
construction and its algorithm evaluation are outside the reviewed scope.
The former section anchor is retained for existing links; source signatures
list the implemented keywords without making claims about that path.

## Final residual representation

For rSVD, first assemble the unscaled incremental sample-domain factor

$$
\widehat q_i^{(0)}=U_{i,L}\bar C_i^H
\in\mathbb C^{N_t\times R_0},
\qquad
H_i\approx\widehat q_i^{(0)}(S^{(0)})^H.
$$

Stack all dynamic blocks with samples varying fastest. Optional global
recompression produces $\widehat q,S$ and final rank $R$ as defined above;
without it, retain the incremental factors.

The stored temporal matrix additionally folds in:

1. the zeroth-order phase $d_{0,ji}=\exp(\mathrm{i}2\pi k_{0,ji})$;
2. the parity-dependent NFFT centre correction $c_{\mathrm{ctr},ji}$;
3. the symmetric normalization $1/\sqrt{N_v}$.

Thus the actual stored rows are

$$
q_i(j,:)=\frac{d_{0,ji}\,c_{\mathrm{ctr},ji}}{\sqrt{N_v}}
\widehat q_i(j,:).
$$

The first-order Fourier phase stays in the NFFT. Here $q,S$ and $R$ refer
to the factors and rank after optional global recompression, keeping the
forward/adjoint dimensions consistent in either configuration.

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

The complex conjugation of $s_r$ follows from $H_i\approx\widehat q_iS^H$.

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

These expressions agree with the implemented conjugation pattern: the forward operator multiplies the image by the coil sensitivity and the conjugated shared-basis vector before the NFFT, whereas the adjoint applies the conjugated temporal coefficient, the NFFT adjoint, the conjugated coil sensitivity, and the non-conjugated shared-basis vector.

Without sample batching, each shared transform evaluates the concatenated
trajectory. The number of NFFT evaluations per forward or adjoint application is

$$
N_{\mathrm{NFFT}}
=
R N_c.
$$

For independent per-dynamic spatial factorizations, the corresponding transform count would scale approximately as $N_{\mathrm{ro}}LN_c$. This comparison concerns the number of NFFT evaluations and does not imply runtime independence from the number of samples or dynamics.

### Weighted normal and rank execution

Let $W$ contain the raw sample weights repeated identically for each coil.
The data normal is $\sum_c A_c^H W^H W A_c$. For a rank partition
$A_c=\sum_g A_{c,g}$, its application must preserve

$$
A^H W^H W A m
=\sum_c\sum_g A_{c,g}^H W^H W
\left(\sum_h A_{c,h}m\right).
$$

Summing only per-rank normals omits the $g\ne h$ terms. The rank backend
sums partial forward signals on the host and broadcasts each complete signal.
Workers then apply the same squared weights and their local adjoints;
partial images are summed. Channel distribution instead partitions independent
receive coils and sums their complete normal images. These decompositions
preserve the chosen low-rank representation rather than adding a new truncation.

For rank batch size $B$, a forward or adjoint traverses
$\lceil N_tN_{\mathrm{ro}}/B\rceil$ batches. Its aggregate transform count is then
$RN_c\lceil N_tN_{\mathrm{ro}}/B\rceil$, with fewer samples in each transform. A normal application executes both
forward and adjoint passes. Sample batching does not remove sample rows or factor columns and does not bound
complete factor, image-grid, host-master, or solver storage. See
[rank execution and batching](/guide/multi-gpu#rank-distributed-shared-execution)
for constraints, the short-tail backend limitation, and resource management.

## Two-stage approximation error

For the rSVD path, before optional global recompression and before temporal phase/normalization folding, the approximation is

$$
H_i
\longrightarrow
U_{i,L}\widetilde V_i^H
\longrightarrow
U_{i,L}\bar C_i^H (S^{(0)})^H
=
\widehat q_i^{(0)}(S^{(0)})^H.
$$

The triangle inequality gives

$$
\begin{aligned}
\lVert H_i-\widehat q_i^{(0)}(S^{(0)})^H\rVert_F
&\leq
\lVert H_i-U_{i,L}\widetilde V_i^H\rVert_F\\
&\quad+
\lVert U_{i,L}(\widetilde V_i-S^{(0)}\bar C_i)^H\rVert_F.
\end{aligned}
$$

Since $U_{i,L}^HU_{i,L}=I_L$,

$$
\lVert U_{i,L}(\widetilde V_i-S^{(0)}\bar C_i)^H\rVert_F
=
\lVert\widetilde V_i-S^{(0)}\bar C_i\rVert_F,
$$

and therefore

$$
\lVert H_i-\widehat q_i^{(0)}(S^{(0)})^H\rVert_F
\leq
\lVert H_i-U_{i,L}\widetilde V_i^H\rVert_F
+
\lVert\widetilde V_i-S^{(0)}\bar C_i\rVert_F.
$$

The first term is the local rSVD approximation error and the second term is the incremental shared-basis approximation error. `shared_basis_tol` controls only the second stage relative to the retained local energy $\eta_i$; it does not bound the local rSVD truncation error relative to the full matrix $H_i$.

For the stacked original phase matrix $H$, adding final recompression gives

$$
\begin{aligned}
\lVert H-\widehat qS^H\rVert_F
&\leq\lVert H-\widehat q^{(0)}(S^{(0)})^H\rVert_F\\
&\quad+\lVert\widehat q^{(0)}(S^{(0)})^H-\widehat qS^H\rVert_F.
\end{aligned}
$$

The second term is controlled relative to the already shared representation,
not relative to $H$. Neither shared tolerance includes numerical NFFT
interpolation, floating-point operator evaluation, physical input errors, or
solver convergence. A phase-matrix Frobenius tolerance is not an image-error
guarantee.

## Parameter interpretation

For the default rSVD construction:

| Parameter | Stage | Meaning |
|:--|:--|:--|
| `basis_layout` | Representation | Shared spatial columns or independent per-readout factors. |
| `L_rank` | Local rSVD | Fixed retained rank $L$ per dynamic. |
| `rsvd_oversample` | Local rSVD | Additional sketch columns; $L+p\leq\min(N_t,N_v)$. |
| `rsvd_seed` | Local rSVD | Base seed; backend and partition also determine the actual sketch. |
| `rsvd_finalize` | Local rSVD | Direct SVD or projected Gram finalization. |
| `rsvd_backend` | Local products | Chunked products or fused kernels; internal 16-column batches do not cap total rank. |
| `shared_basis_tol` | Incremental merge | Residual tolerance relative to retained local energy $\eta_i$. |
| `shared_rank_max` | Incremental merge | Cap on accumulated rank $R_0$ before optional recompression. |
| `shared_noadd_check` | Incremental merge | Optional guarded residual-energy check; default `false`. |
| `global_basis_tol` | Final recompression | Additional error relative to the completed representation; default `nothing`. |
| `normal_distribution` | Execution | Select single, channel, rank, or independent-readout execution. |
| `normal_sample_batch_size` | Rank execution | Bound sample batches, not total memory; default `12_000_000`. |

The local rank $L$ and final shared rank $R$ describe different approximations. $L$ controls the per-dynamic rSVD truncation, whereas $R_0$ is accumulated by incremental sharing and optional recompression
can reduce it to $R\leq R_0$. The final $R$ may still be smaller than, equal
to, or larger than $L$.

Use the [API reference](/reference/highorderlowrankop) for exact defaults and
supported combinations, and the [multi-GPU guide](/guide/multi-gpu) for runnable
configuration. Approximation and adjoint checks should use a tractable explicit
or directly constructed reference with the same mask and phase conventions.
Validation methodology is described separately in
[Scientific validation strategy](/guide/validation).

## Validation scope

Regression tests assess consistency of phase sign, coordinate convention, NFFT node mapping, centring, normalization, masking, data layout, and adjointness on tractable problems. These tests establish implementation consistency but do not constitute an independent physical reference. Validation of absolute physical accuracy requires independent simulation or measured reference data.
