# Joint adaptive low-rank

Joint adaptive low-rank constructs a common spatial parent directly from
representative phase snapshots across readouts, fits all actual time rows,
and recompresses the completed coefficients. A user-supplied global H tolerance
then drives empirical rank selection and bounded resource expansion. This is
distinct from [Shared low-rank](/theory/low-rank), which first constructs and
incrementally merges local readout factors.

Select `shared_basis_method=:joint_adaptive_low_rank` with `epsilon_H` in
`HighOrderLowRankOp`. The compatible alias is `:adaptive`; the default remains
Shared low-rank under its existing `:rsvd` spelling. The older `:joint` method
has different, sampled per-readout acceptance semantics and is not an alias.

::: info Guarantee level
Projection identities, least-squares stability and SVD recompression have
conditional mathematical guarantees derived below. The current snapshot
policy, resource expansions and `sqrt(mean(loss)+4SE)` acceptance rule are
empirical strategies. A successful construction reports an empirical pass,
with `certification_status=:not_requested`; it is not a full-matrix or
probabilistic certificate, a globally minimal-rank result, or an image-error
guarantee.
:::

## Common target and phase contract

Use the [same residual model](/theory/low-rank#residual-encoding-matrix) and
[notation](/theory/symbols) as Shared low-rank. Let $N=N_tN_{\mathrm{ro}}$
and stack all $H_i$ into $H\in\mathbb C^{N\times N_v}$, with samples varying
fastest within each readout. For stacked time index $u$,

$$
H_{uv}=\exp(\mathrm{i}2\pi\boldsymbol\alpha_u^T\boldsymbol\beta_v),
\qquad
\boldsymbol\alpha_u=(t_u,k_{1u}^{\mathrm{res}},\ldots)^T,
\quad
\boldsymbol\beta_v=(\Delta f_{0,v},b_{1v}^{\mathrm{res}},\ldots)^T.
$$

Both feature vectors are real. The residual rows follow the 2D/3D Fourier
split, with supplied concomitant terms appended when present. Seconds times
Hz and the integrated field/basis
products are in cycles; the forward sign is positive. H excludes the first-order
NFFT, zeroth-order phase, centre correction, CSM, data weights and
$1/\sqrt{N_v}$. Actual ADC times and their phase reference are retained.

The target is

$$
e_H={\|H-\widehat qS^H\|_F\over\sqrt{NN_v}}\leq\epsilon_H,
\qquad 0<\epsilon_H<1,
$$

because every original entry has unit modulus. Here $\widehat q$ denotes
the pure, unscaled coefficients. The factor-level function returns these as
`q`; the constructed operator's `op.q` additionally contains the
[temporal phase and normalization folding](/theory/low-rank#final-residual-representation).
Do not compare `op.q * op.basis'` directly with pure H.
For `epsilon_H=0.01`, the mean squared residual target is $10^{-4}$,
not 1% discarded squared energy.

The ideal common-space problem is

$$
\min_{S^HS=I_R,\,\widehat q}\|H-\widehat qS^H\|_F^2.
$$

For fixed $S$, the optimum is $\widehat q=HS$; optimizing $S$ gives the
leading right singular vectors of the full H. The minimum squared residual
is its singular-value tail. [[5]](/references#ref-5 "Eckart C, Young G. The approximation of one matrix by another of lower rank. Psychometrika. 1936;1:211-218.") [[28]](/references#ref-28 "Mirsky L. Symmetric gauge functions and unitarily invariant norms. Q J Math. 1960;11:50-59.")
The implemented snapshots and gappy fitting approximate this objective without
forming H; they do not perform its exact global SVD. Global error weights each
entry equally. For unequal readout lengths the corresponding denominator would
be $N_v\sum_iN_{t,i}$, rather than an unweighted average of readout percentages;
the current constructor uses rectangular sample/readout arrays.

Low rank is input-dependent even for this analytic phase model. For example,
the square matrix $H_{uv}=\exp(\mathrm{i}2\pi uv/n)$, with
$u,v=0,\ldots,n-1$, has equal singular values and best rank-R relative error
$\sqrt{(n-R)/n}$. Thus smooth phase formulas alone cannot guarantee a small
rank at arbitrary phase extent or a successful finite resource budget.

## Phase geometry and representative snapshots

Let $\overline{\boldsymbol\beta}$ and $C_\beta$ be the mean and covariance
of the spatial features over the complete original mask:

$$
C_\beta={1\over N_v}\sum_v
(\boldsymbol\beta_v-\overline{\boldsymbol\beta})
(\boldsymbol\beta_v-\overline{\boldsymbol\beta})^T.
$$

In exact arithmetic, a covariance factor satisfies
$L_\beta L_\beta^T=C_\beta$. The corresponding features
$\boldsymbol\alpha_u^TL_\beta$ have distance

$$
d_{uw}^2=(\boldsymbol\alpha_u-\boldsymbol\alpha_w)^TC_\beta
(\boldsymbol\alpha_u-\boldsymbol\alpha_w).
$$

The covariance can be singular, so this is a pseudometric: different time
states can have distance zero. It becomes an ordinary Euclidean metric after
identifying states with the same transformed features. The code clips negative
eigenvalues of the computed covariance to zero; the displayed identity and
the following bound use the exact covariance, rather than asserting an
identity for a rounded and clipped matrix.

**Phase-distance bound.** If $h_u$ and $h_w$ are the corresponding rows of H,

$$
\min_{|\gamma|=1}{\|h_u-\gamma h_w\|_2\over\sqrt{N_v}}
\leq2\pi d_{uw}.
$$

Choose $\gamma=\exp[\mathrm{i}2\pi(\boldsymbol\alpha_u-
\boldsymbol\alpha_w)^T\overline{\boldsymbol\beta}]$ and apply
$|\exp(\mathrm{i}x)-1|\leq|x|$ to the centred phase at each voxel.
Squaring and averaging gives the bound. A scalar row phase can be represented
by the temporal coefficients without enlarging the spatial span. This argument
does not authorize shifting production times or removing their row phases.

Farthest-point selection in this feature geometry is related to metric
covering methods. [[27]](/references#ref-27 "Gonzalez TF. Clustering to minimize the maximum intercluster distance. Theor Comput Sci. 1985;38:293-306.")
In ideal arithmetic, farthest-first traversal has covering radius at most twice
the optimal K-centre radius on the finite candidate set. For positive radius,
the K centres and one farthest remaining point are mutually separated by the
achieved radius, while two of these K+1 points must share an optimal ball.
Zero radius already gives exact coverage in the feature geometry. The same
argument holds for the pseudometric or its equivalence classes. This coverage
statement is not a low-rank optimality theorem; numerical stopping and rounded
distances introduce further qualifications.

The implementation forms time candidates every 16 samples plus the last sample
on construction readouts. When there are at least four readouts, it reserves
`max(1,nDyn÷10)` randomly selected readouts from initial space construction.
The candidate rule does not guarantee coverage of all omitted time states.
All actual rows, including reserved readouts, enter the later coefficient fit
and the global evaluation population.

## Weighted snapshot POD

In this derivation, let K be the actual number of selected time rows in $I$;
the requested budget is an upper limit, since repeated features or a smaller
candidate set can yield fewer snapshots. Let $k(u)\in\{1,\ldots,K\}$ index
the centre assigned to candidate row $u$, so its snapshot row is $I_{k(u)}$,
and let $w_k$ be the unnormalized assignment count for centre k.
Construct the spatial snapshot matrix

$$
X=H[I,:]^H\operatorname{diag}(\sqrt{w_1},\ldots,\sqrt{w_K})
\in\mathbb C^{N_v\times K}.
$$

Its leading left singular vectors give the parent $Q_0$ of actual width
$1\leq r_0\leq\min(K,N_v)$ for a successful construction.
This is weighted snapshot POD. [[25]](/references#ref-25 "Sirovich L. Turbulence and the dynamics of coherent structures. I. Coherent structures. Q Appl Math. 1987;45:561-571.")
For $Q_0^HQ_0=I$ and $P_0=Q_0Q_0^H$,

$$
\|X-P_0X\|_F^2=\sum_{h>r_0}\sigma_h(X)^2.
$$

This optimality concerns X, not full H. A direct connection to the candidate
population $\mathcal C$ follows from the phase-distance bound:

$$
\|H_{\mathcal C}(I-P_0)\|_F
\leq2\pi\sqrt{N_v\sum_{u\in\mathcal C}d_{u,I_{k(u)}}^2}
+\sqrt{\sum_{h>r_0}\sigma_h(X)^2}.
$$

For the proof, replace each candidate row by its assigned snapshot with the
row phase used above. The first term bounds the replacement error. Projection
of the replacements has squared residual equal to the weighted snapshot tail;
the triangle inequality completes the result. This derived bound applies to
$\mathcal C$ only and is not computed as a full-H certificate by the code.

The implementation accumulates a small snapshot Gram and orthogonally corrects
the retained spatial basis. The actual $r_0$ depends on the parent cap and
numerically resolved spectrum; unresolved directions are not zero-padded to
reach a requested width. Float32 Gram failures can trigger a Float64 Gram
reaccumulation from the same quantized phase samples. This improves the Gram
calculation but cannot recover information already lost in phase evaluation.

## Oversampled coefficient fitting

For J distinct selected voxels, let $M_J\in\mathbb R^{J\times N_v}$ be
the selection matrix and $A_J=M_JQ_0$, with $1\leq r_0\leq J\leq N_v$.
Assuming $A_J$ has full column rank, the
all-time gappy fit is

$$
\widehat q_0=H M_J^T(A_J^\dagger)^H,\qquad
\widetilde H=\widehat q_0Q_0^H.
$$

It solves the complex least-squares problem on the selected voxels for each
actual time row. Gappy POD supplies this sparse-data coefficient recovery
framework. [[26]](/references#ref-26 "Everson R, Sirovich L. Karhunen-Loève procedure for gappy data. J Opt Soc Am A. 1995;12:1657-1664.")
The fit map is obtained by thin QR, and H[:,J] is evaluated in time chunks;
no full H or explicit inverse is required.

The current selector performs column-pivoted QR on the conjugate transpose
of $Q_0$ restricted to a uniform candidate voxel pool, then adds random unused
pool points. Its relation to Q-DEIM and oversampled gappy POD is methodological;
the full-space or randomized-sampling theorems cannot be transferred without
matching their assumptions. [[16]](/references#ref-16 "Drmač Z, Gugercin S. A new selection operator for the discrete empirical interpolation method. SIAM J Sci Comput. 2016;38:A631-A648.") [[17]](/references#ref-17 "Peherstorfer B, Drmač Z, Gugercin S. Stability of discrete empirical interpolation and gappy proper orthogonal decomposition. SIAM J Sci Comput. 2020;42:A2837-A2864.")

**Conditional gappy bound.** For orthonormal $Q_0$ and full-rank $A_J$,

$$
\|H-\widetilde H\|_F
\leq\gamma_J\|H(I-P_0)\|_F,
\qquad \gamma_J=\|A_J^\dagger\|_2.
$$

For a column $x$ of $H^H$, set $\mathcal P=Q_0A_J^\dagger M_J$.
Then $\mathcal P Q_0=Q_0$ and
$(I-\mathcal P)x=(I-\mathcal P)(I-P_0)x$.
For a nontrivial projection, $\|I-\mathcal P\|_2=\|\mathcal P\|_2$;
selection with distinct rows and orthonormal $Q_0$ gives
$\|\mathcal P\|_2=\|A_J^\dagger\|_2$.
Applying the vector inequality to all columns proves the Frobenius result.
The identity-projection case has zero residual.

This bound separates missing spatial information from recovery amplification.
The condition number $\kappa(A_J)$ alone is not $\gamma_J$: $A_J=cI$
has condition number one for arbitrarily small $|c|$, but inverse norm
$1/|c|$. The implemented condition limit is a numerical guard, not an
independent approximation certificate.

### Distinguishing space and fit errors

For any full-rank parent, including a slightly nonorthogonal numerical one,
define the full-domain oracle coefficients

$$
C_*=H Q_0(Q_0^HQ_0)^{-1},\qquad E=H-C_*Q_0^H.
$$

Since $EQ_0=0$,

$$
\|H-\widehat q_0Q_0^H\|_F^2
=\|E\|_F^2+\|(\widehat q_0-C_*)Q_0^H\|_F^2.
$$

This identity is rigorous; the implementation evaluates it on a small set of
time rows over all masked voxels using a stable small solve. It is not a
complete-time certificate. A large first term suggests insufficient space;
a large second term suggests poor coefficient recovery. The thresholds and
resource increments used to act on that evidence remain empirical choices.

## Full-time recompression and total error

For an orthonormal parent, take a stable QR/small SVD of all-time
$\widehat q_0=U_q\Sigma_qV_q^H$. Set

$$
\widehat q_R=\widehat q_0V_{q,R},\qquad S_R=Q_0V_{q,R}.
$$

The implemented blocked QR merges triangular factors and takes the SVD of
the resulting small matrix; it does not estimate weak tails by subtracting
leading energy from a Gram total. TSQR provides the factorization background.
[[22]](/references#ref-22 "Demmel J, Grigori L, Hoemmen M, Langou J. Communication-optimal parallel and sequential QR and LU factorizations. SIAM J Sci Comput. 2012;34:A206-A239.")

The nonzero singular values of $\widetilde H$ equal those of
$\widehat q_0$, so

$$
\|\widetilde H-\widehat q_RS_R^H\|_F^2
=\sum_{h>R}\sigma_h(\widehat q_0)^2.
$$

This is the optimal rank-R approximation of $\widetilde H$, not necessarily
of H. With $e_0=\|H-\widetilde H\|_F/\sqrt{NN_v}$, a conservative total
bound is

$$
e_H(R)\leq e_0+
\sqrt{\frac{\sum_{h>R}\sigma_h(\widehat q_0)^2}{NN_v}}
\leq\gamma_J\frac{\|H(I-P_0)\|_F}{\sqrt{NN_v}}
+\sqrt{\frac{\sum_{h>R}\sigma_h(\widehat q_0)^2}{NN_v}}.
$$

The first term is not measured exactly at full size, so the spectral tail
alone cannot accept the final factors. A related conditional comparison with
the best rank-R approximation $H_R^*$ is

$$
\|H-\widehat q_RS_R^H\|_F
\leq\|H-H_R^*\|_F+2\|H-\widetilde H\|_F.
$$

Indeed the best rank-R truncation of $\widetilde H$ cannot have larger error
to $\widetilde H$ than $H_R^*$; applying the triangle inequality twice gives
the claim. It is useful when the base approximation is already accurate, and
does not establish a globally minimal selected rank.

### Nonorthogonal factors and finite precision

For an exact spectral interpretation of existing full-column-rank factors,
take a thin QR factorization $S=QT$ with $Q^HQ=I$ and replace their
coefficients by $\widehat qT^H$.
The prediction remains
$\widehat qS^H=(\widehat qT^H)Q^H$, before spectral recompression. This
algebra applies equally to local-rSVD/shared factors and joint factors; a fair
comparison gives both the same final treatment. This is a change of basis
one can apply to either representation; the legacy optional shared
recompression does not perform this QR correction.

The current joint parent is numerically orthogonally corrected, not exactly
orthogonal. For its actual factors the additional tail bound has multiplier
$\|Q_0\|_2\leq\sqrt{1+\eta}$ if
$\|Q_0^HQ_0-I\|_2\leq\eta$. Phase evaluation, fitting and output rotations
introduce further numerical defects. Mathematical tail identities therefore
do not certify the saved Float32 factors; the evaluator uses the actual
`Complex{T}` outputs against independently evaluated phase entries.

### Why the original-H error need not decrease at every prefix

For orthonormal $Q_0$, write $C=HQ_0$, $\Delta=\widehat q_0-C$ and
$\Pi_R=V_{q,R}V_{q,R}^H$. Orthogonal row-space components give

$$
\|H-\widehat q_0\Pi_RQ_0^H\|_F^2
=\|H(I-P_0)\|_F^2+\|C(I-\Pi_R)\|_F^2
+\|\Delta\Pi_R\|_F^2.
$$

The second term decreases with nested R, but the retained fitting-error term
can increase. The gappy fit is not the full projection $C$, so base error and
coefficient-tail error cannot generally be added in quadrature. The code
checks ordered prefixes rather than assuming a monotone error and bisecting.

## Global evaluation and rank selection {#rank-selection-and-final-audit}

For fixed factors and R, independently draw row/voxel pairs uniformly with
replacement over the complete H and define

$$
z_a=|H_{u_av_a}-(\widehat q_RS_R^H)_{u_av_a}|^2,
\qquad e_H^2=\mathbb E[z_a],\qquad
\widehat e_H^2={1\over m}\sum_{a=1}^mz_a.
$$

The sample mean is unbiased for squared global error before adaptive selection;
its square root is not exactly unbiased. Repeated draws count as repeated
observations. Construction/evaluation coincidences are allowed and should be
recorded; removing training entries changes the target population. A Cartesian
submatrix is not a collection of independent entry draws. Readout, ROI and
tail errors are separate diagnostics, not additional epsilon_H requirements.

The current builder evaluates all prefixes with two internal sample sets,
by default 100,000 selection and 400,000 acceptance pairs. Their default
random streams use separate seeds; user overrides can make the seeds equal,
so independence is not enforced for every configuration. It chooses the first
prefix passing `sqrt(mean(loss)+4SE) < epsilon_H` in both sets, where SE is
the estimated standard error of the mean squared loss. Both sets therefore
participate in R selection: the second is not an untouched final holdout.
The 4SE margin has no general finite-sample coverage theorem and does not
account for all adaptive queries.

For example, a population of one million entries with one squared loss of
1000 has $e_H=\sqrt{0.001}$, approximately 3.16%. A 400,000-pair sample misses
that entry with probability $(1-10^{-6})^{400000}$, approximately 67.0%; then
its mean and SE are both zero. This artificial example is a limitation of
the general heuristic, not an observed failure of a specific saved factor.

For a chosen rule and factor, freeze both before a new external evaluation.
Record factor/input hashes, sampling scope, overlap, counts and uncertainty.
A new empirical audit does not by itself become a deterministic full-matrix
or high-probability certificate. The constructor leaves
`external_audit_status=:not_requested` for this caller-owned step.

### What a probabilistic certificate would additionally require

For a frozen factor, a fixed sample count $m\geq2$, $0<\delta<1$, and iid
losses in $[0,B_z]$ from fresh uniform entry draws independent of that factor,
the empirical Bernstein theorem gives, with probability at least $1-\delta$,

$$
e_H^2\leq\overline z+
\sqrt{\frac{2s_z^2\log(2/\delta)}{m}}
+\frac{7B_z\log(2/\delta)}{3(m-1)},
$$

where $s_z^2$ is the unbiased sample variance. [[29]](/references#ref-29 "Maurer A, Pontil M. Empirical Bernstein bounds and sample variance penalization. COLT. 2009. Theorem 4.")
A valid but potentially loose loss bound is
$B_z=(1+\max_u\|\widehat q_{R,u,:}\|_2\max_v\|S_{R,v,:}\|_2)^2$.
Here $\delta$ is a statistical failure probability, distinct from the accuracy
target $\epsilon_H$. For a finite collection of M candidates frozen before
sampling, allocating $\delta/M$ to each yields a simultaneous bound even when
they share evaluation draws. Candidates constructed using earlier evaluation
results instead need a suitable uniform theorem, or fresh conditionally
independent samples with a valid summable failure-probability budget at each
query. Simply dividing by the observed number of adaptive queries does not
repair data reuse. Optional stopping in the sample count also requires its
own control. Numerical certification additionally needs a rounding budget for
the loss, variance and bound computations.

The current implementation does not apply this certificate. A loose bound
can be inconclusive even when empirical error is small, and does not justify
automatically increasing R.

## Resource adaptation and failure semantics

The initial budgets are at most 128 snapshots, 512 fit points and parent cap 32;
they are resource starting points, not expected answers. Actual parent width
$r_0$ is read from the constructed space. Failed checks can expand J, the
parent cap or K, subject to `adaptive_options` and `shared_rank_max`.

The current diagnostic uses selected time rows across all voxels. If spatial
projection error is at most `0.9 * epsilon_H`, or fitting error dominates, the
policy first expands fitting. Otherwise it increases the parent when resolved
snapshot directions remain beyond the cap, or expands snapshots. J/K growth,
parent increments, condition limits and the diagnostic thresholds are engineering
rules; they do not prove optimal resource allocation, monotone improvement or
success for every input.

On exhaustion the factor-level function returns `q=S=nothing`, rank zero and
a failure reason. The operator constructor records the report and throws;
it does not silently install partial factors, relax epsilon_H, or change the
NFFT backend. Time and memory budgets have the limitations stated in the
[generated API](/reference/highorderlowrankop#api).

## Guarantees and empirical strategies

| Component | What is established | What is not established |
|:--|:--|:--|
| Phase covariance distance | Derived upper bound on phase-row distance up to scalar phase. | Complete-time coverage from the subsampled candidate rule. |
| Weighted snapshot POD | Optimal truncation of the weighted snapshot matrix in exact arithmetic. | Globally optimal parent for full H. |
| Gappy fit | Complex least squares and a conditional error-amplification bound. | Accuracy from condition number or J alone. |
| Full-time recompression | Optimal rank-R truncation of the fitted approximation for an orthonormal parent. | Original-H accuracy from the coefficient tail alone. |
| Global paired sampling | Unbiased squared-error estimator for fixed factors and independent uniform draws. | General finite-sample coverage of the current 4SE rule after selection. |
| Adaptive K/J/cap | Bounded error-diagnostic-driven policy with explicit failure. | Globally minimal R, optimal budgets or guaranteed success. |
| H tolerance | An explicit target relative to the frozen input phase model. | Correct B0, physical truth, worst-ROI accuracy or the same image-error percentage. |

The [shared theory page](/theory/low-rank#from-h-error-to-encoding-and-reconstruction-error)
derives the weighted encoding-error relation and an illustrative conditioned
Tikhonov perturbation bound. These apply to either factor construction;
finite CG and NFFT errors remain separate.

## Method contribution and related work

Separable approximation of MRI field modulation predates either path;
time segmentation and higher-order singular-vector separation are relevant
foundations. [[30]](/references#ref-30 "Sutton BP, Noll DC, Fessler JA. Fast, iterative image reconstruction for MRI in the presence of field inhomogeneities. IEEE Trans Med Imaging. 2003;22:178-188.") [[2]](/references#ref-2 "Wilm BJ, Barmet C, Pruessmann KP. Fast higher-order MR image reconstruction using singular-vector separation. IEEE Trans Med Imaging. 2012;31:1396-1403; with erratum.")
Snapshot POD, gappy recovery, QR selection and SVD are established numerical
components. Hierarchical POD and MRI group-patch compression provide additional
shared-space context, with different algorithms and assumptions. [[23]](/references#ref-23 "Himpe C, Leibner T, Rave S. Hierarchical approximate proper orthogonal decomposition. SIAM J Sci Comput. 2018;40:A3267-A3292.") [[20]](/references#ref-20 "Tian R, Scheffler K. Group-Patch Joint Compression. Magn Reson Med. 2026.")
Here the approximated matrix excludes receive sensitivities and all actual
time rows use one final spatial basis; joint B0/RF compression of grouped
encoding matrices addresses a different compression target.

The method-specific design combines:

1. a phase-aware representative space shared across actual readouts;
2. one oversampled coefficient map applied to all actual time rows;
3. full-time spectral recompression followed by original-H global residual
   evaluation, rather than accepting a snapshot or coefficient tail alone;
4. bounded K/J/parent expansion guided by separate space and fit diagnostics.

These identify the implementation's algorithmic contribution and its distinction
from local-rSVD sharing. They do not establish a first-in-literature claim for
each ingredient or for all joint MRI compression. The phase-distance and error
decompositions here are model-specific derivations, not claims of new general
linear-algebra theorems. Comparative novelty requires method-level comparison
with the cited literature; speed and memory advantages require measured
matched-accuracy evidence. The global empirical criterion does not favor one
front end in advance: local-rSVD factors can receive the same recompression
and original-H evaluation.

## Interface and interpretation

Given the matching grid, field model, mask and coil maps, select the opt-in path:

```julia
op = HighOrderLowRankOp(grid, kspha, times;
    fieldmap, csm, mask,
    shared_basis_method=:joint_adaptive_low_rank,
    epsilon_H=0.01,
)
```

The experimental factor-level entry
`HighOrderMRI.joint_adaptive_spatial_basis` (compatible alias
`HighOrderMRI.adaptive_spatial_basis`) creates no NFFT plans and returns host
matrices even with GPU construction. `epsilon_H` is a scalar global H target;
it does not automatically select local rSVD rank or prove an optimal sketch
budget. `joint_basis_report` receives construction, empirical, external-audit
and certification states without the large factor arrays. Its empirical pass
should be reported with the evaluation design and actual factor precision.
