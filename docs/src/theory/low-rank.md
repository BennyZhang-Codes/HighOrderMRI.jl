# Low-rank shared subspace

`HighOrderLowRankOp` accelerates repeated applications of the expanded
encoding model with two compression stages by default (`basis_layout=:shared`):

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

## Independent per-readout baseline

Set `basis_layout=:per_readout` to retain each local pair
``U_d,\widetilde V_d`` directly, bypassing the shared-basis construction:

```julia
op = HighOrderLowRankOp(grid, kspha, times;
    fieldmap, csm, mask,
    basis_layout=:per_readout,
    L_rank=6,
    rsvd_oversample=5,
)
```

For each readout and coil, the forward operation is

```math
A_{d,c}m \approx \sum_{\ell=1}^{L}
\operatorname{diag}(u_{d,\ell})\mathcal F_d
\left(m\odot C_c\odot\widetilde v_{d,\ell}^{*}\right).
```

The same zeroth-order phase, centre correction, and normalization are folded
into the temporal factors. One NFFT plan is cached per readout, giving
``N_d L N_c`` transforms per forward or adjoint evaluation. The spatial
factor bank has ``N_v N_d L`` complex entries; cached plans also consume
memory. This baseline can therefore require substantially more memory than
the shared layout.

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
evaluations on the primary GPU. Once the multi-GPU normal backend
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
occupies ``N_v N_d L`` complex entries in host RAM. The default
`per_readout_storage=:gpu` retains all independent factors and plans on the GPU.

Voxel-distributed per-readout setup uses `per_readout_gather=:pipeline` by
default. Each GPU has two pinned host buffers and a separate collector worker.
Downloads and host writes can overlap later rSVD phases. A CUDA event delays
the next overwrite of the spatial factor until its download finishes; a host
buffer is reused only after its final write completes. The queue is bounded
to two outstanding readouts. `per_readout_gather=:synchronous` retains the
blocking gather for comparisons. The independent factors and their precision
are unchanged, and the full host factor bank is still required.

For five GPUs, twelve default Julia threads give the five compute workers,
five collectors and coordinator room to run. With fewer threads the pipeline
remains bounded, but overlap can be reduced. Pipeline worker times overlap
the rSVD phases and must not be added to the total wall time.

When the independent spatial bank exceeds host RAM, use
`per_readout_storage=:regenerate` with `rsvd_distribution=:voxel`,
`rsvd_backend=:kernel` and `rsvd_finalize=:gram`. This retains each readout's
original unmodulated ``Q_d`` and Gram rotation ``Z_d`` and generates

```math
V_d\Sigma_d = (H_d^H Q_d) Z_d
```

with the existing matrix-free adjoint kernel. The complete phase matrix and
spatial bank are never allocated. No additional singular-value truncation is
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
excludes Q/Z, field maps, workspaces and pinned staging. For example,
`per_readout_host_cache_bytes=64 * 1024^3` reserves at most 64 GiB of host
spatial factors. Start with zero or a small budget when measuring regeneration.
The source operator and each normal worker own bounded GPU slots and a
generator workspace; `close(op)` releases these workers/caches, which are
recreated lazily if the operator is subsequently reused.

For diagnostics, `HighOrderMRI.regenerated_basis_stats(op.basis)` reports
logical bank bytes, Q/Z bytes, model bytes and resident host-factor bytes.
`HighOrderMRI.per_readout_cache_stats(cache)` additionally records generation
count/device time, host-cache hits and consumer ready-event wait time. Device
times on different streams overlap and must not be summed as wall time.

Both layouts support the existing local rSVD backends, voxel-distributed
setup, and channel-distributed normal operator. For a controlled comparison,
hold the data, mask, field model, local rank, oversampling, seed, precision,
backend, GPU partition, weights, and solver settings fixed. Only change
`basis_layout`. `shared_rank_max` and `shared_basis_tol` are ignored in the
per-readout layout; `global_basis_tol` must be `nothing` and
`shared_basis_method` must be `:rsvd`. Setting `shared_basis_tol=0` in the
shared layout still constructs a shared basis and is not this baseline.

The independent factorization and per-readout NUFFT sums follow the
[MAXGIRF author's encoding implementation](https://github.com/usc-mrel/lowfield_maxgirf/blob/637719ce1d96e26323803c0abe7607e8d598d3c0/src/encoding_lowrank_maxgirf.m).
This switch retains HighOrderMRI's spherical-harmonic phase model and local
rSVD implementation. It does not add the author's concomitant-field model
or promise bitwise reproduction of the MATLAB reconstruction.

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
| `basis_layout` | `:shared` or independent `:per_readout` factors | Change only this setting for the shared-compression comparison |
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
