# Reconstruction workflow

A reproducible reconstruction has six stages:

1. prepare synchronized field coefficients and sampling times;
2. estimate receive-noise covariance when a noise acquisition is available,
   then optionally apply noise-whitened coil compression;
3. construct the physical grid and spatial inputs;
4. select and validate an encoding operator;
5. create density weights and solver settings;
6. reconstruct and report raw complex accuracy, convergence, timing, and
   provenance.

## Prepare data

For a low-rank dynamic operator, use:

```text
kspha  :: (nTerm, nSam, nDyn)
times  :: (nSam, nDyn)
data   :: (nSam, nDyn, nCha)
weight :: (nSam, nDyn)
```

Create the 2D matrix accepted by [`recon_HOOp`](@ref) without changing the
vector order:

```julia
data_matrix = reshape(data, nSam * nDyn, nCha)
weight_vector = vec(weight)
```

Apply every axis exchange, reversal, and circular shift consistently to the
image, mask, field map, and coil maps before constructing the operator.

## Coil compression

Coil compression is an operator-independent preprocessing step. The preferred
path estimates receive-noise covariance first, then fits one noise-whitened
transform and applies it to both acquired data and coil-sensitivity maps:

```julia
noise_covariance = estimate_noise_covariance(
    noise_data;  # (nNoiseSample, nCha)
    coil_dim=2,
)

# Needed only when the noise scan and MRI acquisition differ. Dwell times must
# use the same units; the receiver ratio corrects effective noise bandwidth.
prewhitening_scale = noise_prewhitening_scale_factor(
    acquisition_dwell_time,
    noise_dwell_time;
    receiver_bandwidth_ratio=noise_receiver_bandwidth_ratio,
)

data_cc, csm_cc, coil_transform = compress_coils(
    data_matrix,  # (nSampleTotal, nCha)
    csm;         # (..., nCha)
    data_coil_dim=2,
    csm_coil_dim=ndims(csm),
    n_virtual_coils=10,
    noise_covariance,
    prewhitening_scale_factor=prewhitening_scale,
)
```

Use `data_cc` in [`recon_HOOp`](@ref) and `csm_cc` when constructing
`HighOrderOp`, `HighOrderOp_Kernel`, or `HighOrderLowRankOp`. The operators
need no compression-specific option: their channel count is inferred from the
compressed CSM.

### Algorithm

Let ``D\in\mathbb C^{N_o\times N_c}`` be the calibration observations,
``\Psi_{\mathrm{noise}}\in\mathbb C^{N_c\times N_c}`` the positive-definite
noise-prescan covariance, and

```math
s = \frac{T_{\mathrm{acq}}}{T_{\mathrm{noise}}}R_{\mathrm{BW}}.
```

The effective covariance for the MRI acquisition is
``\Psi_{\mathrm{acq}}=\Psi_{\mathrm{noise}}/s``. A Cholesky-derived right
whitening matrix ``W`` is chosen so that

```math
W^H\Psi_{\mathrm{acq}}W=I.
```

The implementation computes

```math
DW=U\Sigma V^H,
\qquad
C=WV_r,
```

where ``V_r`` contains the first ``r`` right singular vectors. The same
combined matrix ``C`` is then used for

```math
D_{\mathrm{cc}}=DC,
\qquad
S_{\mathrm{cc}}=SC,
\qquad
C^H\Psi_{\mathrm{acq}} C=I,
```

where ``S`` denotes the CSM reshaped with coils in columns. This is truncated
SVD of the noise-whitened data and is optimal in the Frobenius norm among
rank-``r`` global linear coil subspaces. The implementation follows software
channel compression by [Huang et
al.](https://doi.org/10.1016/j.mri.2007.04.010), the broader array-compression
framework of [Buehrer et al.](https://doi.org/10.1002/mrm.21237), and
receive-noise conditioning described by [Kellman and
McVeigh](https://doi.org/10.1002/mrm.20713). The rank-``r`` optimality is the
Eckart--Young result and applies to this stated global, whitened Frobenius
objective.

When no noise acquisition is available, omit `noise_covariance`. This assumes
identity covariance and reduces to conventional global PCA/SVD compression;
it must not be described as noise-optimal when channels are correlated.

To estimate the transform from only a calibration region and then apply it to
the complete acquisition:

```julia
coil_transform = fit_coil_compression(
    @view(data_matrix[calibration_samples, :]);
    coil_dim=2,
    energy_threshold=0.99,
    noise_covariance,
)
data_cc = apply_coil_compression(data_matrix, coil_transform; coil_dim=2)
csm_cc = apply_coil_compression(csm, coil_transform; coil_dim=ndims(csm))
```

Specify exactly one of `n_virtual_coils` and `energy_threshold`. An energy
threshold measures variance retained in the calibration data; it is not an
image-accuracy guarantee. Freeze the fitted transform across all operators
and reconstruction conditions being compared, and report its rank and
`retained_energy`. With `noise_covariance`, the reported energy is measured
after whitening.

[`estimate_noise_covariance`](@ref) expects noise-only data acquired with the
same receiver gains and channel ordering. When dwell times or effective
receiver bandwidths differ, use [`noise_prewhitening_scale_factor`](@ref).
The convention follows ISMRMRD/mrpro:
``s=(T_acq_dwell/T_noise_dwell)*NoiseReceiverBandwidthRatio``. If they are
identical, use the default `prewhitening_scale_factor=1`. The covariance must
be positive definite; otherwise acquire more noise observations or explicitly
regularize the covariance before fitting. Density compensation is unrelated
to noise conditioning. The compression functions perform the SVD on the CPU
and return CPU arrays; CUDA operators transfer the prepared CSM to their own
backend during construction.

Geometric-decomposition coil compression (GCC) can outperform one global
matrix for Cartesian acquisitions with nonsubsampled dimensions because it
uses spatially varying, aligned compression matrices ([Zhang et
al.](https://doi.org/10.1002/mrm.24267)). That method requires a different
hybrid-space data path and is not interchangeable with the single matrix
accepted by arbitrary-trajectory HighOrderMRI operators. It is therefore not
silently used here. For spiral, arbitrary non-Cartesian, or cross-operator
comparisons, noise-whitened global SVD is the strongest method currently
implemented with exact shared data/CSM consistency.

## Density weights

For a 2D first-order trajectory:

```julia
weight_vector = samplingDensity(
    reshape(kspha[2:3, :, :], 2, :),
    (grid.nX, grid.nY),
)
```

The helper returns square-root density compensation weights compatible with
the weighting operator used by `recon_HOOp`. Reuse the exact same weights
when comparing encoding implementations.

## Solver configuration

The reconstruction wrapper uses MRIReco.jl and
RegularizedLeastSquares.jl:

```julia
using RegularizedLeastSquares

rec_params = Dict{Symbol,Any}(
    :reconSize => (grid.nX, grid.nY),
    :reg => L2Regularization(1f-9),
    :iterations => 20,
    :solver => CGNR,
)
```

For 3D, set `:reconSize => grid.matrixSize`. Keep the solver, regularizer,
initialization, iteration count, and stopping condition identical across a
comparison.

## Reconstruct

```julia
image = recon_HOOp(
    op,
    ComplexF32.(data_matrix),
    ComplexF32.(weight_vector),
    rec_params,
)
```

For a channel-distributed `HighOrderLowRankOp`, the multi-GPU normal backend
is created lazily. `recon_HOOp` releases it in a `finally` block by default.
Use `release_backend=false` only when the same operator and normal backend
will immediately be reused, then call `close(op)` after the final solve.

## Accuracy metrics

Raw complex NRMSE is the primary comparison:

```julia
complex_error = raw_complex_nrmse(image, reference)
mag_error = magnitude_nrmse(image, reference)
ssim = magnitude_ssim(image, reference)
```

`aligned_complex_nrmse` and `magnitude_nrmse(...; align=true)` remove a global
least-squares complex scale. They are secondary metrics and must be labelled
as aligned.

For a low-rank study, also record:

- `L_rank`, final shared rank, `shared_basis_tol`, and `shared_rank_max`;
- rSVD backend, distribution, finalization, oversampling, and seed;
- forward, adjoint, normal, and adjoint-identity errors;
- setup, one forward/adjoint/normal evaluation, solver, and end-to-end time;
- fixed-iteration and common-stopping-condition results;
- peak host and device memory;
- Git SHA, dirty state, Julia/CUDA/package versions, and GPU UUIDs.

The [reconstruction protocol](reconstruction-protocol.md) gives the full
frozen comparison rules.

## Timing CUDA work

Compilation and asynchronous execution can make a single wall-clock timing
misleading. Warm the exact path first and synchronize every participating GPU
around measured regions. Use repeated steady-state measurements and report a
distribution such as median and IQR.

Do not use post-reconstruction intensity fitting to hide a sign, centring,
normalization, or data-layout mismatch.
