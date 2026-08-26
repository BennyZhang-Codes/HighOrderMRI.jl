# Reconstruction workflow

A controlled reconstruction should fix the data layout, field convention, spatial geometry, encoding operator, density weighting, and solver settings before numerical comparisons are performed. The workflow can be organized into six stages:

1. prepare synchronized field coefficients and sampling times;
2. optionally apply a fixed coil-compression transform;
3. construct the physical grid and spatial inputs;
4. construct and verify the encoding operator;
5. define density weights and solver settings;
6. reconstruct and report accuracy, convergence, timing, and provenance.

Receive-noise whitening and coil-compression theory are described separately in [Coil compression](/guide/coil-compression).

## Data layout

For a dynamic low-rank operator, the principal array dimensions are

```julia
size(kspha) == (nTerm, nSam, nDyn)
size(times) == (nSam, nDyn)
size(data) == (nSam, nDyn, nCha)
size(weight) == (nSam, nDyn)
```

The two-dimensional data matrix accepted by `recon_HOOp` is obtained without changing the vectorization order:

```julia
data_matrix = reshape(data, nSam * nDyn, nCha)
weight_vector = vec(weight)
```

Samples vary fastest, followed by dynamics and then receive channels. The same convention is summarized in [Symbols and notation](/theory/symbols#implementation-array-names).

Any axis permutation, reversal, circular shift, or crop should be applied consistently to the image-domain quantities that share the same physical coordinates, including the mask, field map, and coil-sensitivity maps.

## Dynamic field coefficients

Measured or predicted field coefficients must be synchronized to the ADC sample times and converted to the accumulated-phase convention used by the encoding model.

Before constructing the operator, verify that:

- `kspha` contains time-integrated field coefficients;
- phase is expressed in cycles rather than radians;
- the spatial coefficient units are consistent with the physical coordinates of `Grid`;
- the solid-harmonic row order matches the encoding model;
- zeroth-order phase is included exactly once;
- trajectory, synchronization, and density-compensation calculations use the same timing convention.

GIRF-based prediction, interpolation, and model-based temporal synchronization are described in [Field preprocessing and synchronization](/guide/field-preprocessing).

## Optional coil compression

Coil compression should be applied before construction of the final encoding operator. A single transform is fitted and the same right-side matrix is applied to both acquired data and CSM:

```julia
data_cc, csm_cc, coil_transform = compress_coils(
    data_matrix,
    csm;
    data_coil_dim=2,
    csm_coil_dim=ndims(csm),
    n_virtual_coils=10,
    noise_covariance,
)
```

For a controlled comparison, the resulting `data_cc`, `csm_cc`, and fitted transform should be held fixed across all encoding implementations. Separate transforms should not be fitted for different operators.

When representative noise-only samples are available, noise-whitened global SVD compression can be used. The noise covariance model, dwell-time/bandwidth correction, calibration fitting, and distinction from geometric coil compression are described in [Coil compression](/guide/coil-compression).

## Encoding operator

Select the numerical implementation according to the calculation being performed:

- `HighOrderOp` for array-based explicit evaluation and derivative products;
- `HighOrderKernelOp` for fused explicit CUDA evaluation;
- `HighOrderLowRankOp` for repeated large-scale forward and adjoint evaluations using the residual-phase approximation.

Constructor details and implementation differences are summarized in [Encoding operators](/guide/operators). The common physical and matrix formulation is given in [Expanded encoding model](/theory/encoding-model).

For an operator comparison, first establish consistency between the explicit implementations and then quantify the low-rank approximation against a fixed explicit reference. Independent numerical or experimental validation of the physical signal model is a separate step; see [Scientific validation strategy](/guide/validation).

## Density weighting

For a two-dimensional first-order trajectory:

```julia
weight_vector = samplingDensity(
    reshape(kspha[2:3, :, :], 2, :),
    (grid.nX, grid.nY),
)
```

The helper returns square-root density compensation weights compatible with the weighting convention used by `recon_HOOp`.

Let $W$ denote the corresponding diagonal weighting operator. The weighted least-squares normal term is

$$
A^H W^H W A.
$$

The same weights should be used when comparing encoding implementations. Changing the density weights changes the reconstruction objective and should therefore be treated as a separate experimental condition.

## Solver configuration

The reconstruction wrapper uses MRIReco.jl and RegularizedLeastSquares.jl:

```julia
using RegularizedLeastSquares

rec_params = Dict{Symbol,Any}(
    :reconSize => (grid.nX, grid.nY),
    :reg => L2Regularization(1f-9),
    :iterations => 20,
    :solver => CGNR,
)
```

For a three-dimensional reconstruction, set

```julia
:reconSize => grid.matrixSize
```

Within a controlled comparison, keep the solver, regularizer, regularization strength, initialization, numerical precision, iteration count, and stopping criterion unchanged.

## Reconstruction

```julia
image = recon_HOOp(
    op,
    ComplexF32.(data_matrix),
    ComplexF32.(weight_vector),
    rec_params,
)
```

For a channel-distributed `HighOrderLowRankOp`, the multi-GPU normal backend is constructed lazily. By default, `recon_HOOp` releases this backend in a `finally` block after the solve. Set `release_backend=false` only when the same operator and backend will be reused immediately, and call `close(op)` after the final solve.

## Accuracy and convergence

When a complex reference is available, raw complex NRMSE provides a direct image-domain comparison without post-hoc phase or scale alignment:

```julia
complex_error = raw_complex_nrmse(image, reference)
mag_error = magnitude_nrmse(image, reference)
ssim = magnitude_ssim(image, reference)
```

`aligned_complex_nrmse` and `magnitude_nrmse(...; align=true)` remove a global least-squares complex scale before evaluation and should be identified explicitly as aligned metrics.

For low-rank studies, image-domain measures should be accompanied by operator-level quantities:

- forward relative error;
- adjoint relative error;
- weighted normal-operator relative error;
- adjoint-identity error;
- solver residual or data-consistency residual.

The evidence hierarchy and the explicit, direct-global, and incremental comparison baselines are defined in [Scientific validation strategy](/guide/validation).

## Timing and provenance

Compilation and asynchronous CUDA execution can bias isolated wall-clock measurements. Warm the exact execution path before timing, synchronize every participating GPU around the measured region, and use repeated steady-state measurements. A robust summary such as the median and interquartile range is preferable to a single run.

For a low-rank reconstruction study, record at least:

- `L_rank`, final shared rank, `shared_basis_tol`, and `shared_rank_max`;
- rSVD backend, distribution, finalization, oversampling, and random seed;
- setup, forward, adjoint, normal-operator, solver, and end-to-end timing;
- peak host and device memory;
- Git SHA and dirty-worktree state;
- Julia, CUDA, package, NFFT-backend, and GPU UUID information.

The [reconstruction protocol](/guide/reconstruction-protocol) specifies the complete fixed comparison rules. Post-reconstruction intensity alignment should not be used to compensate for a phase-sign, centring, normalization, or data-layout mismatch.
