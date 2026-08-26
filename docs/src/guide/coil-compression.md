# Coil compression

Coil compression is applied as an operator-independent preprocessing step. HighOrderMRI uses one right-side linear transform for both acquired data and coil-sensitivity maps so that the compressed data remain consistent with the same encoding model.

The current implementation performs global SVD/PCA coil compression, optionally after receive-noise whitening. This corresponds to the global software/array-compression family described previously. [[12]](/references#ref-12 "Huang F, Vijayakumar S, Li Y, Hertel S, Duensing GR. A software channel compression technique for faster reconstruction with many channels. Magn Reson Imaging. 2008;26:133-141.") [[13]](/references#ref-13 "Buehrer M, Pruessmann KP, Boesiger P, Kozerke S. Array compression for MRI with large coil arrays. Magn Reson Med. 2007;57:1131-1139.") The implementation does not perform spatially varying geometric-decomposition coil compression (GCC).

## Workflow

When representative noise-only data are available, estimate the receive-noise covariance with [`estimate_noise_covariance`](/reference/estimate-noise-covariance):

```julia
noise_covariance = estimate_noise_covariance(
    noise_data;
    coil_dim=2,
)
```

If the noise scan and MRI acquisition use different dwell times or effective receiver bandwidths, compute the corresponding scale factor with [`noise_prewhitening_scale_factor`](/reference/noise-prewhitening-scale):

```julia
prewhitening_scale = noise_prewhitening_scale_factor(
    acquisition_dwell_time,
    noise_dwell_time;
    receiver_bandwidth_ratio=noise_receiver_bandwidth_ratio,
)
```

One transform can then be fitted and applied to both acquired data and coil-sensitivity maps using [`compress_coils`](/reference/compress-coils):

```julia
data_cc, csm_cc, coil_transform = compress_coils(
    data_matrix,
    csm;
    data_coil_dim=2,
    csm_coil_dim=ndims(csm),
    n_virtual_coils=10,
    noise_covariance,
    prewhitening_scale_factor=prewhitening_scale,
)
```

Use `data_cc` in `recon_HOOp` and `csm_cc` when constructing `HighOrderOp`, `HighOrderKernelOp`, or `HighOrderLowRankOp`. No compression-specific mode is required in the encoding operator; the number of receive channels is inferred from the supplied coil-sensitivity maps.

## Receive-noise whitening

Let $D\in\mathbb C^{N_o\times N_c}$ denote the calibration observations and let $\Psi_{\mathrm{noise}}\in\mathbb C^{N_c\times N_c}$ denote the positive-definite receive-noise covariance estimated from a noise prescan. Receive-noise whitening accounts for correlated channel noise before the compression subspace is fitted. [[14]](/references#ref-14 "Kellman P, McVeigh ER. Image reconstruction in SNR units: A general method for SNR measurement. Magn Reson Med. 2005;54:1439-1447.")

When the noise and acquisition sampling conditions differ, HighOrderMRI uses the dimensionless scale

$$
s
=
\frac{T_{\mathrm{acq}}}{T_{\mathrm{noise}}}
R_{\mathrm{BW}},
$$

and defines the effective acquisition covariance as

$$
\Psi_{\mathrm{acq}}
=
\frac{\Psi_{\mathrm{noise}}}{s}.
$$

A right whitening matrix $W_n$ is constructed such that

$$
W_n^H\Psi_{\mathrm{acq}}W_n
=
I.
$$

If the covariance is represented by the upper Cholesky factor

$$
\Psi_{\mathrm{noise}}
=
U_\Psi^H U_\Psi,
$$

then the implemented right whitening matrix is

$$
W_n
=
\sqrt{s}\,U_\Psi^{-1}.
$$

The notation $W_n$ distinguishes receive-noise whitening from the reconstruction weighting operator $W$ used in expressions such as $A^H W^H W A$.

## Global SVD compression

The noise-whitened calibration matrix is factorized as

$$
D W_n
=
U\Sigma V^H.
$$

If $V_r$ contains the first $r$ right singular vectors, the combined whitening/compression transform is

$$
C
=
W_nV_r.
$$

The same transform is applied to measured data and coil-sensitivity maps:

$$
D_{\mathrm{cc}}
=
DC,
\qquad
S_{\mathrm{cc}}
=
SC,
$$

where $S$ denotes the coil-sensitivity maps reshaped with receive channels in columns. The compressed covariance then satisfies

$$
C^H\Psi_{\mathrm{acq}}C
=
I.
$$

For the stated calibration matrix, truncated SVD provides the minimum Frobenius-norm rank-$r$ approximation among global linear subspaces. [[5]](/references#ref-5 "Eckart C, Young G. The approximation of one matrix by another of lower rank. Psychometrika. 1936;1:211-218.") This result does not imply a general optimum for parallel-imaging g-factor, arbitrary k-space sampling, or spatially varying coil-compression models.

## Calibration-region fitting

The compression transform may be estimated from a calibration subset and subsequently applied to the full acquisition using [`fit_coil_compression`](/reference/fit-coil-compression) and [`apply_coil_compression`](/reference/apply-coil-compression):

```julia
coil_transform = fit_coil_compression(
    @view(data_matrix[calibration_samples, :]);
    coil_dim=2,
    energy_threshold=0.99,
    noise_covariance,
)

data_cc = apply_coil_compression(
    data_matrix,
    coil_transform;
    coil_dim=2,
)

csm_cc = apply_coil_compression(
    csm,
    coil_transform;
    coil_dim=ndims(csm),
)
```

Specify exactly one of `n_virtual_coils` and `energy_threshold`. An energy threshold describes the fraction of squared singular-value energy retained in the fitted calibration matrix; it is not an image-accuracy criterion.

For controlled comparisons, the fitted transform should be held fixed across reconstruction conditions. Report the retained virtual-coil count and `retained_energy`. When `noise_covariance` is supplied, the reported retained energy refers to the whitened calibration matrix.

## Absence of a noise prescan

If `noise_covariance` is omitted, identity covariance is assumed and the procedure reduces to conventional global PCA/SVD compression. This case should not be described as noise-optimal when receive-channel noise is correlated.

`estimate_noise_covariance` expects noise-only data acquired with channel ordering and receiver gains compatible with the MRI acquisition. If the noise scan and MRI acquisition use different dwell times or effective receiver bandwidths, specify the corresponding `noise_prewhitening_scale_factor`. The covariance must be positive definite; otherwise additional noise observations or an explicitly defined covariance-regularization procedure are required.

Density compensation is unrelated to receive-noise covariance estimation and should not be applied before estimating the covariance.

## Relation to geometric-decomposition coil compression

Geometric-decomposition coil compression (GCC) uses spatially varying, aligned compression matrices along a nonsubsampled Cartesian dimension and can retain coil sensitivity more efficiently than a single global transform in the acquisition geometries for which it was designed. [[15]](/references#ref-15 "Zhang T, Pauly JM, Vasanawala SS, Lustig M. Coil compression for accelerated imaging with Cartesian sampling. Magn Reson Med. 2013;69:571-582.")

The current HighOrderMRI encoding operators instead accept a single right-side transform that can be applied consistently to arbitrary Cartesian or non-Cartesian acquired data and their coil-sensitivity maps. Accordingly, the implemented method should be described as **global SVD/PCA coil compression**, optionally preceded by receive-noise whitening, rather than as GCC. A GCC implementation would require a separate hybrid-space/operator treatment and independent validation.

## Reporting

For a methods or performance study, report:

- the source and extent of the calibration data;
- whether receive-noise covariance was used;
- noise/acquisition dwell times and receiver-bandwidth scaling, when applicable;
- virtual-coil count or retained-energy threshold;
- final retained energy;
- confirmation that the same transform was applied to acquired data and coil-sensitivity maps;
- confirmation that the fitted transform was held fixed across compared encoding operators.
