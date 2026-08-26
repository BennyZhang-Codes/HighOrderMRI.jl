# API reference

The API reference is organized as a set of short pages, following the same separation used by established Julia MRI documentation: each page focuses on one public type, constructor, or closely related function family. Scientific derivations and validation requirements remain in the [Theory](/theory/encoding-model) and [Guide](/guide/operators) sections.

## Encoding model and operators

- [`Grid` and `basisfunc_spha`](/reference/grid-basis): physical reconstruction grid and real solid-harmonic basis evaluation.
- [`HighOrderOp`](/reference/highorderop): array-based explicit field-aware encoding.
- [`HighOrderKernelOp`](/reference/highorderkernelop): fused explicit CUDA encoding with optional multi-GPU voxel decomposition.
- [`HighOrderLowRankOp`](/reference/highorderlowrankop): low-rank residual-phase representation with an incremental shared spatial basis.

## Reconstruction

- [`recon_HOOp`](/reference/recon-hoop): iterative reconstruction wrapper.
- [`samplingDensity`](/reference/sampling-density): square-root sampling-density weights for non-Cartesian reconstruction.

## Coil compression

- [`CoilCompressionTransform`](/reference/coil-compression-transform): fitted linear receive-coil transform.
- [`estimate_noise_covariance`](/reference/estimate-noise-covariance): complex receive-noise covariance estimation.
- [`noise_prewhitening_scale_factor`](/reference/noise-prewhitening-scale): dwell-time / receiver-bandwidth scale factor.
- [`fit_coil_compression`](/reference/fit-coil-compression): fit global SVD or noise-whitened SVD compression.
- [`apply_coil_compression`](/reference/apply-coil-compression): apply a fitted transform along a selected coil dimension.
- [`compress_coils`](/reference/compress-coils): fit one transform and apply it consistently to data and CSM.

## Field prediction and synchronization

- [`GIRFModel`](/reference/girf-model): typed frequency-domain GIRF representation.
- [`apply_girf`](/reference/apply-girf): predict realized field channels from nominal physical gradients.
- [`InterpTrajTime`](/reference/interp-traj-time): interpolate field/trajectory coefficients to requested ADC times.
- [`FindDelay`](/reference/find-delay): single-acquisition model-based field/data synchronization.
- [`FindDelay_multishot`](/reference/find-delay-multishot): shared-delay estimation for multi-shot data.

## Metrics and utilities

- [Reconstruction metrics](/reference/image-metrics): raw and aligned complex NRMSE, magnitude NRMSE, SSIM, and compatibility helpers.
- [Array and signal utilities](/reference/utilities): CPU/GPU conversion, trajectory-gradient conversion, resizing, cropping, and factor helpers.
- [Plotting helpers](/reference/plotting): common visualization functions.
- [Resource lifecycle](/reference/resources): explicit cleanup and safe replacement of large low-rank operators.

::: tip Reference vs. methods
Use these pages to check calling conventions and returned objects. Use [Symbols and notation](/theory/symbols), the [expanded encoding model](/theory/encoding-model), [low-rank shared subspace](/theory/low-rank), and the relevant Guide page when interpreting units, phase conventions, approximation parameters, or validation requirements.
:::
