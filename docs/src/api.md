# API reference

Each reference page covers a public type, constructor, or related function
group. The [Theory](/theory/encoding-model) pages explain the equations;
the [Guide](/guide/operators) pages describe configuration and use.

<!--@include: ../.generated/api/highordermri.md-->

## Encoding model and operators

- [`Grid` and `basisfunc_spha`](/reference/grid-basis): physical reconstruction grid and real solid-harmonic basis evaluation.
- [`HighOrderExplicitOp`](/reference/highorderexplicitop): Explicit family selector for Array or CUDA acceleration.
- [Explicit Array options](/reference/explicit-array): array-based explicit encoding.
- [Explicit CUDA options](/reference/explicit-cuda): Direct kernel and GEMM and phase-reuse with blocked/cached cuBLAS, with `gpus` selecting single- or multi-GPU execution.
- [`explicit_gpu_normal`](/reference/explicit-cuda#weighted-normal): resident weighted normal from raw square-root weights; distinct from `normalOperator(op,D)` with an already squared diagonal.
- [`HighOrderLowRankOp`](/reference/highorderlowrankop): Per-readout, Incremental Shared and Direct Shared residual-phase construction, including bounded streaming/regeneration and rank-sharded multi-GPU execution.

## Reconstruction

- [`recon_HOOp`](/reference/recon-hoop): iterative reconstruction wrapper, raw sample-weight convention, CPU/CUDA input requirements, and backend cleanup.
- [`samplingDensity`](/reference/sampling-density): square-root sampling-density weights for non-Cartesian reconstruction.

- [`CoilCombineSOS`](/reference/coil-combine-sos): root-sum-of-squares combination along the selected coil dimension.

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

## Geometry

- [MRIGeometry](/reference/geometry): physical coordinates, slab handling, resampling and NIfTI export, including current restrictions.

## Metrics and utilities

- [Reconstruction metrics](/reference/image-metrics): raw and aligned complex NRMSE, magnitude NRMSE, SSIM, and compatibility helpers.
- [Array and signal utilities](/reference/utilities): CPU/GPU conversion, trajectory-gradient conversion, resizing, cropping, and factor helpers.
- [Plotting helpers](/reference/plotting): common visualization functions.
- [Resource lifecycle](/reference/resources): explicit cleanup and safe replacement of large low-rank operators.

::: tip Reference vs. methods
Use these pages to check calling conventions and returned objects. Use [Symbols and notation](/theory/symbols), the [expanded encoding model](/theory/encoding-model), [low-rank shared subspace](/theory/low-rank), and the relevant Guide page when interpreting units, phase conventions, approximation parameters, or validation requirements.
:::
