# API reference

This page lists the documented public API. HighOrderMRI also re-exports
MRIGeometry.jl functionality; consult the corresponding source docstrings for
geometry conversion, resampling, NIfTI export, and plotting details.

```@docs
HighOrderMRI
```

## Grid and field basis

```@docs
Grid
SphericalHarmonics
basisfunc_spha
```

## Encoding operators

```@docs
HighOrderOp
HighOrderOp_Kernel
HighOrderLowRankOp
@rebuild_HOOp
```

## Reconstruction

```@docs
CoilCompressionTransform
estimate_noise_covariance
noise_prewhitening_scale_factor
fit_coil_compression
apply_coil_compression
compress_coils
recon_HOOp
samplingDensity
CoilCombineSOS
```

## Field prediction and synchronization

```@docs
GIRFModel
apply_girf
InterpTrajTime
FindDelay
FindDelay_multishot
```

## Reconstruction metrics

```@docs
complex_alignment_scale
raw_complex_nrmse
aligned_complex_nrmse
magnitude_nrmse
magnitude_ssim
```

The compatibility functions `HO_MSE`, `HO_RMSE`, `HO_NRMSE`, `HO_SSIM`, and
`HO_img_scale` use reference-first argument order. Prefer the explicitly named
metric functions above in new code.

## Array and signal utilities

```@docs
gpu
cpu
f32
f64
grad2traj
traj2grad
imresize_real
imresize_complex
get_center_range
get_center_crop
get_factors
```

## Plotting

```@docs
plt_plot
plt_scatter
plt_image
plt_B0map
plt_kspha
plt_ksphas
plt_bfield
plt_bfield_com
plt_grad
mosaic
```

## Resource cleanup

A `HighOrderLowRankOp` with `normal_distribution=:channel` owns its
multi-GPU normal backend. Release it with:

```julia
close(op)
# equivalent:
release_highorder_normal_backend!(op)
```
