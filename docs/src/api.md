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
HighOrderKernelOp
HighOrderLowRankOp
@rebuild_HOOp
```

### Automatic joint shared basis

The existing local rSVD construction remains the default. To opt into joint
phase snapshots and automatic selection of the final shared rank:

```julia
report = Ref{Any}()
op = HighOrderLowRankOp(grid, kspha, times;
    fieldmap, csm, mask,
    shared_basis_method=:joint,
    shared_basis_tol=Float32(1e-2),
    shared_rank_max=256,
    joint_snapshots=512,
    joint_samples=512,
    joint_basis_report=report,
    global_basis_tol=nothing,
)
report[].rank
maximum(report[].audit_errors)
maximum(report[].roi_errors)
```

Match tolerance types to the input precision. Existing `arrayType`, `gpus`,
`normal_distribution` and reconstruction options still apply. Joint setup
runs on the primary GPU or CPU; it does not use voxel-distributed rSVD.
`L_rank` does not truncate local factors in this mode. `rsvd_chunk` bounds
temporary phase matrices. The requested `joint_snapshots` budget is used
before choosing rank, subject to available phase geometries. The result uses
the existing operator, NFFT and normal implementations, including all rank
cross terms.

Here `shared_basis_tol` constrains a sampled original-phase matrix error for
each profile, unlike the incremental compression error of the default path.
The search tests rank 1 and blocks of eight, then audits the completed factors
on fresh time samples. Selection reserves 10% of each requested tolerance for
sampling variation; the final audit still uses the requested tolerance and
can fail. This heuristic is not a proof of the minimum rank or an image-error
guarantee. Reports include the seed, fitting/validation/audit indices and the
per-profile errors; tiny inputs record when time samples must be reused.

High-|B0| voxels receive an additional diagnostic. Set
`joint_roi_tol=Float32(1e-2)` to require that region to pass as well. Without
this option, an excessive ROI error is reported and warned about. In the EPI
experiment, a random-voxel 1% pass did **not** imply a high-B0-region 1% pass.
Rank/sample exhaustion and failed audits throw, with diagnostics in `report`;
no partially validated operator is returned. Post-audit `global_basis_tol`
compression is rejected because it would change the checked factors.

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
