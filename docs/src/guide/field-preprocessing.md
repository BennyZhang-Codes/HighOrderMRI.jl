# Field preprocessing and synchronization

Field coefficients must be aligned to ADC sample times and expressed in the
same phase convention as the encoding operator. HighOrderMRI provides three
related tools:

- GIRF-based prediction of gradient and higher-order field waveforms;
- interpolation of measured/predicted coefficients onto ADC times;
- model-based estimation of the field/data synchronization delay.

## GIRF prediction

[`GIRFModel`](@ref) stores a centred frequency-domain transfer function with
shape `(nFreq, 3, nOut)` and its frequency samples:

```julia
model = GIRFModel(Hw, freqs)
```

Apply it to nominal DCS gradients:

```julia
predicted = apply_girf(
    nominal_gradients,
    model;
    dim_spatial=1,
    dim_time=2,
    rbw=1.0,
)
```

The spatial input dimension must contain three physical gradient channels.
The corresponding output dimension is replaced by `nOut`, for example a
zeroth-order term, three linear terms, and higher-order terms.

`apply_girf` expects the DC component of `Hw` to be centred. If the requested
FFT is longer than `nFreq`, the implementation converts the response to time,
zero-pads between its causal and non-causal portions, and transforms it back.
`rbw < 1` suppresses the outer part of the frequency band.

!!! note
    `GIRFModel.freqs` records the frequency grid, but the current convenience
    overload passes only `Hw` to `apply_girf`; it does not resample the
    transfer function from `freqs`. Match the GIRF and waveform sampling
    grids before calling the function.

Convert predicted instantaneous gradients/fields to accumulated trajectory or
phase coefficients with [`grad2traj`](@ref), using the correct gyromagnetic
and unit conversion for the acquisition:

```julia
accumulated = grad2traj(predicted, dt; dim=2)
```

## Interpolate to ADC times

[`InterpTrajTime`](@ref) shifts a coefficient time series and interpolates it
onto requested ADC times:

```julia
kspha_at_adc = InterpTrajTime(
    kspha_samples,  # (nFieldSample, nTerm)
    field_dt,
    start_time + delay,
    adc_times,
)
```

The two-dimensional helper expects samples in rows and field terms in
columns. Transpose the result before passing it to an explicit operator:

```julia
op = HighOrderOp(grid, permutedims(kspha_at_adc), adc_times; ...)
```

For multishot data, use an array shaped `(nShot,nFieldSample,nTerm)`, one
delay per shot, and ADC times shaped `(nShot,nSamplePerShot)`.

Interpolation extrapolates outside the measured range with zeros. Inspect the
requested time interval; silent leading or trailing zeros can otherwise look
like a physical field transition.

## Model-based delay estimation

[`FindDelay`](@ref) implements the model-based field/data synchronization
method of Dubovan and Baron and returns the estimated scalar delay:

```julia
delay = FindDelay(
    grid,
    data,                # (nSample, nCoil)
    field_coefficients,  # (nFieldSample, nTerm)
    adc_times,
    start_time,
    field_dt;
    JumpFact=3,
    Δτ_min=delay_tolerance,
    csm,
    use_gpu=true,
)
```

`StartTime`, `dt`, `adc_times`, and `Δτ_min` must use the same time unit.
At the current delay ``\tau``, the routine interpolates the measured field at
the ADC sample times using `StartTime + τ`. Keep the sign convention explicit
when comparing the returned delay with scanner or field-camera logs.

At every outer iteration the implementation:

1. shifts and interpolates the field coefficients;
2. reconstructs an image with density-weighted data;
3. evaluates the derivative forward product ``B_\tau x_\tau``;
4. estimates a scalar update from
   ``y-A_\tau x_\tau \approx \Delta\tau B_\tau x_\tau``;
5. reduces `JumpFact` after an update sign change and stops when the update is
   no larger than `Δτ_min`.

The derivative coefficients use the five-point kernel implemented in
`FindDelay`. The delay update itself is unweighted; density weights are used
for the image reconstruction step. The current function returns only the
final delay and does not expose iteration history, a convergence-status
object, coarse-search initialization, or a maximum outer iteration count.

For multi-shot data, [`FindDelay_multishot`](@ref) accepts coefficients shaped
`(nShot,nFieldSample,nTerm)`, ADC times shaped
`(nShot,nSamplePerShot)`, and one `StartTime` value per shot:

```julia
delay = FindDelay_multishot(
    grid,
    data,
    field_coefficients_by_shot,
    adc_times_by_shot,
    start_times,
    field_dt;
    Δτ_min=delay_tolerance,
    csm,
)
```

`FindDelay_multishot` estimates one delay shared by all shots while respecting
their individual start offsets.

Coil compression can reduce the cost of the repeated explicit-operator
evaluations. Fit one transform and apply it to both `data` and `csm` before
calling either delay estimator. When noise-only samples are available, use the
same noise-whitened coil-compression workflow described in
[Reconstruction workflow](reconstruction.md#coil-compression).

Before using an estimated delay in a final study:

- plot coefficients before and after shifting;
- confirm no relevant samples fall outside the measured field window;
- record the returned delay, `StartTime`, `dt`, `Δτ_min`, `JumpFact`,
  interpolation mode, solver settings, coil-compression rank, and software
  version with the reconstruction manifest;
- compare predicted/measured first-order trajectories against an independent
  reference.

## Phase preparation checklist

Before operator construction, verify:

1. field coefficients are time integrated;
2. radians have been converted to cycles;
3. spatial coefficient units match the `Grid` coordinate unit;
4. row order matches [`basisfunc_spha`](@ref);
5. the zero-order term has not been corrected twice in both data and `kspha`;
6. the same synchronization and trajectory corrections are used for density
   compensation and encoding.
