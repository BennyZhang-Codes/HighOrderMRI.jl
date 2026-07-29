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

[`FindDelay`](@ref) and [`FindDelay_multishot`](@ref) implement the
model-based field/data synchronization method of Dubovan and Baron. At each
iteration the algorithm:

1. shifts and interpolates the field coefficients;
2. reconstructs an image with the current encoding;
3. evaluates the derivative encoding ``B_\tau x_\tau``;
4. estimates an update from the residual relation
   ``y-A_\tau x_\tau\approx \Delta\tau B_\tau x_\tau``;
5. reduces the jump factor after a sign change and stops when the update falls
   below `Δτ_min`.

The delay unit is the same as `dt`, `StartTime`, `datatime`, and `Δτ_min`.
Keep these quantities in one explicit unit system; the default numerical
value alone does not establish whether an input is seconds or microseconds.

Before using an estimated delay in a final study:

- plot coefficients before and after shifting;
- confirm no relevant samples fall outside the measured field window;
- repeat from several initial offsets when possible;
- store the delay, update history, interpolation method, and field sampling
  interval with the reconstruction manifest;
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
