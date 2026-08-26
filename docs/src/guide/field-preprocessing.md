# Field preprocessing and synchronization

Dynamic field coefficients must be aligned with the ADC sampling times and expressed in the same phase convention as the encoding operator. HighOrderMRI provides three related preprocessing operations: GIRF-based field prediction, interpolation of measured or predicted coefficients to ADC times, and model-based estimation of the synchronization delay. Concurrent spatiotemporal field monitoring provides the experimental basis for measured field-aware reconstruction. [[6]](/references#ref-6 "Barmet C, De Zanche N, Pruessmann KP. Spatiotemporal magnetic field monitoring for MR. Magn Reson Med. 2008;60:187-197.")

## GIRF prediction

A gradient impulse response function (GIRF) describes the gradient chain as a linear time-invariant system and can be used to predict the realized field response from a prescribed waveform. [[7]](/references#ref-7 "Vannesjo SJ, Haeberlin M, Kasper L, et al. Gradient system characterization by impulse response measurements with a dynamic field camera. Magn Reson Med. 2013;69:583-593.")

[`GIRFModel`](/reference/girf-model) stores a centred frequency-domain transfer function with shape `(nFreq,3,nOut)` together with its frequency samples:

```julia
model = GIRFModel(Hw, freqs)
```

Apply the model to nominal DCS gradients with [`apply_girf`](/reference/apply-girf):

```julia
predicted = apply_girf(
    nominal_gradients,
    model;
    dim_spatial=1,
    dim_time=2,
    rbw=1.0,
)
```

The spatial input dimension must contain three physical gradient channels. The corresponding output dimension is replaced by `nOut`, for example a zeroth-order term, three linear terms, and additional higher-order terms.

`apply_girf` expects the DC component of `Hw` to be centred. If the requested FFT length exceeds `nFreq`, the implementation transforms the response to the time domain, inserts zero padding between its causal and non-causal portions, and transforms the result back. Setting `rbw < 1` suppresses the outer part of the frequency band.

::: info Sampling-grid requirement
`GIRFModel.freqs` records the frequency grid, but the current convenience overload passes only `Hw` to `apply_girf`; it does not resample the transfer function using `freqs`. The GIRF and waveform sampling grids must therefore be matched before the function is called.
:::

Convert predicted instantaneous gradients or fields to accumulated trajectory/phase coefficients with `grad2traj`, using the gyromagnetic ratio and unit conversion appropriate to the acquisition:

```julia
accumulated = grad2traj(predicted, dt; dim=2)
```

## Interpolation to ADC times

[`InterpTrajTime`](/reference/interp-traj-time) shifts a coefficient time series and interpolates it to the requested ADC times:

```julia
kspha_at_adc = InterpTrajTime(
    kspha_samples,  # (nFieldSample, nTerm)
    field_dt,
    start_time + delay,
    adc_times,
)
```

The two-dimensional helper expects field samples in rows and field terms in columns. The result is transposed before it is passed to an explicit encoding operator:

```julia
op = HighOrderOp(grid, permutedims(kspha_at_adc), adc_times; ...)
```

For multishot data, use an array with shape `(nShot,nFieldSample,nTerm)`, one temporal offset per shot, and ADC times with shape `(nShot,nSamplePerShot)`.

Interpolation returns zero outside the measured time range. The requested interval should therefore be inspected explicitly; otherwise leading or trailing extrapolated zeros can be mistaken for physical field behavior.

## Model-based delay estimation

[`FindDelay`](/reference/find-delay) implements the model-based synchronization method described by Dubovan and Baron. [[8]](/references#ref-8 "Dubovan PI, Baron CA. Model-based determination of the synchronization delay between MRI and trajectory data. Magn Reson Med. 2023;89:721-728.") It returns an estimated scalar delay:

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

`StartTime`, `dt`, `adc_times`, and `Δτ_min` must use the same time unit. At a current delay $\tau$, the implementation samples the measured field using `StartTime + τ`. Thus, the sign convention corresponds to shifting the field-coefficient time origin by $+\tau$ before evaluation at the ADC times. This convention should be retained explicitly when the returned delay is compared with scanner or field-camera timing logs.

At each outer iteration, an image $x_\tau$ is reconstructed using the current encoding operator $A_\tau$. The derivative forward product $B_\tau x_\tau$ is then evaluated using the first-order approximation

$$
y-A_\tau x_\tau
\approx
\Delta\tau\,B_\tau x_\tau.
$$

With

$$
r_\tau=y-A_\tau x_\tau,
\qquad
b_\tau=B_\tau x_\tau,
$$

the scalar update implemented by `y2 \ y1` is

$$
\Delta\tau
=
J\,\operatorname{Re}\!\left(
\frac{b_\tau^H r_\tau}{b_\tau^H b_\tau}
\right),
$$

where $J$ is the current `JumpFact`. This is the real-valued least-squares delay update associated with the complex residual model, followed by the acceleration factor $J$.

The derivative operator uses

$$
\frac{\partial}{\partial\tau}
\exp\!\left(i2\pi\Phi_\tau\right)
=
\exp\!\left(i2\pi\Phi_\tau\right)
\left(i2\pi\frac{\partial\Phi_\tau}{\partial\tau}\right),
$$

with interpolated derivative coefficients supplying $\partial\Phi_\tau/\partial\tau$. `FindDelay` evaluates these derivative coefficients using the five-tap finite-difference kernel implemented in the source.

Density weights are used during the image-reconstruction step, whereas the scalar delay update is unweighted. `JumpFact` is reduced after a change in update sign, and the iteration stops when $|\Delta\tau|\leq\Delta\tau_{\min}$. The current function returns the final delay only; it does not expose an iteration history, a convergence-status object, coarse-search initialization, or a maximum outer-iteration count.

::: warning Convergence limitation
A returned scalar delay alone is not evidence of robust convergence. Because the current implementation has no explicit outer-iteration limit, quantitative use should document the initialization and tolerance, inspect the resulting alignment, and evaluate sensitivity to the starting delay on representative data.
:::

For multishot data, [`FindDelay_multishot`](/reference/find-delay-multishot) accepts coefficients with shape `(nShot,nFieldSample,nTerm)`, ADC times with shape `(nShot,nSamplePerShot)`, and one `StartTime` value per shot:

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

`FindDelay_multishot` estimates a **single delay shared by all shots** while retaining the individual shot start offsets. It should not be interpreted as a shot-to-shot delay estimator.

Coil compression can reduce the cost of the repeated explicit-operator evaluations. One compression transform should be fitted and then applied to both `data` and `csm` before either delay estimator is called. When representative noise-only samples are available, receive-noise covariance can be included using the procedure described in [Coil compression](/guide/coil-compression).

Other published trajectory-delay and trajectory-correction approaches include spiral gradient-delay calibration, RING for radial acquisitions, and trajectory auto-corrected reconstruction. [[9]](/references#ref-9 "Robison RK, Devaraj A, Pipe JG. Fast, simple gradient delay estimation for spiral MRI. Magn Reson Med. 2010;63:1683-1690.") [[10]](/references#ref-10 "Rosenzweig S, Holme HCM, Uecker M. Simple auto-calibrated gradient delay estimation from few spokes using Radial Intersections (RING). Magn Reson Med. 2019;81:1898-1906.") [[11]](/references#ref-11 "Ianni JD, Grissom WA. Trajectory auto-corrected image reconstruction. Magn Reson Med. 2016;76:757-768.") These methods address related timing or trajectory errors but are not interchangeable with the field/data synchronization model implemented by `FindDelay`.

Before using an estimated delay in a quantitative study:

- inspect field coefficients before and after temporal shifting;
- confirm that relevant ADC samples remain within the measured field interval;
- record the returned delay, `StartTime`, `dt`, `Δτ_min`, `JumpFact`, interpolation convention, solver settings, coil-compression rank, and software version;
- compare predicted or measured first-order trajectories with an independent reference when available.

## Phase preparation checklist

Before construction of the encoding operator, verify that:

1. field coefficients are time integrated;
2. phase supplied in radians has been converted to cycles;
3. spatial coefficient units are consistent with the `Grid` coordinate unit;
4. coefficient row order matches the [solid-harmonic basis table](/theory/encoding-model#solid-harmonic-basis-order);
5. the zeroth-order term has not been corrected twice in both the data and `kspha`;
6. identical synchronization and trajectory conventions are used for density compensation and encoding.

For quantitative validation, these preprocessing checks should be incorporated into the hierarchy described in [Scientific validation strategy](/guide/validation).
