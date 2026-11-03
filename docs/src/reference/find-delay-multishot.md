# `FindDelay_multishot`

Estimate one delay shared by all shots of a two-dimensional image. The image
update combines the observations, then shotwise residual and derivative-forward
products supply a single scalar delay update. It does not estimate independent
per-shot delays. The method background and package-specific update choices are
described in [field preprocessing](/guide/field-preprocessing#model-based-delay-estimation).

## Prepare an explicit-time CPU call

The following block only prepares options; it does not start the unbounded
outer estimator loop. Use Float64 inputs for this current multishot interface,
and state the tolerance in the same time unit as the data. With a field map in
Hz, use seconds.

```julia
using HighOrderMRI
using RegularizedLeastSquares: CGNR, L2Regularization

T = Float64
delay_options = (
    arrayType=Array,
    solver=CGNR,
    reg=L2Regularization(zero(T)),
    Δτ_min=T(5e-9), # example only: 0.005 microseconds in seconds
    iter_max=10,
)
```

After supplying acquisition arrays, the intended call is:

```julia
τ = FindDelay_multishot(grid, data, coefficients_by_shot,
    adc_times_by_shot, start_times, field_dt;
    csm, fieldmap, recon_terms="111", delay_options...)
```

Use explicit ADC times and preserve sample-within-shot ordering in `data`.
The generated API below specifies all shapes, types and the actual defaults.

::: warning Current estimator restrictions
The overload that omits `datatime` builds an incorrect time-array shape.
The explicit-time overload also reuses the first shot's times inside its
per-shot residual operators. Different per-shot ADC-time vectors combined with
a nonzero field map can therefore give incorrect delay updates. Do not apply
this estimator to that case. Identical shot time vectors or a zero field map
avoid that specific phase error, without establishing convergence.
:::

`iter_max` bounds only each inner reconstruction. The outer estimator has no
iteration cap or reliable convergence-status return. Its actual threshold
default is the Float64 literal `0.005`; Float32 callers must supply a same-type
threshold, and later JumpFact reductions introduce a further Float32 state-type
limitation. The iteration log labels sample offsets as microseconds. These
limitations are documented here and remain unchanged in code.

## API

<!--@include: ../../.generated/api/find-delay-multishot.md-->
