# `InterpTrajTime`

Interpolate measured or predicted field coefficients to the ADC times. The
two-dimensional input stores field samples in rows and coefficients in columns.
This operation shifts time coordinates; it does not integrate fields or change
their spatial units.

For input index $j=1,\ldots,n_\mathrm{field}$, the interpolation support is

$$
t_j=(j-1)\,dt-\mathrm{delTime}.
$$

Thus a positive offset evaluates the unshifted field at
$t_\mathrm{ADC}+\mathrm{delTime}$. Values outside the shifted interval are
zero, which should not be mistaken for measured zero field.

## Check the sign on a linear signal

This complete example uses matching Float32 times and a linear interpolation
method. All requested points lie inside the shifted input interval.

```julia
using HighOrderMRI
using Interpolations: LinearMonotonicInterpolation

T = Float32
dt = T(1e-3)
field_times = T.(0:5) .* dt
coefficients = hcat(field_times, 2 .* field_times)
adc_times = field_times[1:3]
shifted = InterpTrajTime(coefficients, dt, dt, adc_times;
    intermode=LinearMonotonicInterpolation())
@assert shifted[:, 1] ≈ adc_times .+ dt
@assert size(shifted) == (3, 2)
```

For an encoding operator, transpose the resulting coefficient matrix to put
terms in rows. Multishot interpolation accepts a separate offset and ADC-time
row for each shot. Its shape contract differs from the current limitations of
[`FindDelay_multishot`](/reference/find-delay-multishot).

Use an explicit same-type ADC-time vector when dwell times differ. The
convenience overload generates times from `dt`; a Float64 `dt` with Float32
coefficients can currently fail dispatch. The complete contracts follow.

## API

<!--@include: ../../.generated/api/interp-traj-time.md-->
