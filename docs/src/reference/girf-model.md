# `GIRFModel`

A GIRF model stores the frequency response from the three device-coordinate
gradient channels to the calibrated output channels. It represents a linear,
time-invariant prediction model; it is not a measured waveform. See
[field preprocessing](/guide/field-preprocessing#girf-prediction) for the
physical assumptions and reference.

## Matched-grid identity example

This self-contained CPU example uses a diagonal identity response. Its eight
frequency samples already match the FFT length, so it does not exercise the
current response-expansion limitation.

```julia
using HighOrderMRI

T = Float64
n = 8
dt = T(1e-5)
freqs = T.(-4:3) ./ (n * dt)
Hw = zeros(Complex{T}, n, 3, 3)
for axis in 1:3
    Hw[:, axis, axis] .= 1
end
model = GIRFModel(Hw, freqs)
nominal = reshape(T.(1:3n), 3, n)
predicted = apply_girf(nominal, model)
@assert predicted ≈ nominal
```

Supply the actual calibrated response and its channel units for acquisition
data. The constructor stores CPU arrays but does not guarantee independent
copies; use `copy` before construction if later mutation must be isolated.

::: warning Sampling metadata does not resample the response
`apply_girf(nominal, model)` forwards `model.Hw`. The stored frequency vector
does not adapt the response to a different waveform dwell time. Check both
sampling and the [FFT boundary limitations](/reference/apply-girf#current-limits).
:::

## API

<!--@include: ../../.generated/api/girf-model.md-->
