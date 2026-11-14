# `apply_girf`

Predict field waveforms by applying a calibrated frequency-domain response to
nominal gradients in device coordinates. Each extra input dimension is processed
independently, so separate shots need not be concatenated across their dead time.
The output channel order comes from the response calibration.

This complete CPU example applies a matched-length identity response directly.
It checks the array interface without invoking response expansion:

```julia
using HighOrderMRI

T = Float64
nominal_gradients = zeros(T, 3, 8)
nominal_gradients[1, 3] = 1
Hw = zeros(Complex{T}, 8, 3, 3)
for axis in 1:3
    Hw[:, axis, axis] .= 1
end
predicted = apply_girf(nominal_gradients, Hw;
    dim_spatial=1, dim_time=2, rbw=1.0)
@assert predicted ≈ nominal_gradients
```

For the model-container form, see the
[complete GIRFModel example](/reference/girf-model#matched-grid-identity-example).
A calibrated response must match the acquisition sampling and physical units.

`rbw` suppresses outer waveform-frequency bins. It is not an estimate of the
model's uncertainty. The function does not integrate gradients or apply a
gyromagnetic ratio; perform the appropriate conversion before using the result
as phase coefficients. See [phase preparation](/guide/field-preprocessing#phase-preparation-checklist).

## Current limits

The FFT multiplication is periodic convolution. Padding the waveform to
`nextprod([2,3,5],max(nPoint,nFreq))` does not by itself prevent a late response
from wrapping to the start. Determine the required pre/post support from the
acquisition and calibrated impulse response.

If this FFT length exceeds the supplied response length, the current time-domain
response expansion omits one middle coefficient. That can remove a valid
response tap, including for odd lengths. An already matched response length
avoids this expansion path; it does not remove periodic-boundary effects.
These numerical limitations remain in the implementation.

The real inverse transform is returned. A physically compatible real response,
consistent frequency ordering and matched sample intervals must be supplied;
the API does not validate all these calibration conditions. A `GIRFModel`
wrapper adds metadata but does not resample its response.

## API

<!--@include: ../../.generated/api/apply-girf.md-->
