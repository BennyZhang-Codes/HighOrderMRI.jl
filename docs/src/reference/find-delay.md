# `FindDelay`

Use `FindDelay` to estimate one temporal offset between MRI samples and measured or predicted field coefficients for a two-dimensional reconstruction. It alternates an image update with a delay update based on the current signal residual. The method is based on Dubovan and Baron; the density weighting and accelerated updates described in the API below are the package's implementation choices. [[8]](/references#ref-8 "Dubovan PI, Baron CA. Model-based determination of the synchronization delay between MRI and trajectory data. Magn Reson Med. 2023;89:721-728.")

For the forward model and update equation, see [Field preprocessing and synchronization](/guide/field-preprocessing#model-based-delay-estimation). A temporal offset for one acquisition is different from estimating a separate offset for every readout or shot.

## Prepare a CPU call

Use seconds for the ADC times, coefficient interval and temporal offsets when the static field map is in Hz. Check the coefficient order, phase units and coil ordering before estimating a delay; a timing adjustment cannot correct incompatible encoding conventions.

This first block is self-contained. It prepares a CPU configuration without running the delay-estimation loop:

```julia
using HighOrderMRI
using RegularizedLeastSquares: CGNR, L2Regularization

T = Float32
# Example tolerance: 0.005 μs expressed in seconds; choose it for your data.
delay_tolerance = T(5e-9)
delay_options = (
    arrayType=Array,
    solver=CGNR,
    reg=L2Regularization(zero(T)),
    Δτ_min=delay_tolerance,
    iter_max=10,
)
```

After loading acquisition arrays, the following call uses that configuration. It assumes a 2D `Float32` grid, host arrays with matching precision and coil ordering, and **nine coefficient columns**. The named acquisition variables must be supplied from your data; this block is not a synthetic convergence test.

```julia
τ = FindDelay(
    grid,
    data,
    field_coefficients,
    adc_times,
    start_time,
    field_dt;
    csm,
    fieldmap,
    recon_terms="111",
    delay_options...,
)
```

Keep explicit ADC times when their sampling interval differs from the field-coefficient interval. The convenience overload makes a more restrictive timing assumption, described in the generated API below.

::: warning Check convergence before interpreting the result
The current implementation has no outer-iteration limit. `iter_max` limits each image reconstruction, so reducing it does not bound the complete delay search. Evaluate the timing estimate on data with an independent alignment reference, and check that shifted ADC times remain inside the measured field interval. The scalar result does not include a success flag.
:::

Pass the tolerance explicitly with the same floating-point type as the input arrays. This both states the intended unit and avoids the current `Float32` default-keyword problem. Also note that the existing iteration log labels sample offsets as microseconds; the API notes below explain how to interpret them.

## API

<!--@include: ../../.generated/api/find-delay.md-->
