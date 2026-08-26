# `GIRFModel`

Typed container for a frequency-domain gradient impulse response function (GIRF).

```julia
GIRFModel(Hw, freqs)
```

## Arguments

- `Hw`: complex frequency-domain transfer function with shape `(nFreq, 3, nOut)`.
- `freqs`: frequency samples corresponding to the first dimension of `Hw`.

The three input channels represent the physical gradient axes. Output channels can contain zeroth-, first-, and higher-order field terms.

## Returns

A `GIRFModel{T}` storing CPU copies of `Hw` and `freqs`.

::: info Sampling-grid behavior
The convenience overload `apply_girf(nominal, model)` currently forwards `model.Hw`; it does not resample the response using `model.freqs`. The GIRF and waveform sampling grids must therefore already be consistent.
:::

## Example

```julia
model = GIRFModel(Hw, freqs)
predicted = apply_girf(nominal_gradients, model)
```

See [Field preprocessing and synchronization](/guide/field-preprocessing) for the physical interpretation and GIRF reference.

[Source: `GIRF.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/GIRF/GIRF.jl)
