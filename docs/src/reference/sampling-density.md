# `samplingDensity`

This trajectory-only helper estimates square-root density weights for the
weighted least-squares wrapper. It is not a coil-aware or field-aware inverse.
Supply trajectory coordinates in dimension-by-sample order; all-zero inputs
are outside the usable normalization domain.

## Example

```julia
using HighOrderMRI

kx = collect(range(-0.45, 0.45; length=32))
trajectory = [kx'; zeros(1, length(kx))]
weights = samplingDensity(trajectory, (8, 8))
@assert length(weights) == size(trajectory, 2)
```

The helper rescales all coordinates by a single maximum. For anisotropic
sampling, prepare axis scaling consistently with the intended reconstruction
grid. Use the returned weights once in [`recon_HOOp`](/reference/recon-hoop).


## API

<!--@include: ../../.generated/api/sampling-density.md-->
