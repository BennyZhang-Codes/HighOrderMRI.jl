# `apply_coil_compression`

Apply a fitted coil-compression transform along a selected receive-channel dimension.

```julia
apply_coil_compression(
    array,
    transform::CoilCompressionTransform;
    coil_dim=ndims(array),
)
```

## Arguments

- `array`: complex data or coil-sensitivity maps.
- `transform`: fitted [`CoilCompressionTransform`](/reference/coil-compression-transform).

## Keywords

- `coil_dim`: array dimension containing the receive channels.

## Returns

A CPU `Array` with the selected coil dimension replaced by the number of virtual coils.

The input coil count must equal `size(transform, 1)`. Apply the **same** fitted transform to acquired data and CSM so that the compressed signal model remains consistent.

## Example

```julia
data_cc = apply_coil_compression(
    data,
    transform;
    coil_dim=2,
)

csm_cc = apply_coil_compression(
    csm,
    transform;
    coil_dim=ndims(csm),
)
```

[Source: `CoilCompression.jl`](https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/docs-modern-ui/src/Recon/CoilCompression.jl)
