"""
    CoilCombineSOS(imgs::AbstractArray{T, N},
        coildim::Int) where {T, N}

Combine coil images by `sqrt.(sum(abs2, imgs; dims=coildim))` and remove the
specified coil dimension. The result has one fewer dimension and real,
nonnegative values for ordinary real or complex floating-point input.
`coildim` identifies the receive-coil axis; the input need not be 4D.

This magnitude combination discards phase and does not use coil sensitivities
or noise covariance. It is neither a sensitivity-weighted inverse nor an
absolute SNR estimate.
"""
function CoilCombineSOS(imgs::AbstractArray{T,N}, coildim::Int) where {T,N}
    # imgs: (nCoils, nZ, nY, nX)
    # coildim: 1
    imgs = sqrt.(sum(abs.(imgs).^2; dims=coildim))
    imgs = dropdims(imgs,dims=coildim)
    return imgs
end