"""
    imgs=CoilCombineSOS(imgs, coildim)

# Description
    Sum-Of-Squares (SOS) coil combination. 

# Arguments
- `imgs::AbstractArray{T,N}`: A 4D array of images.
- `coildim::Int`: The coil dimension.

# Returns
- `imgs::AbstractArray{T,N-1}`: A 3D array of images.
"""
function CoilCombineSOS(imgs::AbstractArray{T,N}, coildim::Int) where {T,N}
    # imgs: (nCoils, nZ, nY, nX)
    # coildim: 1
    imgs = sqrt.(sum(abs.(imgs).^2; dims=coildim))
    imgs = dropdims(imgs,dims=coildim)
    return imgs
end