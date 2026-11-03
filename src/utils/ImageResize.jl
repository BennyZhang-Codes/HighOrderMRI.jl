"""
    imresize_real(img::AbstractArray{T}, dims::Tuple) where T <: Real

Resize a real array with `ImageTransformations.imresize`. `length(dims)` must
equal `ndims(img)`. Interpolation settings and output numeric behavior are those
of the dependency's default `imresize` call; this is not a physical-coordinate
or geometry-aware resampling routine.
"""
function imresize_real(img::AbstractArray{T}, dims::Tuple) where T<:Real
    @assert length(dims) == ndims(img) "length of dims must be equal to ndims(img)"
    return imresize(img, dims...)
end

"""
    imresize_complex(img::AbstractArray{Complex{T}}, dims::Tuple) where T <: Real

Resize magnitude and wrapped phase separately using the default `imresize`,
then return `resized_magnitude .* exp.(im .* resized_phase)`. `length(dims)`
must equal `ndims(img)`.

The phase is not unwrapped. Values near the -pi/pi branch cut can interpolate
through zero phase and create an incorrect complex transition. This method is
not equivalent to interpolating real and imaginary parts, and does not promise
phase-preserving physical resampling.
"""
function imresize_complex(img::AbstractArray{Complex{T}}, dims::Tuple) where T<:Real
    @assert length(dims) == ndims(img) "length of dims must be equal to ndims(img)"
    mag = imresize(abs.(img), dims...)
    pha = imresize(angle.(img), dims...)
    return mag .* exp.(im * pha)
end

