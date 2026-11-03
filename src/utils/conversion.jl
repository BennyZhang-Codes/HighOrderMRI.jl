"""
    grad2traj(x::AbstractArray{T}, dt::S; dim::Int = 1) where {T <: Real, S <: Real}

Cumulatively integrate `x` by the trapezoidal rule along `dim`, prepending a
zero sample. For a nonempty selected axis, the output has the input shape.
The accumulated increments are `(x[j+1] + x[j]) * dt / 2`; arithmetic determines
the result precision, so it need not equal the input element type.

Use a valid dimension and floating-point host arrays. No gyromagnetic ratio,
unit conversion, or initial trajectory offset is applied. For example, Hz/m
integrated over seconds gives cycles/m; a magnetic gradient in T/m requires the
appropriate gyromagnetic conversion before this helper represents k-space.
"""
function grad2traj(x::AbstractArray{T}, dt::S; dim::Int=1) where {T<:Real, S<:Real}
    # Create an output array with the same type and shape as x
    y = similar(x)
    y .= zero(T)  # initialize with zeros

    # Build index templates for slicing
    inds1 = ntuple(_ -> Colon(), ndims(x))  # for x[1:end-1, ...]
    inds2 = ntuple(_ -> Colon(), ndims(x))  # for x[2:end, ...]

    # Set the selected dimension range for trapezoidal rule
    inds1 = Base.setindex(inds1, 1:size(x, dim)-1, dim)
    inds2 = Base.setindex(inds2, 2:size(x, dim), dim)

    # Trapezoidal integration over selected dimension
    @views trapz = (x[inds1...] .+ x[inds2...]) .* (dt / 2)

    # Cumulative sum along the same dimension
    cs = cumsum(trapz; dims=dim)

    # Create a zero slice to prepend (matching shape except dim = 1)
    zeros_shape = size(x)
    zeros_shape = Base.setindex(zeros_shape, 1, dim)
    z = fill(zero(T), zeros_shape)

    # Concatenate zero at the beginning along the integration dimension
    y = cat(z, cs; dims=dim)

    return y
end


"""
    traj2grad(x::AbstractArray{T}, dt::S; dim::Int = 1) where {T <: Real, S <: Real}

Compute adjacent finite differences `(x[j+1] - x[j]) / dt` along `dim`.
The selected output axis is shorter by one; other dimensions are unchanged.
There is no padded endpoint or differentiation of a separately supplied time
vector. Use a nonzero dwell and compatible physical units. No gyromagnetic
conversion is applied, and arithmetic determines the output precision.
"""
function traj2grad(x::AbstractArray{T}, dt::S; dim::Int=1) where {T<:Real, S<:Real}
    # Build slice templates for previous and current
    inds_prev = ntuple(_ -> Colon(), ndims(x))
    inds_curr = ntuple(_ -> Colon(), ndims(x))

    # Set the slicing range along the selected dimension
    inds_prev = Base.setindex(inds_prev, 1:size(x, dim)-1, dim)
    inds_curr = Base.setindex(inds_curr, 2:size(x, dim), dim)

    # Compute and return the result directly
    @views return (x[inds_curr...] .- x[inds_prev...]) ./ dt
end