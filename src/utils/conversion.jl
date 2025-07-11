"""
    grad2traj(x::AbstractArray{Real}, dt::Real; dim::Int=1)

Trapezoidal cumulative integration over time.

# Arguments
- `x::AbstractArray{<:Real}`: Input data (gradient)
- `dt::<:Real`: Time interval between two samples

# Keywords
- `dim::Int`: Dimension along which to integrate

# Returns
- `y::AbstractArray{<:Real}`: Cumulative integral over time, same size as `x`
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
    traj2grad(x::AbstractArray{T}, dt::S; dim::Int=1) where {T<:Real, S<:Real}

Compute time derivative (gradient) of input trajectory using backward difference.

# Arguments
- `x::AbstractArray{<:Real}`: Input data (trajectory)
- `dt::<:Real`: Time interval between samples

# Keywords
- `dim::Int`: Dimension along which to compute the gradient

# Returns
- `dx::AbstractArray{<:Real}`: same size as `x`
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