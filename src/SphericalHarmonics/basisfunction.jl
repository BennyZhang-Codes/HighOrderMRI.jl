
"""
    bfuncOut = basisfunc_spha(x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}, terms::AbstractVector{Int}) where {T<:AbstractFloat}
    Outputs basis function for index terms of spherical harmonics.

# Arguments
- `x::AbstractVector{T}`: x-coordinates of imaging object.
- `y::AbstractVector{T}`: y-coordinates of imaging object.
- `z::AbstractVector{T}`: z-coordinates of imaging object.
- `terms::AbstractVector{Int}`: Indices for basis functions.

# Returns
- `bfuncOut::Array{T, 2}`: Basis function for each index in terms.

# Example
```julia
julia> x = [1, 2, 3]
julia> y = [4, 5, 6]
julia> z = [7, 8, 9]
julia> terms = [1, 2, 3, 4, 5, 6, 7, 8, 9]
julia> bfuncOut = basisfunc_spha(x, y, z, terms)
```
"""
function basisfunc_spha(x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}, terms::AbstractVector{Int}) where {T<:AbstractFloat}
    """
    Outputs basis function for index n of spherical harmonics.
    x, y, z: Coordinates (vectors or arrays).
    n: Indices for basis functions.
    """
    nSample = length(x)
    nTerm = length(terms)

    # Prepare output array
    if isa(x, CuArray)
        bfuncOut = CUDA.zeros(T, nSample, nTerm)
        bfunc = CUDA.zeros(T, nSample) 
    else
        bfuncOut = zeros(T, nSample, nTerm)
        bfunc = zeros(T, nSample) 
    end
    # Iterate over each index in n
    for (idx, term) in enumerate(terms)
        # Switch-case equivalent for selecting basis function
        if term == 1
            bfunc .= 1
        elseif term == 2
            bfunc .= x
        elseif term == 3
            bfunc .= y
        elseif term == 4
            bfunc .= z
        elseif term == 5
            bfunc .= x .* y
        elseif term == 6
            bfunc .= z .* y
        elseif term == 7
            bfunc .= 3 .* z.^2 .- (x.^2 .+ y.^2 .+ z.^2)
        elseif term == 8
            bfunc .= x .* z
        elseif term == 9
            bfunc .= x.^2 .- y.^2
        elseif term == 10
            bfunc .= 3 .* y .* x.^2 .- y.^3
        elseif term == 11
            bfunc .= x .* z .* y
        elseif term == 12
            bfunc .= (5 .* z.^2 .- (x.^2 .+ y.^2 .+ z.^2)) .* y
        elseif term == 13
            bfunc .= 5 .* z.^3 .- 3 .* z .* (x.^2 .+ y.^2 .+ z.^2)
        elseif term == 14
            bfunc .= (5 .* z.^2 .- (x.^2 .+ y.^2 .+ z.^2)) .* x
        elseif term == 15
            bfunc .= (x.^2 .* z .- y.^2 .* z)
        elseif term == 16
            bfunc .= (x.^3 .- 3 .* x .* y.^2)
        elseif term == 17
            bfunc .= (x .* y) .* (x.^2 .- y.^2)
        elseif term == 18
            bfunc .= (y .* z) .* (3 .* x.^2 .- y.^2)
        elseif term == 19
            bfunc .= (x .* y) .* (7 .* z.^2 .- (x.^2 .+ y.^2 .+ z.^2))
        elseif term == 20
            bfunc .= y .* (7 .* z.^3 .- 3 .* z .* (x.^2 .+ y.^2 .+ z.^2))
        elseif term == 21
            bfunc .= 35 .* z.^4 .- 30 .* z.^2 .* (x.^2 .+ y.^2 .+ z.^2) + 3 .* (x.^2 .+ y.^2 .+ z.^2).^2
        elseif term == 22
            bfunc .= x .* (7 .* z.^3 .- 3 .* z .* (x.^2 .+ y.^2 .+ z.^2))
        elseif term == 23
            bfunc .= (x.^2 .- y.^2) .* (7 .* z.^2 .- (x.^2 .+ y.^2 .+ z.^2))
        elseif term == 24
            bfunc .= x .* (x.^2 .- 3 .* y.^2) .* z
        elseif term == 25
            bfunc .= x.^2 .* (x.^2 .- 3 .* y.^2) .- y.^2 .* (3 .* x.^2 .- y.^2)
        elseif term == 26
            bfunc .= y .* (5 .* x.^4 .- 10 .* x.^2 .* y.^2 .+ y.^4)
        elseif term == 27
            bfunc .= x .* y .* z .* (x.^2 .- y.^2)
        elseif term == 28
            bfunc .= y .* ((x.^2 .+ y.^2 .+ z.^2) .- 9 .* z.^2) .* (y.^2 .- 3 .* x.^2)
        elseif term == 29
            bfunc .= x .* y .* z .* (3 .* z.^2 .- (x.^2 .+ y.^2 .+ z.^2))
        elseif term == 30
            bfunc .= y .* ((x.^2 .+ y.^2 .+ z.^2).^2 .- 14 .* (x.^2 .+ y.^2 .+ z.^2) .* z.^2 + 21 .* z.^4)
        elseif term == 31
            bfunc .= 63 .* z.^5 .- 70 .* z.^3 .* (x.^2 .+ y.^2 .+ z.^2) + 15 .* z .* (x.^2 .+ y.^2 .+ z.^2).^2
        elseif term == 32
            bfunc .= x .* ((x.^2 .+ y.^2 .+ z.^2).^2 .- 14 .* (x.^2 .+ y.^2 .+ z.^2) .* z.^2 + 21 .* z.^4)
        elseif term == 33
            bfunc .= z .* (3 .* z.^2 .- (x.^2 .+ y.^2 .+ z.^2)) .* (x.^2 .- y.^2)
        elseif term == 34
            bfunc .= x .* ((x.^2 .+ y.^2 .+ z.^2) .- 9 .* z.^2) .* (3 .* y.^2 .- x.^2)
        elseif term == 35
            bfunc .= z .* (x.^4 .- 6 .* x.^2 .* y.^2 .+ y.^4)
        elseif term == 36
            bfunc .= x.^5 .- 10 .* x.^3 .* y.^2 .+ 5 .* x .* y.^4
        else
            error("Basis function undefined.")
        end
        # Assign to output
        bfuncOut[:, idx] .= bfunc
    end
    return bfuncOut
end


function basisfunc_spha(x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}, term::Int64) where {T<:AbstractFloat}
    # Switch-case equivalent for selecting basis function
    if term == 1
        bfunc = x.*0 .+ 1
    elseif term == 2
        bfunc = x
    elseif term == 3
        bfunc = y
    elseif term == 4
        bfunc = z
    elseif term == 5
        bfunc = x .* y
    elseif term == 6
        bfunc = z .* y
    elseif term == 7
        bfunc = 3 .* z.^2 .- (x.^2 .+ y.^2 .+ z.^2)
    elseif term == 8
        bfunc = x .* z
    elseif term == 9
        bfunc = x.^2 .- y.^2
    elseif term == 10
        bfunc = 3 .* y .* x.^2 .- y.^3
    elseif term == 11
        bfunc = x .* z .* y
    elseif term == 12
        bfunc = (5 .* z.^2 .- (x.^2 .+ y.^2 .+ z.^2)) .* y
    elseif term == 13
        bfunc = 5 .* z.^3 .- 3 .* z .* (x.^2 .+ y.^2 .+ z.^2)
    elseif term == 14
        bfunc = (5 .* z.^2 .- (x.^2 .+ y.^2 .+ z.^2)) .* x
    elseif term == 15
        bfunc = (x.^2 .* z .- y.^2 .* z)
    elseif term == 16
        bfunc = (x.^3 .- 3 .* x .* y.^2)
    elseif term == 17
        bfunc = (x .* y) .* (x.^2 .- y.^2)
    elseif term == 18
        bfunc = (y .* z) .* (3 .* x.^2 .- y.^2)
    elseif term == 19
        bfunc = (x .* y) .* (7 .* z.^2 .- (x.^2 .+ y.^2 .+ z.^2))
    elseif term == 20
        bfunc = y .* (7 .* z.^3 .- 3 .* z .* (x.^2 .+ y.^2 .+ z.^2))
    elseif term == 21
        bfunc = 35 .* z.^4 .- 30 .* z.^2 .* (x.^2 .+ y.^2 .+ z.^2) + 3 .* (x.^2 .+ y.^2 .+ z.^2).^2
    elseif term == 22
        bfunc = x .* (7 .* z.^3 .- 3 .* z .* (x.^2 .+ y.^2 .+ z.^2))
    elseif term == 23
        bfunc = (x.^2 .- y.^2) .* (7 .* z.^2 .- (x.^2 .+ y.^2 .+ z.^2))
    elseif term == 24
        bfunc = x .* (x.^2 .- 3 .* y.^2) .* z
    elseif term == 25
        bfunc = x.^2 .* (x.^2 .- 3 .* y.^2) .- y.^2 .* (3 .* x.^2 .- y.^2)
    elseif term == 26
        bfunc = y .* (5 .* x.^4 .- 10 .* x.^2 .* y.^2 .+ y.^4)
    elseif term == 27
        bfunc = x .* y .* z .* (x.^2 .- y.^2)
    elseif term == 28
        bfunc = y .* ((x.^2 .+ y.^2 .+ z.^2) .- 9 .* z.^2) .* (y.^2 .- 3 .* x.^2)
    elseif term == 29
        bfunc = x .* y .* z .* (3 .* z.^2 .- (x.^2 .+ y.^2 .+ z.^2))
    elseif term == 30
        bfunc = y .* ((x.^2 .+ y.^2 .+ z.^2).^2 .- 14 .* (x.^2 .+ y.^2 .+ z.^2) .* z.^2 + 21 .* z.^4)
    elseif term == 31
        bfunc = 63 .* z.^5 .- 70 .* z.^3 .* (x.^2 .+ y.^2 .+ z.^2) + 15 .* z .* (x.^2 .+ y.^2 .+ z.^2).^2
    elseif term == 32
        bfunc = x .* ((x.^2 .+ y.^2 .+ z.^2).^2 .- 14 .* (x.^2 .+ y.^2 .+ z.^2) .* z.^2 + 21 .* z.^4)
    elseif term == 33
        bfunc = z .* (3 .* z.^2 .- (x.^2 .+ y.^2 .+ z.^2)) .* (x.^2 .- y.^2)
    elseif term == 34
        bfunc = x .* ((x.^2 .+ y.^2 .+ z.^2) .- 9 .* z.^2) .* (3 .* y.^2 .- x.^2)
    elseif term == 35
        bfunc = z .* (x.^4 .- 6 .* x.^2 .* y.^2 .+ y.^4)
    elseif term == 36
        bfunc = x.^5 .- 10 .* x.^3 .* y.^2 .+ 5 .* x .* y.^4
    else
        error("Basis function undefined.")
    end
    return bfunc
end