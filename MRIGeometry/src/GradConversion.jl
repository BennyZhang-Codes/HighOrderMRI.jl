"""
    grad_Nominal2DCS(G_Nominal::AbstractArray{T, N}, geo::Geometry; dim::Int = N == 2 ? 1 : N, R_polarity::AbstractMatrix = geo.R_Nominal_RPS) where {T, N}

Rotate nominal sequence gradients into device coordinates, applying
`geo.R_RPS_DCS * R_polarity` to the three-channel axis `dim`. The result
preserves input shape; no unit conversion, time integration, GIRF response
or measurement correction is applied. By default dim is 1 for matrices and
N otherwise. Confirm the vendor polarity and geometry; placeholder mappings
are not calibrated scanner support. An explicit R_polarity must be compatible
with the chosen coordinate convention.
"""
function grad_Nominal2DCS(
    G_Nominal::AbstractArray{T, N}, 
    geo::Geometry; 
    dim::Int = N == 2 ? 1 : N,
    R_polarity::AbstractMatrix = geo.R_Nominal_RPS
) where {T, N}
    # Combine the final transformation matrix: 
    # Apply polarity mapping (R_polarity) first, then rotate to the actual physical DCS
    R_total = geo.R_RPS_DCS * R_polarity
    
    size(G_Nominal, dim) == 3 || throw(ArgumentError("Spatial dimension (dim=$dim) must have length 3."))
    
    if dim == 1
        res = R_total * reshape(G_Nominal, 3, :)
        return reshape(res, size(G_Nominal))
    else
        perm = (dim, (1:dim-1)..., (dim+1:N)...)
        inv_perm = ((2:dim)..., 1, (dim+1:N)...)
        G_perm = permutedims(G_Nominal, perm)
        res = R_total * reshape(G_perm, 3, :)
        return permutedims(reshape(res, size(G_perm)), inv_perm)
    end
end

"""
    grad_DCS2Nominal(G_DCS::AbstractArray{T, N}, geo::Geometry; dim::Int = N == 2 ? 1 : N, R_polarity::AbstractMatrix = geo.R_Nominal_RPS) where {T, N}

Apply `transpose(R_polarity)*transpose(geo.R_RPS_DCS)` along the three-channel
axis, preserving shape and physical units. This is the inverse of the forward
mapping only when both matrices are orthogonal. It does not invert a GIRF
or estimate measured nominal gradients. Device mappings inherit Geometry's
vendor limitations.
"""
function grad_DCS2Nominal(
    G_DCS::AbstractArray{T, N}, 
    geo::Geometry; 
    dim::Int = N == 2 ? 1 : N,
    R_polarity::AbstractMatrix = geo.R_Nominal_RPS
) where {T, N}
    # Inverse transformation: 
    # Convert physical DCS back to logical RPS, then strip the polarity mapping to restore the nominal sequence design
    R_total_inv = transpose(R_polarity) * transpose(geo.R_RPS_DCS)
    
    size(G_DCS, dim) == 3 || throw(ArgumentError("Spatial dimension (dim=$dim) must have length 3."))
    
    if dim == 1
        res = R_total_inv * reshape(G_DCS, 3, :)
        return reshape(res, size(G_DCS))
    else
        perm = (dim, (1:dim-1)..., (dim+1:N)...)
        inv_perm = ((2:dim)..., 1, (dim+1:N)...)
        G_perm = permutedims(G_DCS, perm)
        res = R_total_inv * reshape(G_perm, 3, :)
        return permutedims(reshape(res, size(G_perm)), inv_perm)
    end
end