"""
    grad_Nominal2DCS(G_Nominal::AbstractArray{T, N}, geo::Geometry; dim::Int = N == 2 ? 1 : N, R_polarity::AbstractMatrix = geo.R_Nominal_RPS) where {T, N}

Converts nominal sequence gradients to the physical device coordinate system (DCS).

# Mathematical Mapping:
`G_DCS = R_RPS_DCS * R_polarity * G_Nominal`

# Arguments
- `G_Nominal`: N-dimensional nominal gradient array (as programmed in the sequence).
- `geo`: Geometry object.
- `dim`: The dimension corresponding to the spatial axes (must have length 3).
- `R_polarity`: The polarity mapping. Defaults to `geo.R_Nominal_RPS`.
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

Converts physical gradients (DCS) back to the nominal sequence coordinate system.
This performs the exact inverse operation of `grad_Nominal2DCS`.
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