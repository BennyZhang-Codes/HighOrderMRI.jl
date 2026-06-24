"""
    grad_RPS2DCS(G_RPS::AbstractArray{T, N}, geo::Geometry; dim::Int = N == 2 ? 1 : N) where {T, N}

Converts nominal gradients from the logical coordinate system (RPS: Read, Phase, Slice) 
to the physical device coordinate system (DCS: Physical X, Y, Z).

Supports arrays of arbitrary dimensions (e.g., [3, Nt] matrices, [nDynamic, nSample, 3] tensors).

# Arguments
- `G_RPS`: N-dimensional nominal gradient array.
- `geo`: Geometry object containing `R_RPS_DCS`.
- `dim`: The dimension corresponding to the spatial axes (must have length 3). 
         Defaults to 1 for 2D matrices, and N for higher-dimensional arrays.
"""
function grad_RPS2DCS(G_RPS::AbstractArray{T, N}, geo::Geometry; dim::Int = N == 2 ? 1 : N) where {T, N}
    R = geo.R_RPS_DCS
    size(G_RPS, dim) == 3 || throw(ArgumentError("The specified spatial dimension (dim=$dim) must have length 3. Found length $(size(G_RPS, dim))."))
    
    if dim == 1
        # Fast path: spatial axis is already contiguous in memory
        res = R * reshape(G_RPS, 3, :)
        return reshape(res, size(G_RPS))
    else
        # Dynamic permutation for arbitrary dimensions
        perm = (dim, (1:dim-1)..., (dim+1:N)...)
        inv_perm = ((2:dim)..., 1, (dim+1:N)...)
        
        G_perm = permutedims(G_RPS, perm)
        res = R * reshape(G_perm, 3, :)
        return permutedims(reshape(res, size(G_perm)), inv_perm)
    end
end

"""
    grad_DCS2RPS(G_DCS::AbstractArray{T, N}, geo::Geometry; dim::Int = N == 2 ? 1 : N) where {T, N}

Converts actual physical gradients or trajectories from the device coordinate system (DCS: X, Y, Z) 
back to the logical coordinate system (RPS).

Supports arrays of arbitrary dimensions.

# Arguments
- `G_DCS`: N-dimensional physical gradient or trajectory array.
- `geo`: Geometry object containing `R_RPS_DCS`.
- `dim`: The dimension corresponding to the spatial axes (must have length 3). 
         Defaults to 1 for 2D matrices, and N for higher-dimensional arrays.
"""
function grad_DCS2RPS(G_DCS::AbstractArray{T, N}, geo::Geometry; dim::Int = N == 2 ? 1 : N) where {T, N}
    # The inverse of an orthogonal matrix is its transpose.
    R_inv = transpose(geo.R_RPS_DCS)
    size(G_DCS, dim) == 3 || throw(ArgumentError("The specified spatial dimension (dim=$dim) must have length 3. Found length $(size(G_DCS, dim))."))
    
    if dim == 1
        # Fast path: spatial axis is already contiguous in memory
        res = R_inv * reshape(G_DCS, 3, :)
        return reshape(res, size(G_DCS))
    else
        # Dynamic permutation for arbitrary dimensions
        perm = (dim, (1:dim-1)..., (dim+1:N)...)
        inv_perm = ((2:dim)..., 1, (dim+1:N)...)
        
        G_perm = permutedims(G_DCS, perm)
        res = R_inv * reshape(G_perm, 3, :)
        return permutedims(reshape(res, size(G_perm)), inv_perm)
    end
end


function grad_DCS2RPS_nominal(G_DCS::AbstractArray{T, N}, geo::Geometry; dim::Int = N == 2 ? 1 : N) where {T, N}
    # The inverse of an orthogonal matrix is its transpose.

    r = [
        -1.0 0.0 0.0;
        0.0 1.0 0.0;
        0.0 0.0 1.0
    ]

    R_inv = transpose(r) * transpose(geo.R_PCS_DCS) 

    # R_inv = transpose(geo.R_RPS_DCS)
    size(G_DCS, dim) == 3 || throw(ArgumentError("The specified spatial dimension (dim=$dim) must have length 3. Found length $(size(G_DCS, dim))."))
    
    if dim == 1
        # Fast path: spatial axis is already contiguous in memory
        res = R_inv * reshape(G_DCS, 3, :)
        return reshape(res, size(G_DCS))
    else
        # Dynamic permutation for arbitrary dimensions
        perm = (dim, (1:dim-1)..., (dim+1:N)...)
        inv_perm = ((2:dim)..., 1, (dim+1:N)...)
        
        G_perm = permutedims(G_DCS, perm)
        res = R_inv * reshape(G_perm, 3, :)
        return permutedims(reshape(res, size(G_perm)), inv_perm)
    end
end
