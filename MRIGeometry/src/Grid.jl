"""
    gen_RPS_grid(fov::Vector{T}, matrix_size::Vector{D}) where {T <: AbstractFloat, D <: Integer}

    gen_RPS_grid(geo::Geometry)

Generate centred voxel coordinates in metres as an `Array{T,4}` of shape
`(nR,nP,nS,3)`. For each axis, spacing is FOV/N and coordinate j is
`(j-1-(N-1)/2)*spacing`. A singleton axis is centred at zero. Supply three
positive FOV values and three positive integer counts; only vector lengths
are explicitly checked. The Geometry overload uses its FOV and MatrixSize.
This coordinate array is separate from the root package's encoding `Grid`;
changing it does not modify an existing NFFT operator.
"""
function gen_RPS_grid(fov::Vector{T}, matrix_size::Vector{D}) where {T <: AbstractFloat, D <: Integer}
    if length(fov) != 3 || length(matrix_size) != 3
        error("Both fov and matrix_size must be 3-element vectors.")
    end

    # Convert to Float64
    fov = T.(fov)
    matrix_size = D.(matrix_size)

    spacing = fov ./ matrix_size

    r = ((0:matrix_size[1]-1) .- (matrix_size[1]-1)/2) .* spacing[1]
    p = ((0:matrix_size[2]-1) .- (matrix_size[2]-1)/2) .* spacing[2]
    s = ((0:matrix_size[3]-1) .- (matrix_size[3]-1)/2) .* spacing[3]

    rr = reshape(r, :, 1, 1)
    pp = reshape(p, 1, :, 1)
    ss = reshape(s, 1, 1, :)

    # Broadcast to form 3D grid
    grid = Array{T, 4}(undef, matrix_size[1], matrix_size[2], matrix_size[3], 3)
    grid[:, :, :, 1] .= rr
    grid[:, :, :, 2] .= pp
    grid[:, :, :, 3] .= ss

    return grid
end

function gen_RPS_grid(geo::Geometry)
    return gen_RPS_grid(geo.FOV, geo.MatrixSize)
end



"""
    RPS2DCS(geo::Geometry, grid_rps::AbstractArray{T, N}) where {T <: Real, N}

Transform RPS coordinates to DCS, preserving array shape. The last axis
must contain three coordinate components; floating-point coordinates use
metres. On column vectors the mapping is `geo.R_RPS_DCS*x+geo.T_DCS`.
Geometry matrices and translations are converted to the input coordinate
element type T. Use floating-point inputs to avoid integer conversion loss.
The inverse functions use transpose, which assumes orthogonal rotations.
DCS results inherit the vendor-mapping limitations of Geometry.
"""
function RPS2DCS(geo::Geometry, grid_rps::AbstractArray{T, N}) where {T<:Real, N}
    size(grid_rps, N) == 3 || error("The last dimension of the grid must be 3 (R, P, S)")

    # 1. Retrieve the pre-calculated rotation matrix and translation vector
    R = Matrix{T}(geo.R_RPS_DCS)
    T_offset = Vector{T}(geo.T_DCS)

    # 2. Flatten the spatial dimensions to shape (K, 3)
    # This allows us to leverage highly optimized BLAS matrix multiplications
    grid_flat = reshape(grid_rps, :, 3)

    # 3. Batch affine transformation: X_DCS = R * X_RPS + T
    # In matrix form for K points: result = grid_flat * R' .+ T_offset'
    result = grid_flat * transpose(R)
    result .+= T_offset'

    # 4. Restore the original array dimensions (e.g., Nx x Ny x Nz x 3)
    return reshape(result, size(grid_rps))
end


"""
    DCS2RPS(geo::Geometry, grid_dcs::AbstractArray{T, N}) where {T <: Real, N}

Transform DCS coordinates to RPS, preserving array shape. The last axis
must contain three coordinate components; floating-point coordinates use
metres. On column vectors the mapping is `transpose(geo.R_RPS_DCS)*(x-geo.T_DCS)`.
Geometry matrices and translations are converted to the input coordinate
element type T. Use floating-point inputs to avoid integer conversion loss.
The inverse functions use transpose, which assumes orthogonal rotations.
DCS results inherit the vendor-mapping limitations of Geometry.
"""
function DCS2RPS(geo::Geometry, grid_dcs::AbstractArray{T, N}) where {T<:Real, N}
    size(grid_dcs, N) == 3 || error("The last dimension of the grid must be 3 (X, Y, Z)")

    # 1. Retrieve the pre-calculated rotation matrix and translation vector
    R = Matrix{T}(geo.R_RPS_DCS)
    T_offset = Vector{T}(geo.T_DCS)

    # 2. Flatten the spatial dimensions to shape (K, 3)
    grid_flat = reshape(grid_dcs, :, 3)

    # 3. Inverse affine transformation: X_RPS = R^-1 * (X_DCS - T)
    # Utilizing the orthogonal matrix property: R^-1 = transpose(R)
    # Batch calculation: Result = (grid_flat .- T_offset') * R
    result = (grid_flat .- T_offset') * R

    # 4. Restore the original array dimensions
    return reshape(result, size(grid_dcs))
end


"""
    RPS2PCS(geo::Geometry, grid_rps::AbstractArray{T, N}) where {T <: Real, N}

Transform RPS coordinates to PCS, preserving array shape. The last axis
must contain three coordinate components; floating-point coordinates use
metres. On column vectors the mapping is `geo.R_RPS_PCS*x+geo.T_PCS`.
Geometry matrices and translations are converted to the input coordinate
element type T. Use floating-point inputs to avoid integer conversion loss.
The inverse functions use transpose, which assumes orthogonal rotations.
DCS results inherit the vendor-mapping limitations of Geometry.
"""
function RPS2PCS(geo::Geometry, grid_rps::AbstractArray{T, N}) where {T<:Real, N}
    size(grid_rps, N) == 3 || error("The last dimension of the grid must be 3 (R, P, S)")

    # 1. Retrieve the pre-calculated rotation matrix and translation vector for PCS
    R = Matrix{T}(geo.R_RPS_PCS)
    T_offset = Vector{T}(geo.T_PCS)

    # 2. Flatten the spatial dimensions to shape (K, 3)
    # This allows us to leverage highly optimized BLAS matrix multiplications
    grid_flat = reshape(grid_rps, :, 3)

    # 3. Batch affine transformation: X_PCS = R * X_RPS + T
    # In matrix form for K points: result = grid_flat * R' .+ T_offset'
    result = grid_flat * transpose(R)
    result .+= T_offset'

    # 4. Restore the original array dimensions
    return reshape(result, size(grid_rps))
end


"""
    PCS2RPS(geo::Geometry, grid_pcs::AbstractArray{T, N}) where {T <: Real, N}

Transform PCS coordinates to RPS, preserving array shape. The last axis
must contain three coordinate components; floating-point coordinates use
metres. On column vectors the mapping is `transpose(geo.R_RPS_PCS)*(x-geo.T_PCS)`.
Geometry matrices and translations are converted to the input coordinate
element type T. Use floating-point inputs to avoid integer conversion loss.
The inverse functions use transpose, which assumes orthogonal rotations.
DCS results inherit the vendor-mapping limitations of Geometry.
"""
function PCS2RPS(geo::Geometry, grid_pcs::AbstractArray{T, N}) where {T<:Real, N}
    size(grid_pcs, N) == 3 || error("The last dimension of the grid must be 3 (PCS components)")

    # 1. Retrieve the pre-calculated rotation matrix and translation vector for PCS
    R = Matrix{T}(geo.R_RPS_PCS)
    T_offset = Vector{T}(geo.T_PCS)

    # 2. Flatten the spatial dimensions to shape (K, 3)
    grid_flat = reshape(grid_pcs, :, 3)

    # 3. Inverse affine transformation: X_RPS = R^-1 * (X_PCS - T)
    # Utilizing the orthogonal matrix property: R^-1 = transpose(R)
    # Batch calculation: Result = (grid_flat .- T_offset') * R
    result = (grid_flat .- T_offset') * R

    # 4. Restore the original array dimensions
    return reshape(result, size(grid_pcs))
end
