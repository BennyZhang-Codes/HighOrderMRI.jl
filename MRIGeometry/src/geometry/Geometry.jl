"""
    geo = Geometry(SystemVendor, Dimension, FOV, MatrixSize, PatientPosition, T_PCS, R_RPS_PCS)

# Description
    The `Geometry{T}` struct holds information about the spatial configuration of an MRI scan, 
    supporting generic floating-point precision (e.g., Float32 or Float64), and handles 
    vendor-specific device coordinate transformations.

# Arguments
- `SystemVendor`: (`::String`) Scanner manufacturer (e.g., "Siemens", "GE", "Philips").
- `Dimension`: (`::Int`) Scan dimensionality; 2 for 2D or 3 for 3D acquisitions.
- `FOV`: (`::Vector{T}`, `[m]`) Field of view in the order [Read, Phase, Slice].
- `MatrixSize`: (`::Vector{Int}`) Number of voxels per dimension [Read, Phase, Slice].
- `PatientPosition`: (`::String`) Patient orientation string (e.g., "HFS", "FFP", etc.).
- `T_PCS`: (`::Vector{T}`, `[m]`) Physical center position of the slice in PCS.
- `R_RPS_PCS`: (`::Matrix{T}`) Rotation matrix from logical RPS to Patient PCS.
"""
mutable struct Geometry{T <: AbstractFloat, D <: Integer}
    SystemVendor    :: String
    Dimension       :: D
    FOV             :: Vector{T}
    MatrixSize      :: Vector{D}
    PatientPosition :: String
    T_PCS           :: Vector{T}
    T_DCS           :: Vector{T}
    R_RPS_PCS       :: Matrix{T}
    R_PCS_DCS       :: Matrix{T}
    R_RPS_DCS       :: Matrix{T}
    # Normal          :: Vector{T}
    # Position        :: Vector{T}
    # InPlaneRot      :: T
    # RotMatrix       :: Matrix{T}
end

"""
    Geometry(SystemVendor, Dimension, FOV, MatrixSize, PatientPosition, T_PCS, R_RPS_PCS)

Outer constructor for `Geometry`.
It automatically infers the highest precision floating-point type `T` from the inputs, 
selects the vendor-specific transformation matrix, and pre-calculates the DCS mapping.
"""
function Geometry(
    SystemVendor    :: String                 ,
    Dimension       :: D                      , 
    FOV             :: AbstractVector{T}      , 
    MatrixSize      :: AbstractVector{D}      , 
    PatientPosition :: String                 , 
    T_PCS           :: AbstractVector{T}      ,
    R_RPS_PCS       :: AbstractMatrix{T}      ,
    # Normal          :: AbstractVector{<:Real} , 
    # Position        :: AbstractVector{<:Real} , 
    # InPlaneRot      :: Real                   , 
    # RotMatrix       :: AbstractMatrix{<:Real} ,
    ) where {T <: AbstractFloat, D <: Integer}
    SystemVendor = uppercase(SystemVendor)

    R_PCS_DCS   = MRIGeometry.PCS2DCS.TRANSFORMATIONS[SystemVendor][PatientPosition]
    R_RPS_DCS   = R_PCS_DCS * R_RPS_PCS
    T_DCS       = R_PCS_DCS * T_PCS
    
    return Geometry{T, D}(
        SystemVendor,
        Dimension, 
        FOV, 
        MatrixSize, 
        PatientPosition, 
        T_PCS,
        T_DCS,
        R_RPS_PCS,
        R_PCS_DCS,
        R_RPS_DCS,
        # Normal, 
        # Position, 
        # InPlaneRot, 
        # RotMatrix,
    )
end

Base.show(io::IO, geo::Geometry{T, D}) where {T, D} = begin
    println(io, ">>> Geometry{$T, $D} <<<")
    println(io, "SystemVendor    : ", geo.SystemVendor              )
    println(io, "Dimension       : ", geo.Dimension                 )
    println(io, "FOV [mm]        : ", geo.FOV*1e3                   )
    println(io, "MatrixSize      : ", geo.MatrixSize                )
    println(io, "Voxel Size [mm] : ", geo.FOV ./ geo.MatrixSize*1e3 )
    println(io, "PatientPosition : ", geo.PatientPosition           )
    println(io, "T_PCS [mm]      : ", geo.T_PCS*1e3                 )
    println(io, "T_DCS [mm]      : ", geo.T_DCS*1e3                 )
    println(io, "R_RPS_PCS       : ", geo.R_RPS_PCS                 )
    println(io, "R_PCS_DCS       : ", geo.R_PCS_DCS                 )
    println(io, "R_RPS_DCS       : ", geo.R_RPS_DCS                 )
    # println(io, "Normal          : ", geo.Normal                    )
    # println(io, "Position        : ", geo.Position                  )
    # println(io, "InPlaneRot [rad]: ", geo.InPlaneRot                )
    # println(io, "RotMatrix       : ", geo.RotMatrix                 )
end



# """
#     get_plane_orientation(g::Geometry) -> Matrix{Float64}

# Computes a rotation matrix to align the imaging plane normal vector in PCS to the DCS.

# Throws an error if the normal vector is not normalized.
# """
# function get_plane_orientation(g::Geometry)
#     normval = norm(g.Normal)
#     abs(1 - normval) > 1e-3 && error("Normal vector is not normalized (‖Normal‖ = $normval)")

#     maindir = argmax(abs.(g.Normal))
#     init_mat = maindir == 1 ? [0 0 1; 0 1 0; -1 0 0] :
#                maindir == 2 ? [0 1 0; 0 0 1; 1 0 0] :
#                               I(3)

#     init_normal = zeros(3); init_normal[maindir] = 1
#     v = cross(init_normal, g.Normal)
#     s = norm(v)
#     c = dot(init_normal, g.Normal)

#     if s ≤ 1e-5
#         return c * init_mat
#     else
#         vx = [  0   -v[3]  v[2];
#                v[3]  0   -v[1];
#               -v[2] v[1]  0  ]
#         rot = I(3) + vx + vx * vx / (1 + c)
#         return rot * init_mat
#     end
# end

# """
#     get_inplane_rotation(g::Geometry{T}) where {T} -> Matrix{T}

# Returns the in-plane rotation matrix for a given `Geometry`.
# """
# function get_inplane_rotation(g::Geometry{T}) where {T}
#     θ = g.InPlaneRot
    
#     return [
#         -sin(θ)  cos(θ)  zero(T);
#         -cos(θ) -sin(θ)  zero(T);
#         zero(T) zero(T)  one(T)
#     ]
# end


# """
#     prs_to_pcs(g::Geometry) -> Matrix{T}

# Returns the combined rotation matrix from PRS (Plane Ref Sys) to PCS.
# """
# prs_to_pcs(g::Geometry) = get_plane_orientation(g) * get_inplane_rotation(g)


# """
#     pcs_to_xyz(g::Geometry) -> Matrix{T}

# Returns the transformation matrix from PCS to XYZ based on patient position.
# """
# pcs_to_xyz(g::Geometry) = get(MRIGeometry.PCS2DCS.TRANSFORMATIONS, g.PatientPosition) do
#     error("Unknown patient position: $(g.PatientPosition)")
# end


# """
#     prs_to_xyz(g::Geometry) -> Matrix{T}

# Returns the transformation matrix from PRS to XYZ.
# """
# prs_to_xyz(g::Geometry) = pcs_to_xyz(g) * prs_to_pcs(g)


# """
#     prs_to_xyz(g::Geometry) -> Matrix{T}

# Returns the fixed transformation matrix from RPS to PRS coordinates.
# """
# function rps_to_prs(g::Geometry{T}) where {T} 
#     return T.([0 1 0; 1 0 0; 0 0 -1])
# end

# """
#     rps_to_xyz(g::Geometry) -> Matrix{T}

# Returns the transformation matrix from RPS to XYZ coordinates.
# """
# rps_to_xyz(g::Geometry) = prs_to_xyz(g) * rps_to_prs(g)


# """
#     rps_point_to_xyz(g::Geometry, x_rps::AbstractArray{T, N}) where {T<:Real, N}

# Transforms an array of RPS (Read, Phase, Slice) logical coordinates to XYZ device coordinates 
# using the geometry's spatial mapping. Supports generic floating-point types (e.g., Float32, Float64).

# # Arguments
# - `g`: The `Geometry` structure containing rotation and position headers.
# - `x_rps`: An N-dimensional array where the last dimension must have a size of 3 (R, P, S).

# # Returns
# - An array of the same shape and element type `T` containing the absolute physical [X, Y, Z] coordinates.
# """
# function rps_point_to_xyz(g::Geometry, x_rps::AbstractArray{T, N}) where {T<:Real, N}
#     size(x_rps, N) == 3 || error("Last dimension must be 3 (R, P, S)")

#     # Retrieve and convert the rotation matrix and offset vector to match the input element type T.
#     # This prevents implicit type promotion (e.g., Float32 arrays promoting to Float64).
#     rot = Matrix{T}(rps_to_xyz(g))
#     offset_xyz = Vector{T}(pcs_to_xyz(g) * g.Position)

#     # Flatten the multi-dimensional coordinate grid to a (K, 3) matrix 
#     # to leverage highly optimized BLAS matrix multiplications
#     shape_flat = reshape(x_rps, :, 3)
    
#     # Batch affine transformation: Result = shape_flat * transpose(rot) .+ offset_xyz'
#     result = shape_flat * transpose(rot)
#     result .+= offset_xyz'
    
#     # Restore the original array dimensions and structure
#     return reshape(result, size(x_rps))
# end


# """
#     xyz_to_rps_point(g::Geometry, x_xyz::AbstractArray{T, N}) where {T<:Real, N}

# Transforms an array of XYZ device coordinates back to RPS (Read, Phase, Slice) logical 
# coordinates for a given Geometry. Supports generic floating-point types (e.g., Float32, Float64).

# # Arguments
# - `g`: The target `Geometry` structure to map into.
# - `x_xyz`: An N-dimensional array where the last dimension must have a size of 3 (X, Y, Z).

# # Returns
# - An array of the same shape and element type `T` containing the logical [R, P, S] coordinates.
# """
# function xyz_to_rps_point(g::Geometry, x_xyz::AbstractArray{T, N}) where {T<:Real, N}
#     size(x_xyz, N) == 3 || error("Last dimension must be 3 (X, Y, Z)")

#     # Retrieve and convert geometry parameters to type T to ensure full type stability
#     rot = Matrix{T}(rps_to_xyz(g))
#     offset_xyz = Vector{T}(pcs_to_xyz(g) * g.Position)

#     # Flatten the multi-dimensional coordinate grid to a (K, 3) matrix
#     shape_flat = reshape(x_xyz, :, 3)
    
#     # Inverse affine transformation utilizing orthogonal matrix inversion property (rot^-1 = rot')
#     # Batch calculation: Result = (shape_flat .- offset_xyz') * rot
#     result = (shape_flat .- offset_xyz') * rot
    
#     # Restore the original array dimensions and structure
#     return reshape(result, size(x_xyz))
# end



# # ==============================================================================
# # LPH (Left, Posterior, Head / Superior) Coordinate System
# # The international DICOM standard space (same as LPS).
# # ==============================================================================

# """
#     pcs_to_lph(g::Geometry{T}) where {T} -> Matrix{T}

# Returns the transformation matrix from Siemens PCS (Right, Posterior, Head) 
# to international LPH (Left, Posterior, Head).
# """
# function pcs_to_lph(g::Geometry{T}) where {T}
#     # X axis (Right -> Left) is flipped, Y and Z axes remain the same.
#     return T.([-1  0  0; 
#                 0  1  0; 
#                 0  0  1])
# end

# """
#     rps_to_lph(g::Geometry) -> Matrix{T}

# Returns the transformation matrix from logical RPS to absolute LPH space.
# """
# function rps_to_lph(g::Geometry)
#     # RPS -> PRS -> PCS -> LPH
#     return pcs_to_lph(g) * prs_to_pcs(g) * rps_to_prs(g)
# end

# """
#     rps_point_to_lph(g::Geometry, x_rps::AbstractArray{T, N}) where {T<:Real, N}

# Transforms local RPS voxel grid coordinates directly into LPH (LPS) absolute 
# physical coordinates.
# """
# function rps_point_to_lph(g::Geometry, x_rps::AbstractArray{T, N}) where {T<:Real, N}
#     size(x_rps, N) == 3 || error("Last dimension must be 3 (R, P, S)")

#     rot_lph = Matrix{T}(rps_to_lph(g))
    
#     offset_lph = Vector{T}(pcs_to_lph(g) * g.Position)

#     shape_flat = reshape(x_rps, :, 3)
#     result = shape_flat * transpose(rot_lph)
#     result .+= offset_lph'
    
#     return reshape(result, size(x_rps))
# end

