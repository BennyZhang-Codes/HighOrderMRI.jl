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
- `T_DCS`: (`::Vector{T}`, `[m]`) Physical center position of the slice in DCS.
- `R_RPS_PCS`: (`::Matrix{T}`) Rotation matrix from logical RPS to Patient PCS.
- `R_PCS_DCS`: (`::Matrix{T}`) Rotation matrix from Patient PCS to Device DCS.
- `R_RPS_DCS`: (`::Matrix{T}`) Rotation matrix from logical RPS to Device DCS.
- `Idx_Slice`: (`::Int`) 1-based index of the current slice/slab in the scan.
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
    Idx_Slice       :: D
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
    Idx_Slice       :: D                      ,
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
        Idx_Slice,
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
    println(io, "Idx_Slice       : ", geo.Idx_Slice                 )
end