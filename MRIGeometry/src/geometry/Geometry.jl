"""
    Geometry{T <: AbstractFloat, D <: Integer}

Mutable acquisition geometry containing RPS (read/phase/slice), PCS
(patient LPS) and DCS (device) transforms. FOV and translations are in metres;
MatrixSize has three positive counts and rotations act on column coordinates.
`T_PCS` is the volume or slice centre, not its first voxel centre. `Idx_Slice`
is a one-based acquisition label, not a physical sorting coordinate.

The object stores both original and derived transforms. Direct field mutation
does not recompute dependent fields, and the constructor does not fully check
positive sizes, orthogonality or consistency. Keep all fields consistent.
GE, PHILIPS and UIH entries in RotMat are placeholders; their default device
mapping and gradient polarity are not verified vendor support. Siemens
conventions also require confirmation against the acquisition's coordinate
definition. See [Geometry](/reference/geometry) before exporting physical data.
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
    R_Nominal_RPS   :: Matrix{T}
    Idx_Slice       :: D
end

"""
    Geometry(SystemVendor::String, Dimension::D, FOV::AbstractVector{T}, MatrixSize::AbstractVector{D}, PatientPosition::String, T_PCS::AbstractVector{T}, R_RPS_PCS::AbstractMatrix{T}, Idx_Slice::D) where {T <: AbstractFloat, D <: Integer}

    Geometry(d::AbstractDict; T::Type{<:AbstractFloat} = Float64, D::Type{<:Integer} = Int)

Construct a geometry from physical metadata or a dictionary of stored fields.
In the positional form, FOV and T_PCS have length three, R_RPS_PCS is 3×3,
and MatrixSize has length three. FOV, centre and rotation must share the same
floating-point type T; dimension, matrix counts and Idx_Slice share integer
type D. There is no automatic highest-precision promotion. Vendor names are
uppercased; patient-position keys must match the vendor table. Derived fields
are `R_RPS_DCS=R_PCS_DCS*R_RPS_PCS` and `T_DCS=R_PCS_DCS*T_PCS`.

The dictionary form defaults to Float64/Int and reads all stored fields,
including derived transforms, rather than recalculating them. Required keys
are the Geometry field names. It does not validate that those fields agree.
The positional constructor uses currently unverified placeholder device
matrices for GE/PHILIPS/UIH. Neither form proves physical vendor correctness.
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

    R_PCS_DCS   = MRIGeometry.RotMat.PCS2DCS[SystemVendor][PatientPosition]
    R_RPS_DCS   = R_PCS_DCS * R_RPS_PCS
    T_DCS       = R_PCS_DCS * T_PCS

    # Extract baseline polarity mapping automatically based on vendor
    R_Nominal_RPS = MRIGeometry.RotMat.NOMINAL2RPS[SystemVendor]

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
        R_Nominal_RPS,
        Idx_Slice,
    )
end


function Geometry(
    d::AbstractDict;
    T::Type{<:AbstractFloat} = Float64,
    D::Type{<:Integer} = Int,
)
    _scalar = x -> x isa AbstractArray ? only(x) : x
    
    Geometry{T,D}(
        String(_scalar(d["SystemVendor"])),
        D(_scalar(d["Dimension"])),
        T.(vec(d["FOV"])),
        D.(vec(d["MatrixSize"])),
        String(_scalar(d["PatientPosition"])),
        T.(vec(d["T_PCS"])),
        T.(vec(d["T_DCS"])),
        T.(d["R_RPS_PCS"]),
        T.(d["R_PCS_DCS"]),
        T.(d["R_RPS_DCS"]),
        T.(d["R_Nominal_RPS"]),
        D(_scalar(d["Idx_Slice"])),
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
    println(io, "R_Nominal_RPS   : ", geo.R_Nominal_RPS             )
    println(io, "Idx_Slice       : ", geo.Idx_Slice                 )
end