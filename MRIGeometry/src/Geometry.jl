# src/geometry.jl

"""
    geo = Geometry(Dimension, FOV, MatrixSize, Normal, Position, InPlaneRot, PatientPosition, RotMatrix)

# Description
    The `Geometry` struct holds information about the spatial configuration of an MRI scan.

# Arguments
- `Dimension`: (`::Int`) Scan dimensionality; 2 for 2D or 3 for 3D acquisitions
- `FOV`: (`::Vector{Float64}`, `[m]`) Field of view in the order [Read, Phase, Slice]
- `MatrixSize`: (`::Vector{<:Real}`) Number of voxels per dimension [Read, Phase, Slice]
- `Normal`: (`::Vector{Float64}`) Normal vector of the imaging slice in the PCS (Patient Coordinate System)
- `Position`: (`::Vector{Float64}`, `[m]`) Physical center position of the slice in PCS
- `InPlaneRot`: (`::Float64`, `[rad]`) In-plane rotation angle
- `PatientPosition`: (`::String`) Patient orientation string (e.g., "HFS", "FFP", etc.)
- `RotMatrix`: (`::Matrix{Float64}`) 3x3 matrix mapping RPS to XYZ device coordinates
"""
mutable struct Geometry
    Dimension       :: Int
    FOV             :: Vector{Float64}
    MatrixSize      :: Vector{<:Real}
    Normal          :: Vector{Float64}
    Position        :: Vector{Float64}
    InPlaneRot      :: Float64
    PatientPosition :: String
    RotMatrix       :: Matrix{Float64}
end

Base.show(io::IO, geo::Geometry) = begin
    println(io, ">>> Geometry <<<")
    println(io, "Dimension       : ", geo.Dimension                 )
    println(io, "FOV [mm]        : ", geo.FOV*1e3                   )
    println(io, "MatrixSize      : ", geo.MatrixSize                )
    println(io, "Voxel Size [mm] : ", geo.FOV ./ geo.MatrixSize*1e3 )
    println(io, "Normal          : ", geo.Normal                    )
    println(io, "Position        : ", geo.Position                  )
    println(io, "InPlaneRot [rad]: ", geo.InPlaneRot                )
    println(io, "PatientPosition : ", geo.PatientPosition           )
    println(io, "RotMatrix       : ", geo.RotMatrix                 )
end

"""
    get_plane_orientation(g::Geometry) -> Matrix{Float64}

Computes a rotation matrix to align the imaging plane normal vector in PCS to the DCS.

Throws an error if the normal vector is not normalized.
"""
function get_plane_orientation(g::Geometry)
    normval = norm(g.Normal)
    abs(1 - normval) > 1e-3 && error("Normal vector is not normalized (‖Normal‖ = $normval)")

    maindir = argmax(abs.(g.Normal))
    init_mat = maindir == 1 ? [0 0 1; 0 1 0; -1 0 0] :
               maindir == 2 ? [0 1 0; 0 0 1; 1 0 0] :
                              I(3)

    init_normal = zeros(3); init_normal[maindir] = 1
    v = cross(init_normal, g.Normal)
    s = norm(v)
    c = dot(init_normal, g.Normal)

    if s ≤ 1e-5
        return c * init_mat
    else
        vx = [  0   -v[3]  v[2];
               v[3]  0   -v[1];
              -v[2] v[1]  0  ]
        rot = I(3) + vx + vx * vx / (1 + c)
        return rot * init_mat
    end
end

"""
    get_inplane_rotation(g::Geometry) -> Matrix{Float64}

Returns the in-plane rotation matrix for a given `Geometry`.
"""
function get_inplane_rotation(g::Geometry)
    θ = g.InPlaneRot
    return [
        -sin(θ)  cos(θ)  0;
        -cos(θ) -sin(θ)  0;
         0       0      1
    ]
end


"""
    prs_to_pcs(g::Geometry) -> Matrix{Float64}

Returns the combined rotation matrix from PRS (Plane Ref Sys) to PCS.
"""
prs_to_pcs(g::Geometry) = get_plane_orientation(g) * get_inplane_rotation(g)


"""
    pcs_to_xyz(g::Geometry) -> Matrix{Float64}

Returns the transformation matrix from PCS to XYZ based on patient position.
"""
pcs_to_xyz(g::Geometry) = get(MRIGeometry.PCS.TRANSFORMATIONS, g.PatientPosition) do
    error("Unknown patient position: $(g.PatientPosition)")
end


"""
    prs_to_xyz(g::Geometry) -> Matrix{Float64}

Returns the transformation matrix from PRS to XYZ.
"""
prs_to_xyz(g::Geometry) = pcs_to_xyz(g) * prs_to_pcs(g)


"""
    rps_to_prs() -> Matrix{Float64}

Returns the fixed transformation matrix from RPS to PRS coordinates.
"""
rps_to_prs() = [0 1 0; 1 0 0; 0 0 -1]


"""
    rps_to_xyz(g::Geometry) -> Matrix{Float64}

Returns the transformation matrix from RPS to XYZ coordinates.
"""
rps_to_xyz(g::Geometry) = prs_to_xyz(g) * rps_to_prs()


"""
    rps_point_to_xyz(g::Geometry, x_rps::AbstractArray{Float64, N}) -> Array{Float64, N}

Transforms an array of RPS coordinates to XYZ using the geometry's spatial mapping.
"""
function rps_point_to_xyz(g::Geometry, x_rps::AbstractArray{Float64, N}) where N
    size(x_rps, N) == 3 || error("Last dimension must be 3 (R, P, S)")

    rot = rps_to_xyz(g)
    offset_xyz = pcs_to_xyz(g) * g.Position

    shape_flat = reshape(x_rps, :, 3)
    result = shape_flat * transpose(rot)
    result .+= offset_xyz'
    return reshape(result, size(x_rps))
end