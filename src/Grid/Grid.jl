
export Grid

"""
    Grid{T}

Physical Cartesian reconstruction grid with floating-point coordinates. Fields
store the three axis sizes and spacings, flattened coordinate vectors, and
`matrixSize`/`resolution` tuples. Use the dimension-and-spacing constructor
for a centred grid. The keyword field constructor does not validate consistency
between manually supplied coordinates and metadata.

Coordinates are in metres when used with phase coefficients in cycles/m^p.
Low-rank NFFT execution requires the default centred Cartesian geometry on
its active Fourier axes; changing `x`, `y`, or `z` alone does not change the
NFFT plan. Explicit operators evaluate the supplied coordinates directly.
"""
Base.@kwdef struct Grid{T<:AbstractFloat}
    nX::Int64 = 0
    nY::Int64 = 0
    nZ::Int64 = 0
    Δx::T = one(T)
    Δy::T = one(T)
    Δz::T = one(T)
    x::AbstractVector{T} = T[0]
    y::AbstractVector{T} = T[0]
    z::AbstractVector{T} = T[0]
    matrixSize::NTuple{3,Int64} = (nX, nY, nZ)
    resolution::NTuple{3,T} = (Δx, Δy, Δz)
end

@functor Grid

Base.show(io::IO, b::Grid) = begin
    Δx = round(b.Δx*1e3, digits=2)
    Δy = round(b.Δy*1e3, digits=2)
    Δz = round(b.Δz*1e3, digits=2)
	print(io, "Grid [ MatrixSize: $(b.nX) x $(b.nY) x $(b.nZ), Resolution: $(Δx) x $(Δy) x $(Δz) mm³ ]")
end


"""
    Grid(nX::Int64,
        nY::Int64,
        nZ::Int64,
        Δx::T,
        Δy::T,
        Δz::T;
        exchange_xy::Bool = false,
        reverse_x::Bool = false,
        reverse_y::Bool = false,
        reverse_z::Bool = false) where T <: AbstractFloat

Create a centred Cartesian grid. All spacings must have the same floating-point
precision `T`. Axis centres follow `r[j] = (j - (N + 1)/2) * Δ`; even sizes use
half-voxel centres. The x index varies fastest in the flattened coordinate arrays.

`exchange_xy` swaps coordinate vectors; each `reverse_*` reverses that flattened
coordinate vector. These switches do not permute image, mask, field-map or
coil-map arrays and do not update grid dimensions/spacings. Supply all physical
inputs in a consistent coordinate frame. For low-rank NFFT use the default
untransformed active axes. Positive dimensions/spacings are caller requirements;
the constructor does not enforce them.
"""
function Grid(
    nX::Int64, 
    nY::Int64, 
    nZ::Int64, 
    Δx::T, 
    Δy::T, 
    Δz::T;
    exchange_xy::Bool=false,
    reverse_x::Bool=false,
    reverse_y::Bool=false,
    reverse_z::Bool=false,
    ) where {T<:AbstractFloat}
    # x up->down, y left->right
    x, y, z = 1:nX, 1:nY, 1:nZ
    x, y = vec(x .+ y'*0.0), vec(x*0.0 .+ y') 
    x, y, z = vec(x .+ z'*0.0), vec(y .+ z'*0.0), vec(x*0.0 .+ z') #grid points
    x, y, z = x.-(nX+1)/2, y.-(nY+1)/2, z.-(nZ+1)/2
    x, y, z = x * Δx, y * Δy, z * Δz
    if exchange_xy
        x, y = y, x
    end
    if reverse_x
        x = reverse(x)
    end
    if reverse_y
        y = reverse(y)
    end
    if reverse_z
        z = reverse(z)
    end
    return Grid(nX=nX, nY=nY, nZ=nZ, Δx=Δx, Δy=Δy, Δz=Δz, x=T.(x), y=T.(y), z=T.(z))
end

struct_to_dict(grid::Grid) = Dict(
    "schema_version" => 1,
    "nX" => grid.nX,
    "nY" => grid.nY,
    "nZ" => grid.nZ,
    "delta_x" => grid.Δx,
    "delta_y" => grid.Δy,
    "delta_z" => grid.Δz,
    "x" => Array(grid.x),
    "y" => Array(grid.y),
    "z" => Array(grid.z),
    "matrix_size" => collect(grid.matrixSize),
    "resolution" => collect(grid.resolution),
)

"""
    Grid(d::AbstractDict;
        T::Type{<:AbstractFloat} = Float64)

Restore a grid from a dictionary using the fields written by
`struct_to_dict(grid)`: `nX`, `nY`, `nZ`, `delta_x`, `delta_y`, `delta_z`,
`x`, `y`, and `z`. Scalars may also be one-element arrays. Coordinates and
spacings are converted to the requested floating-point type `T`; size and
resolution tuples are reconstructed from those scalar fields.

The dictionary must contain consistent full-length coordinate vectors. This
constructor does not validate a schema version or the Cartesian NFFT geometry.
"""
function Grid(
    d::AbstractDict;
    T::Type{<:AbstractFloat}=Float64,
)
    _scalar = x -> x isa AbstractArray ? only(x) : x

    nX = Int64(_scalar(d["nX"]))
    nY = Int64(_scalar(d["nY"]))
    nZ = Int64(_scalar(d["nZ"]))
    Δx = T(_scalar(d["delta_x"]))
    Δy = T(_scalar(d["delta_y"]))
    Δz = T(_scalar(d["delta_z"]))

    return Grid{T}(
        nX=nX,
        nY=nY,
        nZ=nZ,
        Δx=Δx,
        Δy=Δy,
        Δz=Δz,
        x=T.(vec(d["x"])),
        y=T.(vec(d["y"])),
        z=T.(vec(d["z"])),
        matrixSize=(nX, nY, nZ),
        resolution=(Δx, Δy, Δz),
    )
end

