
export Grid

"""
    Grid(nX, nY, nZ, Δx, Δy, Δz;
         exchange_xy=false,
         reverse_x=false,
         reverse_y=false,
         reverse_z=false)

Create the physical Cartesian reconstruction grid used by the high-order
encoding operators.

Voxel centres along an axis of length `N` and spacing `Δ` follow
`r[i] = (i - (N + 1) / 2) * Δ`. Odd dimensions therefore contain a voxel at
zero, while even dimensions are centred on half-voxel coordinates.

`exchange_xy` and the `reverse_*` keywords transform the coordinate vectors
only. Apply the same transformations to images, masks, field maps, and coil
sensitivity maps before constructing an encoding operator.
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

