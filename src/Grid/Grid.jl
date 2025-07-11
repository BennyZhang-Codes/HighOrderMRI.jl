
export Grid
Base.@kwdef struct Grid{T<:AbstractFloat}
    nX::Int64 = 0
    nY::Int64 = 0
    nZ::Int64 = 0
    Δx::Real = 1.0
    Δy::Real = 1.0
    Δz::Real = 1.0
    x::AbstractVector{T} = [0.]
    y::AbstractVector{T} = [0.]
    z::AbstractVector{T} = [0.]
    matrixSize::Tuple{Int64,Int64,Int64} = (nX, nY, nZ)
    resolution::Tuple{Real,Real,Real} = (Δx, Δy, Δz)
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




