module MRIGeometry

using LinearAlgebra
using Statistics
using PyPlot

include("PCS.jl")

include("Grid.jl")
export generate_rps_grid


include("Geometry.jl")
export Geometry

include("plot/plt_grid.jl")
export plt_grid




#Package version
using Pkg
__VERSION__ = VersionNumber(Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"])

end # module MRIGeometry
