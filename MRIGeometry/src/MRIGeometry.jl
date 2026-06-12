module MRIGeometry

using LinearAlgebra
using Statistics
using PyPlot
using PyCall
using Interpolations

using MRIBase

include("PCS2DCS.jl")

include("geometry/geo.jl")


include("Grid.jl")
export gen_RPS_grid, RPS2PCS, PCS2RPS, RPS2DCS, DCS2RPS

include("plot/plot.jl")

include("Resample.jl")
export resample

#Package version
using Pkg
__VERSION__ = VersionNumber(Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"])

end # module MRIGeometry
