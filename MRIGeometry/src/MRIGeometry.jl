module MRIGeometry

using LinearAlgebra
using Statistics
using PyPlot
using PyCall
using Interpolations

using MRIBase

using NIfTI


include("PCS2DCS.jl")

include("geometry/geo.jl")


include("Grid.jl")
export gen_RPS_grid, RPS2PCS, PCS2RPS, RPS2DCS, DCS2RPS

include("GradConversion.jl")
export grad_DCS2RPS, grad_RPS2DCS

include("plot/plot.jl")

include("Resample.jl")
export resample

include("ExportNIfTI.jl")
export export_nifti

#Package version
using Pkg
__VERSION__ = VersionNumber(Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"])

end # module MRIGeometry
