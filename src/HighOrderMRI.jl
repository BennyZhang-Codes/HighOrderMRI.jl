"""
    HighOrderMRI.jl

A Julia package for high-order MRI reconstruction.
"""
module HighOrderMRI

using MRIReco
using RegularizedLeastSquares
using LinearOperators
using LinearAlgebra
using PyPlot

# using Parameters
using CUDA

using ProgressMeter

using Interpolations
using Statistics

using AbstractNFFTs, NFFTTools
using FFTW: fftshift, ifftshift, fft, ifft

import ImageTransformations: imresize
import DSP: conv               
import Functors: @functor

include("utils/utils.jl")

include("Grid/Grid.jl")

include("SphericalHarmonics/SphericalHarmonics.jl")

include("EncodingOperator/EncodingOperator.jl")

include("Recon/Recon.jl")

include("Synchronization/Synchronization.jl")

include("plot/plot.jl")

#Package version
using Pkg
__VERSION__ = VersionNumber(Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"])

end # module HighOrderMRI
