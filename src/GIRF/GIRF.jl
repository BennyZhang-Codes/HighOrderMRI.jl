include("apply_girf.jl")
export apply_girf


"""
    GIRFModel{T <: AbstractFloat}

A dedicated struct to encapsulate the Gradient Impulse Response Function (GIRF) model.
Provides a strict type wrapper to prevent Type Piracy and enable safe multiple dispatch 
across the MRI reconstruction ecosystem.

# Fields
- `Hw::Array{Complex{T}, 3}`: The frequency-domain GIRF transfer function. 
                              Expected dimensions: `[nFreq, 3, nOut]`.
- `freqs::Vector{T}`: The frequency points vector matching the 1st dimension of `Hw`.
"""
struct GIRFModel{T <: AbstractFloat}
    Hw::Array{Complex{T}, 3}
    freqs::Vector{T}
end

# Outer constructor for convenient instantiation and automatic type conversion
GIRFModel(Hw::AbstractArray{Complex{T}, 3}, freqs::AbstractVector{T}) where {T <: AbstractFloat} = 
    GIRFModel{T}(Array(Hw), vec(freqs))



Base.show(io::IO, GIRF::GIRFModel{T}) where {T} = begin
    nFreq, nIn, nOut = size(GIRF.Hw)
    
    println(io, ">>> GIRFModel{$T} <<<")
    println(io, "Dimensions       : ", size(GIRF.Hw), " [nFreq, nIn, nOut]")
    println(io, "Input Channels   : ", nIn, " (Typically X, Y, Z)")
    println(io, "Output Channels  : ", nOut, " (Spherical Harmonics)")
    
    if !isempty(GIRF.freqs)
        f_min = minimum(GIRF.freqs) / 1e3
        f_max = maximum(GIRF.freqs) / 1e3
        df    = length(GIRF.freqs) > 1 ? GIRF.freqs[2] - GIRF.freqs[1] : 0.0
        
        println(io, "Frequency Points : ", nFreq)
        println(io, "Frequency Range  : ", round(f_min, digits=2), " to ", round(f_max, digits=2), " kHz")
        println(io, "Frequency Res.   : ", round(df, digits=4), " Hz")
    end
end

# Convenience overload: Allow apply_girf to directly accept the model
apply_girf(G_DCS_nom::AbstractArray, GIRF::GIRFModel; kwargs...) = apply_girf(G_DCS_nom, GIRF.Hw; kwargs...)


export GIRFModel