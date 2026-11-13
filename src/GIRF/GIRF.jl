include("apply_girf.jl")
export apply_girf


"""
    GIRFModel{T <: AbstractFloat}

Container for a centred frequency-domain gradient impulse response model.
`Hw::Array{Complex{T},3}` has intended shape `(nFreq,3,nOut)` and
`freqs::Vector{T}` records the corresponding frequencies in Hz. The three
input channels are device-coordinate gradient axes; the output channels and
their units must match the measured model calibration.

These fields are CPU arrays, not a promise of independent copies: compatible
input arrays may remain aliased. Shape, frequency-grid consistency and
Hermitian symmetry are not fully validated by construction. `apply_girf`
uses `Hw` and does not resample it using `freqs`. A model alone therefore
does not establish compatible waveform sampling or a finite-record boundary
condition. See [GIRF prediction](/guide/field-preprocessing#girf-prediction).
"""
struct GIRFModel{T <: AbstractFloat}
    Hw::Array{Complex{T}, 3}
    freqs::Vector{T}
end

# Outer constructor for convenient instantiation and automatic type conversion
"""
    GIRFModel(Hw::AbstractArray{Complex{T}, 3}, freqs::AbstractVector{T}) where T <: AbstractFloat

Construct the model from complex transfer samples and a frequency vector
with the same real precision T. The outer method converts Hw with `Array`
and uses `vec(freqs)`; an already compatible array can remain shared. The
frequency-vector length and monotonicity are caller responsibilities.
This constructor does not estimate a GIRF or resample its frequency grid.
"""
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
"""
    apply_girf(G_DCS_nom::AbstractArray, GIRF::GIRFModel; kwargs...)

Apply `GIRF.Hw` with the array overload of `apply_girf`, forwarding all
keywords. `GIRF.freqs` is metadata only in this call. The sampling-grid,
response-expansion and circular-convolution limitations of that overload
also apply here.
"""
apply_girf(G_DCS_nom::AbstractArray, GIRF::GIRFModel; kwargs...) = apply_girf(G_DCS_nom, GIRF.Hw; kwargs...)


export GIRFModel