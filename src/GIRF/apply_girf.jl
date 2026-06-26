"""
    apply_girf(G_DCS_nom::AbstractArray{T, N}, Hw::AbstractArray; dim_spatial::Int=1, dim_time::Int=2, rbw::Float64=1.0) where {T, N}

Applies the Gradient Impulse Response Function (GIRF) model in the frequency domain 
to predict actual waveforms. 

# Arguments
- `G_DCS_nom`: N-dimensional nominal physical gradient array.
- `Hw`: Frequency-domain GIRF model array (e.g., [nFreq, 3, nOut]). Assumes DC component is centered (fftshift-ed).
- `dim_spatial`: Dimension corresponding to the spatial gradient axes (X, Y, Z). Default is 1.
- `dim_time`: Dimension corresponding to the time points (nPoint). Default is 2.
- `rbw`: Relative bandwidth (0 to 1.0). Nulls high-frequency components outside this ratio to prevent noise amplification (Default: 1.0, no filtering).

# Returns
- `G_act_all`: Predicted array replacing the size of `dim_spatial` with `nOut` (e.g., 16 SH channels).
"""
function apply_girf(
    G_DCS_nom   :: AbstractArray{T, N}, 
    Hw          :: AbstractArray      ; 
    dim_spatial :: Int     = 1        , 
    dim_time    :: Int     = 2        , 
    rbw         :: Float64 = 1.0
) where {T, N}
    size(G_DCS_nom, dim_spatial) == 3 || throw(ArgumentError("Input gradient must have 3 channels along dim_spatial."))
    
    # Modify parsing order to adapt Hw dimensions to [nFreq, 3, nOut]
    nFreq_orig, nIn_Hw, nOut = size(Hw)
    @assert nIn_Hw == 3 "GIRF model must have 3 input channels (X, Y, Z)"

    nPoint = size(G_DCS_nom, dim_time)
    
    # 1. DSP Optimization: Calculate optimal FFT length (avoid wrap-around & speed up)
    N_fft = nextprod([2, 3, 5], max(nPoint, nFreq_orig))
    
    # 2. Hw Resolution Expansion (Time-domain padding)
    if N_fft > nFreq_orig
        @info "GIRF Resolution Expansion: Padding Hw from $nFreq_orig to $N_fft points to safely cover gradient length."
        
        # ifftshift must be applied before converting back to time domain
        hw_t = ifft(ifftshift(Hw, 1), 1)
        # Modify initialization dimension order of padded array to [N_fft, 3, nOut]
        hw_t_padded = zeros(eltype(Hw), N_fft, nIn_Hw, nOut)
        
        # Split and retain causal part (first half) and non-causal part (second half), zero-padding in the middle
        half_N = nFreq_orig ÷ 2
        hw_t_padded[1:half_N, :, :] .= hw_t[1:half_N, :, :]
        hw_t_padded[end - (nFreq_orig - half_N) + 2 : end, :, :] .= hw_t[half_N + 2 : end, :, :]
        
        # Convert back to frequency domain and re-apply fftshift to bring DC to the center
        Hw_work = fftshift(fft(hw_t_padded, 1), 1)
    else
        Hw_work = Hw
    end

    # 3. Align & Flatten Dimensions
    other_dims = Tuple(setdiff(1:N, (dim_spatial, dim_time)))
    perm = (dim_spatial, dim_time, other_dims...)
    G_perm = permutedims(G_DCS_nom, perm)
    
    N_other = N > 2 ? prod(size(G_perm)[3:end]) : 1
    G_3d = reshape(G_perm, 3, nPoint, N_other)
    
    # 4. Multi-threading Core Prediction
    out_3d = zeros(eltype(G_3d), nOut, nPoint, N_other)
    
    Threads.@threads for i in 1:N_other
        out_3d[:, :, i] = _apply_girf_core(view(G_3d, :, :, i), Hw_work, nPoint, N_fft, nOut, rbw)
    end
    
    # 5. Restore Original Dimensions
    out_sz_perm = (nOut, nPoint, (size(G_DCS_nom, d) for d in other_dims)...)
    out_perm = reshape(out_3d, out_sz_perm)
    
    inv_perm_array = zeros(Int, N)
    inv_perm_array[dim_spatial] = 1
    inv_perm_array[dim_time] = 2
    for (i, d) in enumerate(other_dims)
        inv_perm_array[d] = i + 2
    end
    
    return permutedims(out_perm, Tuple(inv_perm_array))
end

"""
    _apply_girf_core(...)

(Internal) Core 2D thread-safe kernel for applying GIRF computation over frequency.
"""
function _apply_girf_core(G_2d::AbstractMatrix, Hw_work::AbstractArray, nPoint::Int, N_fft::Int, nOut::Int, rbw::Float64)
    # Safely pad gradient to N_fft 
    grad_padded = zeros(eltype(G_2d), 3, N_fft)
    grad_padded[:, 1:nPoint] .= G_2d

    # Transform to frequency domain and SHIFT DC to center
    Grad_freq = fftshift(fft(grad_padded, 2), 2)
    
    # Apply Relative Bandwidth (rbw) Low-Pass Filter
    if rbw < 1.0
        nzpts = round(Int, N_fft * (1.0 - rbw) / 2.0)
        if nzpts > 0
            Grad_freq[:, 1:nzpts] .= 0
            Grad_freq[:, (N_fft - nzpts + 1):N_fft] .= 0
        end
    end

    Out_freq = zeros(Complex{eltype(G_2d)}, nOut, N_fft)

    # SIMD optimized frequency-domain multiplication
    for i in 1:3
        for o in 1:nOut
            @inbounds @simd for f in 1:N_fft
                # Adjust index to [f, i, o] to match [nFreq, 3, nOut]
                Out_freq[o, f] += Hw_work[f, i, o] * Grad_freq[i, f]
            end
        end
    end

    # Transform back: Shift DC back to index 1, then IFFT
    out_time_all = real.(ifft(ifftshift(Out_freq, 2), 2))
    
    # Return valid sequence length
    return out_time_all[:, 1:nPoint]
end

"""
    unpack_girf_channels(G_act_all::AbstractArray{T, N}; dim_spatial::Int=1) where {T, N}

Separates the 0th order (B0), 1st order (X, Y, Z), and higher-order Spherical Harmonic (SH) components 
from the multi-dimensional output of the GIRF model.
"""
function unpack_girf_channels(G_act_all::AbstractArray{T, N}; dim_spatial::Int=1) where {T, N}
    nOut = size(G_act_all, dim_spatial)
    
    if nOut >= 4
        # Channel 1: k0 (B0 field / Global phase)
        k0_act    = copy(selectdim(G_act_all, dim_spatial, 1:1))   
        # Channels 2-4: Linear gradients (Gx, Gy, Gz)
        G_DCS_act = copy(selectdim(G_act_all, dim_spatial, 2:4))   
        # Channels 5-end: Higher order Spherical Harmonics
        G_HO_act  = nOut > 4 ? copy(selectdim(G_act_all, dim_spatial, 5:nOut)) : nothing 
        return G_DCS_act, k0_act, G_HO_act
    else
        @warn "GIRF output has less than 4 channels. Treating all channels as 1st-order gradients."
        shape_k0 = ntuple(d -> d == dim_spatial ? 1 : size(G_act_all, d), N)
        k0_act = zeros(eltype(G_act_all), shape_k0)
        return G_act_all, k0_act, nothing
    end
end