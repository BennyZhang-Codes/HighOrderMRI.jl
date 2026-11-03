"""
    apply_girf(G_DCS_nom::AbstractArray{T, N}, Hw::AbstractArray; dim_spatial::Int = 1, dim_time::Int = 2, rbw::Float64 = 1.0) where {T, N}

Apply a frequency-domain linear, time-invariant gradient response model.
Use real floating-point nominal gradients with three physical DCS channels
along `dim_spatial` and waveform samples along the distinct `dim_time` axis.
`Hw` has shape `(nFreq,3,nOut)` with DC centred (`fftshift` ordering).
Other input dimensions are processed independently. The output preserves
their order and replaces the three-channel dimension by nOut. Its element
type follows the nominal waveform; the inverse transform's real part is used.

The frequency response and input waveform must use matching sample intervals
and compatible physical units. `rbw` is intended in `(0,1]`; below one, the
outer frequency bins of the waveform spectrum are set to zero. This keyword
is not a calibrated anti-aliasing or noise model, and its range is not checked.
The function does not insert gamma, integrate fields or convert to cycles.

# Current FFT boundary and response-expansion limits

The FFT length is `nextprod([2,3,5],max(nPoint,nFreq))`. The waveform is padded
with trailing zeros to that length, multiplied by the transfer matrix, and
the first nPoint real output samples are returned. This is circular FFT
convolution; this padding alone does not guarantee finite-record linear
convolution or prevent a response from wrapping into the beginning.

When the FFT length exceeds nFreq, the implementation expands the response
in the time domain. Its current split omits index `fld(nFreq,2)+1`, which can
discard a valid impulse-response coefficient even for odd nFreq. This is a
known numerical limitation, not safe general-purpose impulse padding. Prefer
an already matched response length and assess the input/output boundary
support explicitly; no padding fix is performed by this API.

`apply_girf(nominal, model::GIRFModel; ...)` delegates here and does not use
the model's stored frequency vector to resample the response.
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
    _apply_girf_core(G_2d::AbstractMatrix, Hw_work::AbstractArray, nPoint::Int, N_fft::Int, nOut::Int, rbw::Float64)

Internal per-waveform FFT evaluation for a three-channel matrix. The caller
provides the chosen FFT length and response array. Returns the first nPoint
samples of the real inverse transform; the finite FFT implies periodic
convolution. This helper does not add a separate linear-convolution guard.
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
    unpack_girf_channels(G_act_all::AbstractArray{T, N}; dim_spatial::Int = 1) where {T, N}

Internal output-layout helper. For at least four output channels, return
`(linear, zeroth, higher)` by selecting channels 2:4, 1:1 and 5:end along
`dim_spatial`; higher is `nothing` when exactly four channels are present.
With fewer than four channels, return all channels as linear, a same-shaped
single-channel zero zeroth term, and `nothing`, with a warning. This convention
does not verify the physical order or units of a supplied model.
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