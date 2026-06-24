"""
    apply_girf(G_DCS_nom::AbstractArray{T, N}, Hw::AbstractArray; xyz_dim::Int=1, point_dim::Int=2) where {T, N}

Applies the Gradient Impulse Response Function (GIRF) model in the frequency domain 
to predict actual waveforms. Supports arbitrary multi-dimensional arrays (e.g., handling 
multiple echoes, dynamic scans, or slices automatically via multi-threading acceleration).

# Arguments
- `G_DCS_nom`: N-dimensional nominal physical gradient array.
- `Hw`: Frequency-domain GIRF model array (e.g., [Nfreq, Nout, 3]).
- `xyz_dim`: The dimension corresponding to the spatial gradient axes (X, Y, Z). Default is 1.
- `point_dim`: The dimension corresponding to the time points (nPoint). Default is 2.

# Returns
- `G_act_all`: Predicted array with the same number of dimensions, replacing the size of 
               `xyz_dim` with `Nout` (typically 16 for 0th-3rd order spherical harmonics or 3).
"""
function apply_girf(G_DCS_nom::AbstractArray{T, N}, Hw::AbstractArray; xyz_dim::Int=1, point_dim::Int=2) where {T, N}
    size(G_DCS_nom, xyz_dim) == 3 || throw(ArgumentError("Input gradient must have 3 channels along xyz_dim."))
    
    Nfreq, Nout, Nin_Hw = size(Hw)
    @assert Nin_Hw == 3 "GIRF model must have 3 input channels (X, Y, Z)"

    nPoint = size(G_DCS_nom, point_dim)
    if nPoint > Nfreq
        @warn "The number of points ($nPoint) exceeds the frequency points in the GIRF model ($Nfreq). Truncation will occur!"
    end
    
    # 1. Permute dims so that xyz_dim is 1st and point_dim is 2nd
    other_dims = Tuple(setdiff(1:N, (xyz_dim, point_dim)))
    perm = (xyz_dim, point_dim, other_dims...)
    G_perm = permutedims(G_DCS_nom, perm)
    
    # 2. Reshape to 3D: (3, nPoint, N_other) to generalize loop iterations
    N_other = N > 2 ? prod(size(G_perm)[3:end]) : 1
    G_3d = reshape(G_perm, 3, nPoint, N_other)
    
    # 3. Apply GIRF using multi-threading
    out_3d = zeros(eltype(G_3d), Nout, nPoint, N_other)
    
    Threads.@threads for i in 1:N_other
        out_3d[:, :, i] = _apply_girf_core(view(G_3d, :, :, i), Hw, nPoint, Nfreq, Nout)
    end
    
    # 4. Reshape and inverse permute
    # Target shape before inv_perm: (Nout, nPoint, size(G_DCS_nom)[other_dims]...)
    out_sz_perm = (Nout, nPoint, (size(G_DCS_nom, d) for d in other_dims)...)
    out_perm = reshape(out_3d, out_sz_perm)
    
    # Construct inverse permutation array
    inv_perm_array = zeros(Int, N)
    inv_perm_array[xyz_dim] = 1
    inv_perm_array[point_dim] = 2
    for (i, d) in enumerate(other_dims)
        inv_perm_array[d] = i + 2
    end
    
    return permutedims(out_perm, Tuple(inv_perm_array))
end

"""
    _apply_girf_core(G_2d::AbstractMatrix, Hw::AbstractArray, nPoint::Int, Nfreq::Int, Nout::Int)

(Internal) Core 2D thread-safe kernel for applying GIRF computation over time/frequency.
"""
function _apply_girf_core(G_2d::AbstractMatrix, Hw::AbstractArray, nPoint::Int, Nfreq::Int, Nout::Int)
    if nPoint > Nfreq
        grad_padded = G_2d[:, 1:Nfreq]
    else
        grad_padded = zeros(eltype(G_2d), 3, Nfreq)
        grad_padded[:, 1:nPoint] .= G_2d
    end

    Grad_freq = fft(grad_padded, 2)
    Out_freq = zeros(Complex{eltype(G_2d)}, Nout, Nfreq)

    # Performance optimization: Fast contiguous memory access for inner SIMD loop
    for i in 1:3
        for o in 1:Nout
            @inbounds @simd for f in 1:Nfreq
                Out_freq[o, f] += Hw[f, o, i] * Grad_freq[i, f]
            end
        end
    end

    out_time_all = real.(ifft(Out_freq, 2))
    return out_time_all[:, 1:nPoint]
end

"""
    unpack_girf_channels(G_act_all::AbstractArray{T, N}; xyz_dim::Int=1) where {T, N}

(Helper function) Separates the 0th order, 1st order, and higher-order components 
from the multi-dimensional output of the GIRF model.

# Arguments
- `G_act_all`: Multi-dimensional output array from `apply_girf`.
- `xyz_dim`: The dimension corresponding to the GIRF output channels. Default is 1.
"""
function unpack_girf_channels(G_act_all::AbstractArray{T, N}; xyz_dim::Int=1) where {T, N}
    Nout = size(G_act_all, xyz_dim)
    
    if Nout >= 4
        # Using selectdim() dynamically supports arrays of any shape/dimensions
        k0_act    = copy(selectdim(G_act_all, xyz_dim, 1:1))   
        G_DCS_act = copy(selectdim(G_act_all, xyz_dim, 2:4))   
        G_HO_act  = Nout > 4 ? copy(selectdim(G_act_all, xyz_dim, 5:Nout)) : nothing 
        return G_DCS_act, k0_act, G_HO_act
    else
        # Default to all being 1st-order gradients.
        shape_k0 = ntuple(d -> d == xyz_dim ? 1 : size(G_act_all, d), N)
        k0_act = zeros(eltype(G_act_all), shape_k0)
        return G_act_all, k0_act, nothing
    end
end