"""
    resample(src_data::AbstractArray{T, 3}, geo_src::Geometry, geo_tgt::Geometry; oversample=[1, 1, 1], space=:PCS) where T

Accurately resamples 3D source data to match the target sequence geometry using Sub-voxel Volumetric Integration.

# Arguments
- `oversample::AbstractVector{<:Integer}`: Sub-voxel integration points along `[Readout, Phase, Slice]`. Default `[1, 1, 1]`.
- `space::Symbol`: Target registration space. `:PCS` (Patient Coordinate System, default) or `:DCS` (Device Coordinate System).
"""
function resample(
    src_data   :: AbstractArray{T, 3}, 
    geo_src    :: Geometry, 
    geo_tgt    :: Geometry;
    oversample :: AbstractVector{<:Integer} = [1, 1, 1],
    space      :: Symbol = :PCS
) where T
    length(oversample) == 3 || error("oversample must be a 3-element vector [R, P, S]")

    # 1. Build continuous interpolator (extrapolate out-of-bounds with 0)
    itp = extrapolate(interpolate(src_data, BSpline(Linear())), zero(T))

    # 2. Generate target grid & Transform to source RPS space based on chosen physical space
    tgt_rps = gen_RPS_grid(geo_tgt)
    
    if space == :PCS
        tgt_phys = RPS2PCS(geo_tgt, tgt_rps)
        src_rps  = PCS2RPS(geo_src, tgt_phys)
        R_T2S    = transpose(geo_src.R_RPS_PCS) * geo_tgt.R_RPS_PCS
    elseif space == :DCS
        tgt_phys = RPS2DCS(geo_tgt, tgt_rps)
        src_rps  = DCS2RPS(geo_src, tgt_phys)
        R_T2S    = transpose(geo_src.R_RPS_DCS) * geo_tgt.R_RPS_DCS
    else
        error("Unsupported space: $space. Choose :PCS or :DCS.")
    end

    K = prod(geo_tgt.MatrixSize)
    pts_src = reshape(src_rps, K, 3)

    dx_src     = geo_src.FOV ./ geo_src.MatrixSize
    dx_tgt     = geo_tgt.FOV ./ geo_tgt.MatrixSize
    Nx, Ny, Nz = geo_src.MatrixSize

    # 3. Calculate sub-voxel relative offsets
    get_offsets(res, os) = os <= 1 ? [0.0] : range(-res/2, res/2, length=os+2)[2:end-1]
    
    r_off  = get_offsets(dx_tgt[1], oversample[1])
    p_off  = get_offsets(dx_tgt[2], oversample[2])
    s_off  = get_offsets(dx_tgt[3], oversample[3])
    n_smps = prod(oversample)

    # 4. Precompute absolute sub-voxel shift vectors in source space
    v_r, v_p, v_s = R_T2S[:, 1], R_T2S[:, 2], R_T2S[:, 3]
    shifts = Vector{Vector{Float64}}(undef, n_smps)
    
    idx = 1
    for ds in s_off, dp in p_off, dr in r_off
        shifts[idx] = dr .* v_r .+ dp .* v_p .+ ds .* v_s
        idx += 1
    end

    out_flat = zeros(T, K)

    # 5. High-speed evaluation with volumetric integration
    for i in 1:K
        r_base, p_base, s_base = pts_src[i, 1], pts_src[i, 2], pts_src[i, 3]
        v_sum = zero(T)
        
        for shift in shifts
            r_m = r_base + shift[1]
            p_m = p_base + shift[2]
            s_m = s_base + shift[3]

            # Float indexing logic: idx = x / dx + (N + 1) / 2
            idx_r = r_m / dx_src[1] + (Nx + 1) / 2
            idx_p = p_m / dx_src[2] + (Ny + 1) / 2
            idx_s = Nz > 1 ? (s_m / dx_src[3] + (Nz + 1) / 2) : 1.0

            v_sum += itp(idx_r, idx_p, idx_s)
        end
        
        out_flat[i] = v_sum / n_smps
    end

    return reshape(out_flat, geo_tgt.MatrixSize...)
end

"""
    resample(src_data::AbstractArray{T, 4}, geo_src::Geometry, geo_tgt::Geometry; kwargs...) where T

Wrapper for resampling multi-channel (4D) data (e.g., Coil-Sensitivity Maps).
"""
function resample(
    src_data   :: AbstractArray{T, 4}, 
    geo_src    :: Geometry, 
    geo_tgt    :: Geometry;
    oversample :: AbstractVector{<:Integer} = [1, 1, 1],
    space      :: Symbol = :PCS
) where T
    tgt_sz = geo_tgt.MatrixSize
    n_cha  = size(src_data, 4)
    out    = zeros(T, tgt_sz[1], tgt_sz[2], tgt_sz[3], n_cha)
    
    for c in 1:n_cha
        out[:, :, :, c] = resample(
            @views(src_data[:, :, :, c]), geo_src, geo_tgt; 
            oversample=oversample, space=space
        )
    end
    
    return out
end