"""
    resample(src_data::AbstractArray{T, 3}, geo_src::Geometry, geo_tgt::Geometry; kwargs...) where T

Accurately resamples 3D source data to match the target sequence geometry using Sub-voxel Volumetric Integration.

# Arguments
- `oversample::AbstractVector{<:Integer}`: Sub-voxel integration points along `[Readout, Phase, Slice]`. Default `[1, 1, 1]`.
- `space::Symbol`: Target registration space. `:PCS` (default) or `:DCS`.
- `is_mask::Bool`: Set to `true` if resampling categorical labels (e.g., phantoms, segmentations). Uses Nearest Neighbor interpolation, disables oversampling, and returns an `Int` array. Default `false`.
"""
function resample(
    src_data   :: AbstractArray{T, 3}, 
    geo_src    :: Geometry, 
    geo_tgt    :: Geometry;
    oversample :: AbstractVector{<:Integer} = [1, 1, 1],
    space      :: Symbol = :PCS,
    is_mask    :: Bool = false
) where T
    length(oversample) == 3 || error("oversample must be a 3-element vector [R, P, S]")

    # Mask Protection: Disable volume integration to prevent label corruption
    if is_mask && oversample != [1, 1, 1]
        @warn "Volumetric integration (oversample) is disabled for masks to prevent averaging categorical labels."
        oversample = [1, 1, 1]
    end

    # 1. Build continuous interpolator
    # Nearest Neighbor for masks, Linear for continuous signals
    interp_type = is_mask ? BSpline(Constant()) : BSpline(Linear())
    
    # Interpolations.jl works best with floats. Auto-cast integers to Float32 safely.
    if T <: Integer
        itp = extrapolate(interpolate(Float32.(src_data), interp_type), 0.0f0)
        WorkType = Float32
    else
        itp = extrapolate(interpolate(src_data, interp_type), zero(T))
        WorkType = T
    end

    # 2. Generate target grid & Transform to source RPS space
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

    out_flat = zeros(WorkType, K)

    # 5. High-speed evaluation
    for i in 1:K
        r_base, p_base, s_base = pts_src[i, 1], pts_src[i, 2], pts_src[i, 3]
        v_sum = zero(WorkType)
        
        for shift in shifts
            r_m = r_base + shift[1]
            p_m = p_base + shift[2]
            s_m = s_base + shift[3]

            idx_r = r_m / dx_src[1] + (Nx + 1) / 2
            idx_p = p_m / dx_src[2] + (Ny + 1) / 2
            idx_s = Nz > 1 ? (s_m / dx_src[3] + (Nz + 1) / 2) : 1.0

            v_sum += itp(idx_r, idx_p, idx_s)
        end
        
        out_flat[i] = v_sum / n_smps
    end

    out_reshaped = reshape(out_flat, geo_tgt.MatrixSize...)

    # 6. Safe Output Casting
    if is_mask
        return round.(Int, real.(out_reshaped)) # Ensure purely integer outputs for masks
    else
        return T <: Integer ? round.(T, out_reshaped) : out_reshaped
    end
end

"""
    resample(src_data::AbstractArray{T, 4}, geo_src::Geometry, geo_tgt::Geometry; kwargs...) where T

Wrapper for resampling multi-channel (4D) data.
"""
function resample(
    src_data :: AbstractArray{T, 4}, 
    geo_src  :: Geometry, 
    geo_tgt  :: Geometry;
    kwargs...
) where T
    # Dynamically determine pre-allocation type based on kwargs
    is_mask = haskey(kwargs, :is_mask) ? kwargs[:is_mask] : false
    OutType = is_mask ? Int : T

    tgt_sz = geo_tgt.MatrixSize
    n_cha  = size(src_data, 4)
    out    = zeros(OutType, tgt_sz[1], tgt_sz[2], tgt_sz[3], n_cha)
    
    for c in 1:n_cha
        out[:, :, :, c] = resample(
            @views(src_data[:, :, :, c]), geo_src, geo_tgt; 
            kwargs...
        )
    end
    
    return out
end

"""
    resample(src_data::AbstractArray{T, 3}, geo_src::Geometry, geo_tgts::Vector{<:Geometry}; kwargs...) where T

Resamples source data for multiple target geometries independently (Multi-slice / Multi-slab).
"""
function resample(
    src_data :: AbstractArray{T, 3}, 
    geo_src  :: Geometry, 
    geo_tgts :: Vector{<:Geometry};
    kwargs...
) where T
    is_mask = haskey(kwargs, :is_mask) ? kwargs[:is_mask] : false
    OutType = is_mask ? Int : T
    out = Vector{Array{OutType, 3}}(undef, length(geo_tgts))
    
    Threads.@threads for i in eachindex(geo_tgts)
        out[i] = resample(src_data, geo_src, geo_tgts[i]; kwargs...)
    end
    
    return out
end

"""
    resample(src_data::AbstractArray{T, 4}, geo_src::Geometry, geo_tgts::Vector{<:Geometry}; kwargs...) where T

Multi-geometry resampling wrapper for 4D data.
"""
function resample(
    src_data :: AbstractArray{T, 4}, 
    geo_src  :: Geometry, 
    geo_tgts :: Vector{<:Geometry};
    kwargs...
) where T
    is_mask = haskey(kwargs, :is_mask) ? kwargs[:is_mask] : false
    OutType = is_mask ? Int : T
    out = Vector{Array{OutType, 4}}(undef, length(geo_tgts))
    
    Threads.@threads for i in eachindex(geo_tgts)
        out[i] = resample(src_data, geo_src, geo_tgts[i]; kwargs...)
    end
    
    return out
end