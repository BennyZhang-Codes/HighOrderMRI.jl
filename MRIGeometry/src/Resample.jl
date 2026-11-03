"""
    resample(src_data::AbstractArray{T, 3}, geo_src::Geometry, geo_tgt::Geometry; oversample::AbstractVector{<:Integer} = [1, 1, 1], space::Symbol = :PCS, is_mask::Bool = false) where T

Sample a 3D source array on a target Geometry in patient (:PCS) or device
(:DCS) space. Source shape must match geo_src.MatrixSize. Use positive
three-element oversample counts. Continuous signals use linear interpolation
with zero extrapolation; oversample averages equally weighted interior
sub-voxel sample points. The default [1,1,1] evaluates voxel centres only.
This is numerical point averaging, not exact volume-overlap integration.

With is_mask=true, use nearest interpolation, force [1,1,1] sampling and
return integer labels. Otherwise the result has the target spatial shape and
the source type (integer sources are temporarily converted to Float32 and
rounded back). Geometric transforms assume orthogonal rotations and consistent
metre units. Nonpositive oversample values and inconsistent shapes are not
fully guarded and can fail inside the computation.

Current limitation: a source with a singleton spatial axis can fail while
building the interpolation object. The later Nz==1 coordinate shortcut does
not fix that failure or define a finite slice-thickness support. Do not treat
this implementation as validated single-slice-to-volume resampling.
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
    resample(src_data::AbstractArray{T, 4}, geo_src::Geometry, geo_tgt::Geometry; normalize_csm::Bool = false, kwargs...) where T

Apply the 3D resampler independently to each channel on axis four. Keyword
arguments are forwarded to it, including its singleton-axis limitation.
Return `(target_nX,target_nY,target_nZ,nCha)`. `normalize_csm=true` divides
each channel vector by its RSS plus 1e-8 and replaces NaNs with zero; use it
only for floating-point coil maps, not categorical labels or integer data.
This sets a local sensitivity convention and does not establish physical
energy conservation or automatically transform acquisition data.
"""
function resample(
    src_data :: AbstractArray{T, 4}, 
    geo_src  :: Geometry, 
    geo_tgt  :: Geometry;
    normalize_csm :: Bool = false,
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

    # 2. CSM RSS Normalization & NaN safety check
    if normalize_csm
        if is_mask
            @warn "normalize_csm is enabled, but is_mask=true. Normalization is intended for complex coil maps."
        end
        # Calculate L2 norm across coils (dimension 4)
        norm_map = sqrt.(sum(abs.(out).^2, dims=4))
        
        # Apply normalization using broadcast fusion (@.)
        @. out = out / (norm_map + 1e-8)
        
        # Safe, non-allocating NaN replacement
        for i in eachindex(out)
            if isnan(out[i])
                out[i] = zero(OutType)
            end
        end
    end
    return out
end

"""
    resample(src_data::AbstractArray{T, 3}, geo_src::Geometry, geo_tgts::Vector{<:Geometry}; kwargs...) where T

Return a vector of 3D arrays by resampling the same source independently to
each target Geometry, using threaded work. Target ordering is preserved.
All restrictions of the single-target 3D method apply.
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
    resample(src_data::AbstractArray{T, 4}, geo_src::Geometry, geo_tgts::Vector{<:Geometry}; normalize_csm::Bool = false, kwargs...) where T

Return a vector of four-dimensional channel arrays, one per target Geometry.
Forward normalize_csm and other keywords to each single-target call. The
singleton-axis, interpolation and RSS-normalization limitations are unchanged;
the wrapper does not assemble the targets into a common volume.
"""
function resample(
    src_data :: AbstractArray{T, 4}, 
    geo_src  :: Geometry, 
    geo_tgts :: Vector{<:Geometry};
    normalize_csm :: Bool = false,
    kwargs...
) where T
    is_mask = haskey(kwargs, :is_mask) ? kwargs[:is_mask] : false
    OutType = is_mask ? Int : T
    out = Vector{Array{OutType, 4}}(undef, length(geo_tgts))
    
    Threads.@threads for i in eachindex(geo_tgts)
        out[i] = resample(src_data, geo_src, geo_tgts[i]; normalize_csm=normalize_csm, kwargs...)
    end
    
    return out
end