
function _compute_global_geometry(
    geos         :: AbstractVector{<:Geometry{T, D}}, 
    total_slices :: Int
    ) where {T <: AbstractFloat, D <: Integer}
    
    base_geo = geos[1]
    slice_normal = base_geo.R_RPS_PCS[:, 3]
    
    slice_thickness = base_geo.FOV[3] / base_geo.MatrixSize[3]
    new_fov = T[base_geo.FOV[1], base_geo.FOV[2], total_slices * slice_thickness]
    new_matrix_size = D[base_geo.MatrixSize[1], base_geo.MatrixSize[2], total_slices]
    
    first_slab_start = base_geo.T_PCS .- slice_normal .* (base_geo.FOV[3] / T(2))
    last_slab_end    = geos[end].T_PCS .+ slice_normal .* (geos[end].FOV[3] / T(2))
    new_T_PCS        = (first_slab_start .+ last_slab_end) ./ T(2)

    global_geo = Geometry(
        String(base_geo.SystemVendor),
        D(base_geo.Dimension),
        new_fov,
        new_matrix_size,
        String(base_geo.PatientPosition),
        Vector{T}(new_T_PCS),
        Matrix{T}(base_geo.R_RPS_PCS),
        D(1)
    )
    return global_geo
end

"""
    merge_multislabs(slabs_data, geos, overlap_slices; dim_z=3)

Generic multi-slab merger. Drops the overlapping edge slices and concatenates.
"""
function merge_multislabs(
    slabs_data     :: Vector{<:AbstractArray{Complex{T}}}, 
    geos           :: AbstractVector{<:Geometry{TReal, D}}, 
    overlap_slices :: Int;
    dim_z          :: Int = 3
    ) where {T <: AbstractFloat, TReal <: AbstractFloat, D <: Integer}

    @assert iseven(overlap_slices) "overlap_slices must be an even number to ensure symmetric cropping."
    n_slabs = length(slabs_data)

    slice_normal = geos[1].R_RPS_PCS[:, 3]
    slab_centers = [dot(geo.T_PCS, slice_normal) for geo in geos]
    sort_idx = sortperm(slab_centers)
    slabs_data = slabs_data[sort_idx]
    geos = geos[sort_idx]

    drop_half = overlap_slices ÷ 2
    cropped = Vector{AbstractArray{Complex{T}}}(undef, n_slabs)
    
    for i in 1:n_slabs
        nS = size(slabs_data[i], dim_z)
        start_s = (i == 1) ? 1 : (1 + drop_half)
        end_s   = (i == n_slabs) ? nS : (nS - drop_half)
        cropped[i] = selectdim(slabs_data[i], dim_z, start_s:end_s)
    end

    merged_data = cat(cropped..., dims=dim_z)
    global_geo = _compute_global_geometry(geos, size(merged_data, dim_z))

    return merged_data, global_geo
end


"""
    merge_motsa(slabs_data, geometries, overlap_slices; dim_z=3, method=:mip)

Merge overlapping 3D slabs for TOF-MRA (MOTSA).

# Supported Methods for Overlap Region:
- `:mip`     : Maximum Intensity Projection (Best for TOF vessels).
- `:blend`   : Linear cross-fade blending (Smooth background transitions).
- `:discard` : Hard crop of the overlap halves.
"""
function merge_motsa(
    slabs_data     :: Vector{<:AbstractArray{T}}, 
    geos           :: AbstractVector{<:Geometry{TReal, D}}, 
    overlap_slices :: Int;
    dim_z          :: Int = 3,
    method         :: Symbol = :mip
    ) where {T <: Number, TReal <: AbstractFloat, D <: Integer}

    @assert iseven(overlap_slices) "overlap_slices must be an even number to ensure symmetric cropping."
    
    n_slabs = length(slabs_data)
    nS_per_slab = size(slabs_data[1], dim_z)
    
    slice_normal = geos[1].R_RPS_PCS[:, 3]
    slab_centers = [dot(geo.T_PCS, slice_normal) for geo in geos]
    sort_idx = sortperm(slab_centers)
    slabs_data = slabs_data[sort_idx]
    geos = geos[sort_idx]

    final_slices = n_slabs * nS_per_slab - (n_slabs - 1) * overlap_slices
    final_size = collect(size(slabs_data[1]))
    final_size[dim_z] = final_slices
    
    merged_data = zeros(T, final_size...)

    if method == :discard
        drop_half = overlap_slices ÷ 2
        cropped = [selectdim(slabs_data[i], dim_z, 
            (i == 1 ? 1 : 1 + drop_half) : (i == n_slabs ? nS_per_slab : nS_per_slab - drop_half)) 
            for i in 1:n_slabs]
        merged_data = cat(cropped..., dims=dim_z)
        
    else
        current_s = 1
        
        for i in 1:n_slabs
            idx = current_s : (current_s + nS_per_slab - 1)
            view_merged = selectdim(merged_data, dim_z, idx)
            view_slab   = slabs_data[i]
            
            if i == 1
                view_merged .= view_slab
            else
                overlap_idx_merged = 1 : overlap_slices
                overlap_idx_slab   = 1 : overlap_slices
                
                view_overlap_merged = selectdim(view_merged, dim_z, overlap_idx_merged)
                view_overlap_slab   = selectdim(view_slab, dim_z, overlap_idx_slab)
                
                if method == :mip
                    view_overlap_merged .= ifelse.(abs.(view_overlap_slab) .> abs.(view_overlap_merged), 
                                                   view_overlap_slab, view_overlap_merged)
                elseif method == :blend
                    weight_shape = ones(Int, ndims(slabs_data[1]))
                    weight_shape[dim_z] = overlap_slices
                    
                    w_up   = reshape(range(TReal(0), TReal(1), length=overlap_slices), weight_shape...)
                    w_down = reshape(range(TReal(1), TReal(0), length=overlap_slices), weight_shape...)
                    
                    view_overlap_merged .= view_overlap_merged .* w_down .+ view_overlap_slab .* w_up
                end
                
                non_overlap_idx_merged = (overlap_slices + 1) : nS_per_slab
                non_overlap_idx_slab   = (overlap_slices + 1) : nS_per_slab
                
                selectdim(view_merged, dim_z, non_overlap_idx_merged) .= 
                    selectdim(view_slab, dim_z, non_overlap_idx_slab)
            end
            
            current_s += nS_per_slab - overlap_slices
        end
    end

    global_geo = _compute_global_geometry(geos, size(merged_data, dim_z))
    return merged_data, global_geo
end