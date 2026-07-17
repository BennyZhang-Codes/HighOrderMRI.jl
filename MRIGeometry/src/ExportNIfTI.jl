"""
    export_nifti(x::AbstractArray{<:Real}, geo::Geometry, outpath::String, name::String)

Base method: Exports a single 3D/4D REAL array based on a single Geometry object.
Calculates the NIfTI standard (RAS) affine matrix and exports as .nii.gz.
"""
function export_nifti(x::AbstractArray{<:Real}, geo::Geometry, outpath::String, name::String)
    # 1. Extract basic geometry information and convert to millimeters (mm)
    Nx, Ny, Nz  = geo.MatrixSize
    vox_size_mm = Tuple(geo.FOV ./ geo.MatrixSize .* 1e3)

    # 2. RPS logic coordinate system -> LPS physical coordinate system transformation
    Rot            = geo.R_RPS_PCS
    affine_rot_LPS = Rot * Diagonal([vox_size_mm...])
    center_pos_LPS = geo.T_PCS .* 1e3

    # Calculate the physical origin corresponding to the array index [1, 1, 1]
    offset_vec_LPS = affine_rot_LPS * [(Nx - 1) / 2, (Ny - 1) / 2, (Nz - 1) / 2]
    origin_LPS     = center_pos_LPS - offset_vec_LPS

    # 3. LPS -> RAS (NIfTI standard coordinate system) transformation
    R_LPS2RAS = Diagonal([-1.0, -1.0, 1.0])
    affine_rot_RAS = R_LPS2RAS * affine_rot_LPS
    origin_RAS     = R_LPS2RAS * origin_LPS

    # Construct the final 3x4 affine matrix
    orientation_mat = Float32.(hcat(affine_rot_RAS, origin_RAS))

    # 4. Safely construct save paths (joinpath handles OS-specific slashes)
    filepath = joinpath(outpath, "$(name).nii.gz")

    # 5. Write image (ensure dense array using Array())
    niwrite(filepath, NIVolume(Array(x); voxel_size=vox_size_mm, orientation=orientation_mat, time_step=0f0))
    @info "Successfully exported NIfTI to: $filepath"

    # Return the affine matrix for potential subsequent use
    return orientation_mat
end

"""
    export_nifti(x::AbstractArray{<:Complex}, geo::Geometry, outpath::String, name::String)

Overload for COMPLEX arrays: Automatically splits into magnitude (_mag) and phase (_pha) 
and delegates directly to the Real array base method via multiple dispatch.
"""
function export_nifti(x::AbstractArray{<:Complex}, geo::Geometry, outpath::String, name::String)
    mat_mag = export_nifti(abs.(x), geo, outpath, "$(name)_mag")
    mat_pha = export_nifti(angle.(x), geo, outpath, "$(name)_pha")
    return mat_mag
end

"""
    export_nifti(x::AbstractArray{<:Real}, geos::AbstractVector{<:Geometry}, outpath::String, name::String)

Overload for multi-slice / multi-slab REAL data (Single Array, Vector of Geometries).
Automatically sorts the 3rd dimension (Z-axis) of the array based on the physical `Idx_Slice` 
to handle interleaved or out-of-order acquisitions natively.
"""
function export_nifti(x::AbstractArray{<:Real}, geos::AbstractVector{<:Geometry}, outpath::String, name::String)
    # 1. Absolute Physical Sorting based on Idx_Slice
    sort_p      = sortperm([g.Idx_Slice for g in geos])
    sorted_geos = geos[sort_p]
    
    # Sort the data array along the 3rd dimension (slice dimension)
    if ndims(x) == 3
        x_sorted = x[:, :, sort_p]
    elseif ndims(x) == 4
        x_sorted = x[:, :, sort_p, :]
    else
        error("Array to be sorted must be 3D or 4D. Got $(ndims(x))D array.")
    end

    # 2. Use the first physically sorted slice as the absolute reference origin
    geo_ref = sorted_geos[1]
    Nx, Ny, Nz_per_slab = geo_ref.MatrixSize
    
    # 3. Calculate actual slice spacing (distance between centers of physical slice 1 and 2)
    if length(sorted_geos) > 1
        slice_spacing = norm(sorted_geos[2].T_PCS .- sorted_geos[1].T_PCS) * 1e3
    else
        slice_spacing = (geo_ref.FOV[3] / Nz_per_slab) * 1e3
    end
    
    # 4. Override the Z-axis voxel size with the computed true physical spacing
    vox_size_mm = (geo_ref.FOV[1] / Nx * 1e3, geo_ref.FOV[2] / Ny * 1e3, slice_spacing)

    # RPS to LPS transformation
    Rot            = geo_ref.R_RPS_PCS
    affine_rot_LPS = Rot * Diagonal([vox_size_mm...])
    center_pos_LPS = geo_ref.T_PCS .* 1e3

    # The origin is anchored to the first sorted slice
    offset_vec_LPS = affine_rot_LPS * [(Nx - 1) / 2, (Ny - 1) / 2, (Nz_per_slab - 1) / 2]
    origin_LPS     = center_pos_LPS - offset_vec_LPS

    # LPS to RAS (NIfTI standard) transformation
    R_LPS2RAS = Diagonal([-1.0, -1.0, 1.0])
    affine_rot_RAS = R_LPS2RAS * affine_rot_LPS
    origin_RAS     = R_LPS2RAS * origin_LPS

    orientation_mat = Float32.(hcat(affine_rot_RAS, origin_RAS))

    filepath = joinpath(outpath, "$(name).nii.gz")

    # Write the physically sorted array!
    niwrite(filepath, NIVolume(Array(x_sorted); voxel_size=vox_size_mm, orientation=orientation_mat, time_step=0f0))
    @info "Successfully exported Unified & Sorted Multi-slice NIfTI to: $filepath"

    return orientation_mat
end

"""
    export_nifti(x::AbstractArray{<:Complex}, geos::AbstractVector{<:Geometry}, outpath::String, name::String)

Overload for multi-slice / multi-slab COMPLEX data.
Automatically splits into magnitude (_mag) and phase (_pha) and delegates to the Real multi-slice method.
"""
function export_nifti(x::AbstractArray{<:Complex}, geos::AbstractVector{<:Geometry}, outpath::String, name::String)
    mat_mag = export_nifti(abs.(x), geos, outpath, "$(name)_mag")
    mat_pha = export_nifti(angle.(x), geos, outpath, "$(name)_pha")
    return mat_mag
end