"""
    export_nifti(x::AbstractArray{<:Real}, geo::Geometry, outpath::String, name::String)

Write a real 3D or 4D image as `joinpath(outpath,name*".nii.gz")` using a
single regular Geometry. The first three array dimensions must match
geo.MatrixSize; this precondition is not explicitly checked. Create outpath
before calling. Non-spatial dimensions are retained, but temporal metadata
is not provided (`time_step=0`).

The affine uses voxel-centre spacing FOV/MatrixSize, the centre T_PCS and
R_RPS_PCS; LPS coordinates are converted to NIfTI RAS and metres to millimetres.
Returns the Float32 3×4 affine passed to the writer. This does not validate
vendor mapping or modify the input image. Check the resulting affine against
known physical landmarks independently of an implementation round-trip.
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

Write magnitude and principal phase as separate real NIfTI files named
`name_mag.nii.gz` and `name_pha.nii.gz`; phase values are in radians. Return
the magnitude file's affine. The single-geometry shape, units and output-folder
requirements apply to both files. No complex-valued NIfTI or phase unwrapping
is performed.
"""
function export_nifti(x::AbstractArray{<:Complex}, geo::Geometry, outpath::String, name::String)
    mat_mag = export_nifti(abs.(x), geo, outpath, "$(name)_mag")
    mat_pha = export_nifti(angle.(x), geo, outpath, "$(name)_pha")
    return mat_mag
end

"""
    export_nifti(x::AbstractArray{<:Real}, geos::AbstractVector{<:Geometry}, outpath::String, name::String)

Write a slice stack using a vector of geometries. The implementation sorts
by Idx_Slice and uses that permutation directly on array axis three. It then
constructs an affine from the first geometry and the unsigned distance between
the first two centres. It does not validate regular spacing or orientations.

Known limitations: one geometry must represent one plane (`MatrixSize[3]==1`)
for this indexing scheme. Supplying multi-plane slabs can silently drop planes.
Descending physical centres can produce an affine pointing in the wrong
direction. Nonuniform spacing and in-plane offsets are not represented safely.
Do not use this overload for those cases. Prefer an explicitly validated
single regular Geometry and correctly ordered complete image array. A safe
slice-only case needs Idx_Slice order to match increasing positions along a
common normal, uniform spacing and consistent in-plane geometry. Output and
return conventions follow the real single-geometry overload.
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

Split a complex stack into magnitude and principal phase and delegate to the
vector-of-geometries real overload. Its plane-loss, sorting and affine
limitations apply unchanged. Returns the magnitude affine; this wrapper does
not make multi-slab export safe.
"""
function export_nifti(x::AbstractArray{<:Complex}, geos::AbstractVector{<:Geometry}, outpath::String, name::String)
    mat_mag = export_nifti(abs.(x), geos, outpath, "$(name)_mag")
    mat_pha = export_nifti(angle.(x), geos, outpath, "$(name)_pha")
    return mat_mag
end