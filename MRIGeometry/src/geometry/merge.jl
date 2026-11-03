import Base: merge

export merge
"""
    Base.merge(geos::AbstractVector{<:Geometry{T, D}}) where {T <: AbstractFloat, D <: Integer}

Construct one nominal 3D Geometry from a nonempty vector of slice geometries.
It sorts centres along the first geometry's slice normal, uses their midpoint
as the new centre, and sets the third count to the number of entries. This
method returns only geometry; it does not reorder an image array for you.

Current limitation: the third FOV is centre-span plus the first slice's FOV.
For gapped slices this changes the implied centre spacing and does not preserve
the original centres. Restrict use to mutually aligned, uniformly spaced,
single-plane geometries whose slice thickness equals centre spacing. A single
entry is promoted with a third count of one, so this is not a general slab
merge. Heterogeneous orientation, spacing and thickness are not validated.
Use explicit physical geometry rather than this shortcut for gapped or
nonuniform slice stacks.
"""
function Base.merge(geos::AbstractVector{<:Geometry{T, D}}) where {T <: AbstractFloat, D <: Integer}
    n_slices = length(geos)
    @assert n_slices > 0 "Cannot merge an empty array of geometries."
    
    # Base case: If there is only one slice, elevate its dimension to 3D and return
    if n_slices == 1
        base_geo = geos[1]
        return Geometry(
            String(base_geo.SystemVendor),
            D(3), # Promote dimension to 3D
            T.(base_geo.FOV),
            D[base_geo.MatrixSize[1], base_geo.MatrixSize[2], 1],
            String(base_geo.PatientPosition),
            T.(base_geo.T_PCS),
            T.(base_geo.R_RPS_PCS),
            D(1) # Reset global index
        )
    end

    # 1. Extract base parameters and the slice normal vector
    # The 3rd column of the rotation matrix R_RPS_PCS represents the slice normal in PCS
    base_geo = geos[1]
    slice_normal = base_geo.R_RPS_PCS[:, 3]

    # 2. Sort geometries spatially (Crucial for interleaved acquisitions)
    # Calculate the projection of each slice's physical center onto the slice normal
    slab_centers = [dot(geo.T_PCS, slice_normal) for geo in geos]
    sort_idx = sortperm(slab_centers)
    
    sorted_geos = geos[sort_idx]
    sorted_centers = slab_centers[sort_idx]

    # 3. Calculate the new global FOV and MatrixSize
    slice_thickness = base_geo.FOV[3]
    
    # Total FOV along the slice dimension = distance between outermost slice centers + one slice thickness
    # This mathematical formulation naturally accommodates slice gaps
    total_slice_fov = (sorted_centers[end] - sorted_centers[1]) + slice_thickness

    # Construct strictly typed arrays to prevent implicit conversions
    new_fov = T[base_geo.FOV[1], base_geo.FOV[2], total_slice_fov]
    new_matrix_size = D[base_geo.MatrixSize[1], base_geo.MatrixSize[2], n_slices]

    # 4. Calculate the new physical center (T_PCS)
    # Compute the geometric midpoint between the first and last sorted slices.
    # Note: Explicitly cast 2 to type T to ensure type stability (avoiding Float64 contamination)
    new_T_PCS = (sorted_geos[1].T_PCS .+ sorted_geos[end].T_PCS) ./ T(2)

    # 5. Construct and return the new 3D Geometry
    return Geometry(
        String(base_geo.SystemVendor),
        D(3),                           # Geometrically promoted to a 3D volume
        new_fov,
        new_matrix_size,
        String(base_geo.PatientPosition),
        Vector{T}(new_T_PCS),           # Ensure closed Array type
        Matrix{T}(base_geo.R_RPS_PCS),
        D(1)                            # Reset global index to 1
    )
end