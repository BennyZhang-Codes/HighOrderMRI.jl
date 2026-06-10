"""
    get_geometry(raw::RawAcquisitionData; T=Float64, D=Int64) -> Vector{Geometry{T}}

A universal function to extract an array of `Geometry` objects from MRI raw data.
Automatically adapts to 2D single-slice, 2D multi-slice, 3D single-slab, and 3D multi-slab 
by reading ISMRMRD reconSpace parameters directly.

# Returns
- A `Vector{Geometry{T}}` where the length corresponds to the number of unique slices or slabs.
"""
function get_geometry(
    raw::RawAcquisitionData; 
    T::Type{<:AbstractFloat}=Float64,
    D::Type{<:Integer}=Int64)

    # 1. Extract the set of unique geometries (contains 'geometry' and 'idx_slice' fields)
    unique_geo_bundles = extract_unique_geometries(raw; decimals=4)
    n_unique = length(unique_geo_bundles)

    # 2. Extract global parameters 
    # (ISMRMRD's reconFOV and reconSize are natively scaled per slice/slab)
    SystemVendor    = String(raw.params["systemVendor"])
    FOV             = collect(raw.params["reconFOV"]) .* 1e-3 # Convert to [m]
    MatrixSize      = collect(raw.params["reconSize"])        # [Nx, Ny, Nz_local]
    PatientPosition = String(raw.params["patientPosition"])

    # 3. Minimalist dimension inference: if Z-dimension size is 1, it's 2D; otherwise, 3D
    Dimension = MatrixSize[3] == 1 ? 2 : 3

    # 4. Pre-allocate the Geometry array with strict typing 
    # Ensures maximum performance (zero allocations) and type stability in the loop
    geometries = Vector{Geometry{T}}(undef, n_unique)

    # 5. Loop through and instantiate the Geometry for each unique slice/slab
    for (i, bundle) in enumerate(unique_geo_bundles)
        
        # Extract the actual geometric data from the NamedTuple
        geo_info = bundle.geometry
        
        # Retrieve and convert the absolute physical center and rotation vectors
        T_PCS     = collect(geo_info.position) .* 1e-3  # Convert to [m]
        R_RPS_PCS = hcat(collect(geo_info.read_dir), collect(geo_info.phase_dir), collect(geo_info.slice_dir))

        # Instantiate and store in the pre-allocated array
        geometries[i] = Geometry(
            SystemVendor,
            D(Dimension), 
            T.(FOV), 
            D.(MatrixSize), 
            PatientPosition,
            T.(T_PCS),
            T.(R_RPS_PCS)
        )
    end

    return geometries
end