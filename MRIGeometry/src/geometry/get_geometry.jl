"""
    get_geometry(raw::RawAcquisitionData; T::Type{<:AbstractFloat} = Float64, D::Type{<:Integer} = Int64)

Read geometry records from `RawAcquisitionData`. Returns
`Vector{Geometry{T,D}}`, one item per distinct rounded profile geometry.
It uses systemVendor, reconFOV, reconSize and patientPosition from raw.params,
and position/read_dir/phase_dir/slice_dir from profile headers. FOV and centre
are converted from millimetres to metres; slice labels are shifted from zero
to one based. reconSize[3]==1 selects Dimension=2, otherwise Dimension=3.

Profiles are grouped at four decimal places in the original metadata units;
the returned geometry uses the first original values in each group. Entries
follow first occurrence, not a guaranteed physical order. The function does
not return the profile-to-geometry index map, validate every ISMRMRD producer,
or verify the vendor transform tables. Required metadata must be present and
its coordinate definitions must match the conventions used here.
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
    geometries = Vector{Geometry{T, D}}(undef, n_unique)

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
            T.(R_RPS_PCS),
            D(bundle.idx_slice + 1),  # ISMRMRD's slice indices start at 0
        )
    end

    return geometries
end