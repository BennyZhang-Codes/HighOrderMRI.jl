"""
    extract_unique_geometries(raw; decimals=4)

Extract all unique geometric configurations (Position and Read/Phase/Slice directions) 
from MRI raw data profiles, while rigorously tracking the original ADC profile indices 
that belong to each unique geometry.

# Arguments
- `raw`: RawAcquisitionData object (from MRIReco.jl / ISMRMRD format).
- `decimals`: Integer. Precision used to round floats before determining uniqueness. 

# Returns
A Vector of NamedTuples, where each element represents a unique slab/slice:
`[(geometry = (position=..., read_dir=...), indices = [1, 2, 5, ...]), ...]`
"""
function extract_unique_geometries(raw; decimals::Int=4)
    unique_geos  = []                    # Track the unique geometric NamedTuples
    indices_map  = Vector{Vector{Int}}() # Track the ADC profile indices for each geometry
    geo_to_index = Dict{Any, Int}()      # Hash map for O(1) lookup during classification

    for (i, p) in enumerate(raw.profiles)
        # 1. Extract and round to eliminate floating-point noise
        pos   = collect(p.head.position)
        r_dir = collect(p.head.read_dir)
        p_dir = collect(p.head.phase_dir)
        s_dir = collect(p.head.slice_dir)

        pos_dec   = round.(Float64.(p.head.position),  digits=decimals)
        r_dir_dec = round.(Float64.(p.head.read_dir),  digits=decimals)
        p_dir_dec = round.(Float64.(p.head.phase_dir), digits=decimals)
        s_dir_dec = round.(Float64.(p.head.slice_dir), digits=decimals)

        geo_dec = (position=pos_dec, read_dir=r_dir_dec, phase_dir=p_dir_dec, slice_dir=s_dir_dec)

        geo_orig = (position=pos, read_dir=r_dir, phase_dir=p_dir, slice_dir=s_dir)

        # 2. Check if this geometry has been registered
        if haskey(geo_to_index, geo_dec)
            # If exists, append the current ADC index 'i' to the corresponding group
            idx = geo_to_index[geo_dec]
            push!(indices_map[idx], i)
        else
            # If new, register the geometry and initialize its index list
            push!(unique_geos, geo_orig)
            push!(indices_map, [i])
            geo_to_index[geo_dec] = length(unique_geos)
        end
    end

    n_unique = length(unique_geos)
    println("[Geometry Extraction]: Grouped $(length(raw.profiles)) profiles into $n_unique unique geometries.")

    # 3. Bundle the geometry and its associated indices into a clean NamedTuple
    result = [(
        geometry=unique_geos[k], 
        # indices=indices_map[k],
        idx_slice=Int(raw.profiles[indices_map[k][1]].head.idx.slice)) for k in 1:n_unique]
    return result
end