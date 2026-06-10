"""
    get_fov_faces(grid::Array{<:Real,4}; spacing=nothing, R=nothing)

Extract the exact 3D physical bounding box faces and RPS spanning vectors from a voxel grid.
Perfectly handles half-voxel shifts for true outer boundaries.

# Returns
A NamedTuple containing:
- `faces`: Vector of 6 faces (each face is a Vector of 4 3D points) for Poly3DCollection.
- `points`: 3x8 Matrix of the 8 extreme corner points (useful for global bounding box).
- `vectors`: NamedTuple `(origin, r, p, s)` containing the RPS corner origin and 3 span vectors.
"""
function get_fov_faces(grid::Array{<:Real,4}; spacing=nothing, R=nothing)
    # 1. Extract Voxel Centers (Converted to mm)
    c000 = grid[1,   1,   1,   :] .* 1e3
    c100 = grid[end, 1,   1,   :] .* 1e3
    c010 = grid[1,   end, 1,   :] .* 1e3
    c110 = grid[end, end, 1,   :] .* 1e3
    c001 = grid[1,   1,   end, :] .* 1e3
    c101 = grid[end, 1,   end, :] .* 1e3
    c011 = grid[1,   end, end, :] .* 1e3
    c111 = grid[end, end, end, :] .* 1e3

    nx_full, ny_full, nz_full, _ = size(grid)

    # 2. Calculate Half-Voxel Extensions
    v_shift_r = [0.0, 0.0, 0.0]
    v_shift_p = [0.0, 0.0, 0.0]
    v_shift_s = [0.0, 0.0, 0.0]

    if !isnothing(spacing) && !isnothing(R)
        v_shift_r = R[:, 1] .* (spacing[1] / 2) .* 1e3
        v_shift_p = R[:, 2] .* (spacing[2] / 2) .* 1e3
        v_shift_s = R[:, 3] .* (spacing[3] / 2) .* 1e3
    else
        # Fallback inferred from grid
        if nx_full > 1; v_shift_r = (c100 .- c000) ./ (nx_full - 1) ./ 2; end
        if ny_full > 1; v_shift_p = (c010 .- c000) ./ (ny_full - 1) ./ 2; end
        if nz_full > 1; v_shift_s = (c001 .- c000) ./ (nz_full - 1) ./ 2; end
    end

    # Apply shifts for true outer boundaries
    p000 = c000 .- v_shift_r .- v_shift_p .- v_shift_s
    p100 = c100 .+ v_shift_r .- v_shift_p .- v_shift_s
    p010 = c010 .- v_shift_r .+ v_shift_p .- v_shift_s
    p110 = c110 .+ v_shift_r .+ v_shift_p .- v_shift_s
    p001 = c001 .- v_shift_r .- v_shift_p .+ v_shift_s
    p101 = c101 .+ v_shift_r .- v_shift_p .+ v_shift_s
    p011 = c011 .- v_shift_r .+ v_shift_p .+ v_shift_s
    p111 = c111 .+ v_shift_r .+ v_shift_p .+ v_shift_s

    # 3. Assemble Faces (Counter-clockwise)
    faces = [
        [p000, p100, p110, p010], # Bottom
        [p001, p101, p111, p011], # Top
        [p000, p100, p101, p001], # Front
        [p010, p110, p111, p011], # Back
        [p000, p010, p011, p001], # Left
        [p100, p110, p111, p101]  # Right
    ]

    # 4. Assemble output structures
    points = hcat(p000, p100, p010, p110, p001, p101, p011, p111)
    
    vectors = (
        origin = p000,
        r = p100 .- p000,
        p = p010 .- p000,
        s = p001 .- p000
    )

    return (; faces, points, vectors)
end