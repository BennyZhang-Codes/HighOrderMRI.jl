"""
    generate_rps_grid(fov::Vector{<:Real}, matrix_size::Vector{<:Real}) -> Array{Float64,4}

Generate a 3D RPS (Read, Phase, Slice) grid of voxel center coordinates.

# Arguments
- `fov`: Field of view in millimeters, ordered as [FOV_R, FOV_P, FOV_S]
- `matrix_size`: Number of voxels along [R, P, S] axes

# Returns
- A 4D array of shape (nR, nP, nS, 3) where each voxel contains its RPS center coordinate
"""
function generate_rps_grid(fov::Vector{<:Real}, matrix_size::Vector{<:Real})
    if length(fov) != 3 || length(matrix_size) != 3
        error("Both fov and matrix_size must be 3-element vectors.")
    end

    # Convert to Float64
    fov = Float64.(fov)
    matrix_size = Int.(matrix_size)

    spacing = fov ./ matrix_size

    r = ((0:matrix_size[1]-1) .- (matrix_size[1]-1)/2) .* spacing[1]
    p = ((0:matrix_size[2]-1) .- (matrix_size[2]-1)/2) .* spacing[2]
    s = ((0:matrix_size[3]-1) .- (matrix_size[3]-1)/2) .* spacing[3]

    rr = reshape(r, :, 1, 1)
    pp = reshape(p, 1, :, 1)
    ss = reshape(s, 1, 1, :)

    # Broadcast to form 3D grid
    grid = Array{Float64, 4}(undef, matrix_size[1], matrix_size[2], matrix_size[3], 3)
    grid[:, :, :, 1] .= rr
    grid[:, :, :, 2] .= pp
    grid[:, :, :, 3] .= ss

    return grid
end
