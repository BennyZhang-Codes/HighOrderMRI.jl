"""
    plt_grid(grid::Array{<:Real,4}; step::Int=4, space::Symbol=:xyz)

Visualize a 3D voxel grid with subsampling using PyPlot.

# Arguments
- `grid`: 4D array of shape (nRead, nPhase, nSlice, 3), voxel center coordinates
- `step`: subsampling step to reduce plotted points (default: 4)
- `space`: coordinate space type, either `:xyz` (default) or `:rps`
"""
function plt_grid(grid::Array{<:Real,4}; step::Int=4, space::Symbol=:xyz,   
    title              = ""       ,
    width              = 5        ,
    height             = 5        ,
    fontsize_title     = 10       ,
    fontsize_label     = 8        ,
    fontsize_tick      = 6        ,
    color_facecolor    = "#1F1F1F",
    color_label        = "#CCCCCC",
    plot_reference     = true     )
    grid_sub = grid[1:step:end, 1:step:end, 1:step:end, :]
    coords = reshape(grid_sub, :, 3)
    x, y, z = coords[:, 1], coords[:, 2], coords[:, 3]

    fig = figure(figsize=(width/2.53999863, height/2.53999863), facecolor=color_facecolor)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(color_facecolor)

    # Axes colors
    ax.xaxis.line.set_color(color_label)
    ax.yaxis.line.set_color(color_label)
    ax.zaxis.line.set_color(color_label)

    ax.xaxis.pane.set_color(color_label)
    ax.yaxis.pane.set_color(color_label)
    ax.zaxis.pane.set_color(color_label)
    ax.xaxis.pane.set_edgecolor(color_label)
    ax.yaxis.pane.set_edgecolor(color_label)
    ax.zaxis.pane.set_edgecolor(color_label)
    # Tick colors
    ax.tick_params(axis="x", colors=color_label, labelsize=fontsize_tick)
    ax.tick_params(axis="y", colors=color_label, labelsize=fontsize_tick)
    ax.tick_params(axis="z", colors=color_label, labelsize=fontsize_tick)

    # Convert to mm
    x_mm, y_mm, z_mm = x * 1e3, y * 1e3, z * 1e3

    # Get subsampled grid dimensions
    nx, ny, nz, _ = size(grid_sub)
    r = 1:nx
    p = 1:ny
    s = 1:nz

    rr = reshape(r, :, 1, 1)
    pp = reshape(p, 1, :, 1)
    ss = reshape(s, 1, 1, :)

    # Broadcast to form 3D grid
    grid = Array{Float64, 4}(undef, nx, ny, nz, 3)
    grid[:, :, :, 1] .= rr
    grid[:, :, :, 2] .= pp
    grid[:, :, :, 3] .= ss

    # Normalize indices to [0.2, 0.8] linearly — avoid repeating color patterns
    cmin, cmax = 0.2, 0.8

    norm_range(x) = cmin .+ ((x .- minimum(x)) ./ (maximum(x) - minimum(x) + eps())) .* (cmax - cmin)

    # Normalize to RGB color (avoiding very light/dark)
    x_norm = norm_range(reshape(grid, :, 3)[:, 1])
    y_norm = norm_range(reshape(grid, :, 3)[:, 2])
    z_norm = norm_range(reshape(grid, :, 3)[:, 3])
    colors = hcat(x_norm, y_norm, z_norm)

    ax.scatter(x_mm, y_mm, z_mm, s=2, alpha=1, c=colors)

    if space == :xyz
        ax.set_xlabel("X [mm]"       , color=color_label, fontsize=fontsize_label)
        ax.set_ylabel("Y [mm]"       , color=color_label, fontsize=fontsize_label)
        ax.set_zlabel("Z [mm]"       , color=color_label, fontsize=fontsize_label)
        ax.set_title("XYZ Voxel Grid", color=color_label, fontsize=fontsize_title)
    elseif space == :rps
        ax.set_xlabel("Readout (R) [mm]", color=color_label, fontsize=fontsize_label)
        ax.set_ylabel("Phase (P) [mm]"  , color=color_label, fontsize=fontsize_label)
        ax.set_zlabel("Slice (S) [mm]"  , color=color_label, fontsize=fontsize_label)
        ax.set_title("RPS Voxel Grid"   , color=color_label, fontsize=fontsize_title)
    else
        error("Unsupported coordinate space: choose :xyz or :rps")
    end

    # Equal aspect ratio
    xplot, yplot, zplot = x_mm, y_mm, z_mm
    xrange = maximum(xplot) - minimum(xplot)
    yrange = maximum(yplot) - minimum(yplot)
    zrange = maximum(zplot) - minimum(zplot)
    maxrange = maximum([xrange, yrange, zrange])
    xmid = mean(extrema(xplot))
    ymid = mean(extrema(yplot))
    zmid = mean(extrema(zplot))
    ax.set_xlim(xmid - maxrange/2, xmid + maxrange/2)
    ax.set_ylim(ymid - maxrange/2, ymid + maxrange/2)
    ax.set_zlim(zmid - maxrange/2, zmid + maxrange/2)
    if plot_reference
        # Draw origin marker
        ax.scatter([0], [0], [0], color="#FFD700", s=15, label="Origin")

        # Draw coordinate axes as lines (instead of arrows)
        axis_len = 100
        ax.plot([0, axis_len], [0, 0], [0, 0], color="r", label="X")
        ax.plot([0, 0], [0, axis_len], [0, 0], color="g", label="Y")
        ax.plot([0, 0], [0, 0], [0, axis_len], color="b", label="Z")
    end

    fig.tight_layout(pad=0.1)
    return fig
end
