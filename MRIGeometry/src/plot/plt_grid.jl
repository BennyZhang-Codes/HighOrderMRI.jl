"""
    plt_grid(grid::Array{<:Real,4}; step::Int=4, space::Symbol=:DCS, kwargs...)

Visualize a 3D voxel grid with subsampling using PyPlot. The color gradients 
automatically map to the Readout (Red), Phase (Green), and Slice (Blue) logical dimensions.

# Arguments
- `grid`: 4D array of shape (nRead, nPhase, nSlice, 3) containing the absolute physical coordinates of the voxels.
- `step`: Integer. Subsampling step to reduce plotted points and save memory (default: 4).
- `space`: Symbol. The coordinate space type for axis labels. Choose from `:xyz`, `:RPS`, `:DCS` (default), or `:PCS`.

# Keyword Arguments
- `title`: String. Title of the plot.
- `width`, `height`: Plot dimensions in inches.
- `fontsize_*`: Font sizes for title, labels, and ticks.
- `color_facecolor`: Background color of the plot (default: "#1F1F1F").
- `color_label`: Color for axes, ticks, and labels (default: "#CCCCCC").
- `plot_RPS_vectors`: Boolean. If true, draws the true 3D bounding vectors for the Readout (Red), Phase (Green), and Slice (Blue) directions starting from the matrix origin (1,1,1).
- `elev`: Float. Initial camera elevation angle in degrees (default: 15).
- `azim`: Float. Initial camera azimuth angle in degrees (default: 135).
"""
function plt_grid(grid::Array{T,4}; step::Int=4, space::Symbol=:DCS,   
    title              = ""       ,
    width              = 18       ,
    height             = 15       ,
    fontsize_title     = 12       ,
    fontsize_label     = 10       ,
    fontsize_tick      = 8        ,
    color_facecolor    = "#1F1F1F",
    color_label        = "#CCCCCC",
    plot_RPS_vectors   = true     ,
    elev               = 15       ,  
    azim               = 135      ,
    ) where T <: Real
    grid_sub = @views grid[1:step:end, 1:step:end, 1:step:end, :]
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
    index_grid = Array{T, 4}(undef, nx, ny, nz, 3)
    index_grid[:, :, :, 1] .= rr
    index_grid[:, :, :, 2] .= pp
    index_grid[:, :, :, 3] .= ss

    # Normalize indices to [0.2, 0.8] linearly — avoid repeating color patterns
    cmin, cmax = 0.2, 0.8
    
    r_color = nx == 1 ? [0.5] : range(cmin, cmax, length=nx)
    p_color = ny == 1 ? [0.5] : range(cmin, cmax, length=ny)
    s_color = nz == 1 ? [0.5] : range(cmin, cmax, length=nz)

    R_grid = [r for r in r_color, p in p_color, s in s_color]
    P_grid = [p for r in r_color, p in p_color, s in s_color]
    S_grid = [s for r in r_color, p in p_color, s in s_color]
    
    colors = hcat(R_grid[:], P_grid[:], S_grid[:])

    # norm_range(x) = cmin .+ ((x .- minimum(x)) ./ (maximum(x) - minimum(x) + eps())) .* (cmax - cmin)

    # # Normalize to RGB color (avoiding very light/dark)
    # x_norm = norm_range(reshape(index_grid, :, 3)[:, 1])
    # y_norm = norm_range(reshape(index_grid, :, 3)[:, 2])
    # z_norm = norm_range(reshape(index_grid, :, 3)[:, 3])
    # colors = hcat(x_norm, y_norm, z_norm)

    ax.scatter(x_mm, y_mm, z_mm, s=2, alpha=1, c=colors)

    if space == :DCS
        ax.set_xlabel("X (Right) [mm]"   , color=color_label, fontsize=fontsize_label)
        ax.set_ylabel("Y (Up) [mm]"      , color=color_label, fontsize=fontsize_label)
        ax.set_zlabel("Z (Out) [mm]"     , color=color_label, fontsize=fontsize_label)
        ax.set_title(title == "" ? "Device Coordinate System (DCS)" : title, color=color_label, fontsize=fontsize_title)
    elseif space == :PCS
        ax.set_xlabel("dSag (Left) [mm]" , color=color_label, fontsize=fontsize_label)
        ax.set_ylabel("dCor (Post) [mm]" , color=color_label, fontsize=fontsize_label)
        ax.set_zlabel("dTra (Head) [mm]" , color=color_label, fontsize=fontsize_label)
        ax.set_title(title == "" ? "Patient Coordinate System (PCS)" : title, color=color_label, fontsize=fontsize_title)
    elseif space == :RPS
        ax.set_xlabel("Readout (R) [mm]" , color=color_label, fontsize=fontsize_label)
        ax.set_ylabel("Phase (P) [mm]"   , color=color_label, fontsize=fontsize_label)
        ax.set_zlabel("Slice (S) [mm]"   , color=color_label, fontsize=fontsize_label)
        ax.set_title(title == "" ? "Readout-Phase-Slice (RPS)" : title, color=color_label, fontsize=fontsize_title)
        error("Unsupported coordinate space: choose :XYZ, :RPS, :DCS or :PCS")
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

    if plot_RPS_vectors
        # ==========================================================
        # 1. Find the "Absolute Physical Origin" of the matrix
        # Get the absolute physical coordinates of the voxel at index 
        # (1, 1, 1) and convert to mm.
        # ==========================================================
        start_pt = grid[1, 1, 1, :] .* 1e3
        cx, cy, cz = start_pt[1], start_pt[2], start_pt[3]

        # Plot the origin marker (Highlighting the starting point of the data matrix)
        ax.scatter([cx], [cy], [cz], color=color_label, s=40, label="Matrix Origin (1,1,1)")
        ax.text(cx, cy, cz, " (1,1,1)", color=color_label, fontsize=fontsize_label)

        # ==========================================================
        # 2. Extract the three bounding edges directly from the grid data
        # (These represent the true physical spanning vectors)
        # ==========================================================
        nx_full, ny_full, nz_full, _ = size(grid)

        # ---- Draw the Readout vector: from index [1, 1, 1] to [end, 1, 1] ----
        if nx_full > 1
            v_r = (grid[end, 1, 1, :] .- grid[1, 1, 1, :]) .* 1e3 * 1.1
            ax.quiver(cx, cy, cz, v_r[1], v_r[2], v_r[3], color="#FF5252", arrow_length_ratio=0.05, linewidth=1.5)
            ax.text(cx + v_r[1], cy + v_r[2], cz + v_r[3], " Readout", color="#FF5252", fontsize=fontsize_label)
        end

        # ---- Draw the Phase vector: from index [1, 1, 1] to [1, end, 1] ----
        if ny_full > 1
            v_p = (grid[1, end, 1, :] .- grid[1, 1, 1, :]) .* 1e3 * 1.1
            ax.quiver(cx, cy, cz, v_p[1], v_p[2], v_p[3], color="#00E676", arrow_length_ratio=0.05, linewidth=1.5)
            ax.text(cx + v_p[1], cy + v_p[2], cz + v_p[3], " Phase", color="#00E676", fontsize=fontsize_label)
        end

        # ---- Draw the Slice vector: from index [1, 1, 1] to [1, 1, end] ----
        if nz_full > 1
            v_s = (grid[1, 1, end, :] .- grid[1, 1, 1, :]) .* 1e3 * 1.1
            ax.quiver(cx, cy, cz, v_s[1], v_s[2], v_s[3], color="#448AFF", arrow_length_ratio=0.05, linewidth=1.5)
            ax.text(cx + v_s[1], cy + v_s[2], cz + v_s[3], " Slice", color="#448AFF", fontsize=fontsize_label)
        end
    end

    ax.set_box_aspect((1, 1, 1))

    ax.view_init(elev=elev, azim=azim)

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    return fig
end