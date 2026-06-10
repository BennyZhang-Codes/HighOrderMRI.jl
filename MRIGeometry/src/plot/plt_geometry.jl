"""
    plt_geometry(geos::Vector{<:Geometry}; space::Symbol=:DCS, kwargs...)

Visualize the macroscopic 3D Field of View (FOV) Bounding Boxes for an array of `Geometry` objects.
Highly optimized: Computes only the 8 extreme corner points per geometry instead of generating full dense grids.

# Arguments
- `geos`: A single `Geometry` or a Vector of `Geometry` objects.
- `space`: Symbol. Target coordinate space (`:DCS`, `:PCS`, or `:RPS`). Default is `:DCS`.

# Keyword Arguments
- `plot_RPS_vectors`: Boolean. Draws the RPS physical spanning vectors (default: false).
- `vol_alpha`: Float. Transparency of the volume faces (default: 0.15).
- `colors`: Vector of Strings. Palette used for different geometries.
"""
function plt_geometry(geos::Vector{<:Geometry}; 
    space::Symbol      = :DCS,
    plot_RPS_vectors   = false,
    title              = "",
    width              = 6,
    height             = 6,
    fontsize_title     = 10,
    fontsize_label     = 8,
    fontsize_tick      = 6,
    color_facecolor    = "#1F1F1F",
    color_label        = "#CCCCCC",
    vol_alpha          = 0.15,
    colors             = ["#4488FF", "#FF8844", "#44FF88", "#FF44FF", "#FFFF44", "#44FFFF"])

    # Setup PyPlot Figure
    fig = figure(figsize=(width/2.53999863, height/2.53999863), facecolor=color_facecolor)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(color_facecolor)
    
    # Theme colors for axes
    ax.xaxis.line.set_color(color_label); ax.yaxis.line.set_color(color_label); ax.zaxis.line.set_color(color_label)
    ax.xaxis.pane.set_color(color_label); ax.yaxis.pane.set_color(color_label); ax.zaxis.pane.set_color(color_label)
    ax.xaxis.pane.set_edgecolor(color_label); ax.yaxis.pane.set_edgecolor(color_label); ax.zaxis.pane.set_edgecolor(color_label)
    ax.tick_params(axis="x", colors=color_label, labelsize=fontsize_tick)
    ax.tick_params(axis="y", colors=color_label, labelsize=fontsize_tick)
    ax.tick_params(axis="z", colors=color_label, labelsize=fontsize_tick)

    art3d = pyimport("mpl_toolkits.mplot3d.art3d")
    global_points = Array{Float64}(undef, 3, 0) # Track points for global adaptive bounding box

    # Loop through each Geometry
    for (i, geo) in enumerate(geos)
        # ==========================================================
        # 1. Extreme Optimization: Build a 2x2x2 miniature RPS extreme grid
        # ==========================================================
        spacing = geo.FOV ./ geo.MatrixSize
        
        # Get the extreme coordinates [min, max] of the logical grid
        r_ext = [-(geo.MatrixSize[1]-1)/2, (geo.MatrixSize[1]-1)/2] .* spacing[1]
        p_ext = [-(geo.MatrixSize[2]-1)/2, (geo.MatrixSize[2]-1)/2] .* spacing[2]
        s_ext = [-(geo.MatrixSize[3]-1)/2, (geo.MatrixSize[3]-1)/2] .* spacing[3]

        grid_rps_tiny = zeros(Float64, 2, 2, 2, 3)
        for (ir, r) in enumerate(r_ext), (ip, p) in enumerate(p_ext), (is, s) in enumerate(s_ext)
            grid_rps_tiny[ir, ip, is, :] = [r, p, s]
        end

        # ==========================================================
        # 2. Transform the miniature grid to the target physical space (DCS / PCS)
        # ==========================================================
        if space == :DCS
            R_mat = geo.R_RPS_DCS
            T_vec = geo.T_DCS
        elseif space == :PCS
            R_mat = geo.R_RPS_PCS
            T_vec = geo.T_PCS
        elseif space == :RPS || space == :rps
            R_mat = Matrix{Float64}(I, 3, 3)
            T_vec = [0.0, 0.0, 0.0]
        else
            error("Unsupported space. Choose :DCS, :PCS, or :RPS.")
        end

        # Flatten for batch affine transformation
        grid_flat = reshape(grid_rps_tiny, :, 3)
        grid_target_flat = grid_flat * transpose(R_mat) .+ T_vec'
        grid_target = reshape(grid_target_flat, 2, 2, 2, 3)

        # ==========================================================
        # 3. Extract the true physical faces and vectors
        # (Supports perfect half-pixel extension via spacing and R_mat)
        # ==========================================================
        fov = get_fov_faces(grid_target, spacing=spacing, R=R_mat)
        global_points = hcat(global_points, fov.points)

        # ==========================================================
        # 4. Render the cuboids (Poly3DCollection)
        # ==========================================================
        face_color = colors[(i - 1) % length(colors) + 1]
        poly = art3d.Poly3DCollection(fov.faces, alpha=vol_alpha, 
                                      facecolor=face_color, 
                                      edgecolor=face_color, 
                                      linewidths=1.0)
        ax.add_collection3d(poly)

        # ==========================================================
        # 5. Render physical orientation axes (conditionally based on MatrixSize)
        # ==========================================================
        if plot_RPS_vectors
            orig = fov.vectors.origin
            vr, vp, vs = fov.vectors.r, fov.vectors.p, fov.vectors.s
            
            # Mark the origin
            ax.scatter([orig[1]], [orig[2]], [orig[3]], color=face_color, s=20)
            
            # Draw the Readout vector
            if geo.MatrixSize[1] > 1
                ax.quiver(orig[1], orig[2], orig[3], vr[1], vr[2], vr[3], color="#FF5555", arrow_length_ratio=0.1)
                ax.text(orig[1]+vr[1], orig[2]+vr[2], orig[3]+vr[3], " Readout", color="#FF5555", fontsize=fontsize_label)
            end
            
            # Draw the Phase vector
            if geo.MatrixSize[2] > 1
                ax.quiver(orig[1], orig[2], orig[3], vp[1], vp[2], vp[3], color="#55FF55", arrow_length_ratio=0.1)
                ax.text(orig[1]+vp[1], orig[2]+vp[2], orig[3]+vp[3], " Phase", color="#55FF55", fontsize=fontsize_label)
            end
            
            # Draw the Slice vector
            if geo.MatrixSize[3] > 1
                ax.quiver(orig[1], orig[2], orig[3], vs[1], vs[2], vs[3], color="#5555FF", arrow_length_ratio=0.1)
                ax.text(orig[1]+vs[1], orig[2]+vs[2], orig[3]+vs[3], " Slice", color="#5555FF", fontsize=fontsize_label)
            end
        end
    end

    # ==========================================================
    # 6. Axes labels and Adaptive Viewport
    # ==========================================================
    if space == :DCS
        ax.set_xlabel("X (Right) [mm]", color=color_label, fontsize=fontsize_label)
        ax.set_ylabel("Y (Up) [mm]", color=color_label, fontsize=fontsize_label)
        ax.set_zlabel("Z (Out) [mm]", color=color_label, fontsize=fontsize_label)
        ax.set_title(title == "" ? "Device Coordinate System (DCS)" : title, color=color_label, fontsize=fontsize_title)
    elseif space == :PCS
        ax.set_xlabel("dSag (Left) [mm]", color=color_label, fontsize=fontsize_label)
        ax.set_ylabel("dCor (Post) [mm]", color=color_label, fontsize=fontsize_label)
        ax.set_zlabel("dTra (Head) [mm]", color=color_label, fontsize=fontsize_label)
        ax.set_title(title == "" ? "Patient Coordinate System (PCS)" : title, color=color_label, fontsize=fontsize_title)
    elseif space == :RPS
            ax.set_xlabel("Readout (R) [mm]", color=color_label, fontsize=fontsize_label)
            ax.set_ylabel("Phase (P) [mm]"  , color=color_label, fontsize=fontsize_label)
            ax.set_zlabel("Slice (S) [mm]"  , color=color_label, fontsize=fontsize_label)
            ax.set_title(title == "" ? "Readout-Phase-Slice (RPS)" : title, color=color_label, fontsize=fontsize_title)
    end

    # Calculate the maximum bounding box based on all collected vertices
    xmin, xmax = minimum(global_points[1, :]), maximum(global_points[1, :])
    ymin, ymax = minimum(global_points[2, :]), maximum(global_points[2, :])
    zmin, zmax = minimum(global_points[3, :]), maximum(global_points[3, :])

    xrange = xmax - xmin; yrange = ymax - ymin; zrange = zmax - zmin
    padded_range = maximum([xrange, yrange, zrange, eps()]) * 1.3 

    xmid = (xmin + xmax) / 2; ymid = (ymin + ymax) / 2; zmid = (zmin + zmax) / 2

    ax.set_xlim(xmid - padded_range/2, xmid + padded_range/2)
    ax.set_ylim(ymid - padded_range/2, ymid + padded_range/2)
    ax.set_zlim(zmid - padded_range/2, zmid + padded_range/2)

    fig.tight_layout(pad=0.1)
    return fig
end

# Convenience interface: allow passing a single Geometry object without an array wrapper
plt_geometry(geo::Geometry; kwargs...) = plt_geometry([geo]; kwargs...)