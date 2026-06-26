using PyCall
using PyPlot
using LinearAlgebra

"""
    plt_grad(G_nominal::AbstractMatrix, geo::Geometry; kwargs...)

Visualizes the transformation from nominal gradient design to physical (DCS) gradients.
Delegates the mathematical mapping entirely to `grad_Nominal2DCS`.
Units are automatically converted from Hz/m to mT/m.

# Layout
Generates a 3x1 subplot figure (Vertical stack) with a shared Y-axis and bottom-only X-axis:
1. **Nominal DCS**: The raw gradient waveforms directly from the sequence design.
2. **Nominal RPS**: The logical gradient waveforms after applying the vendor-specific baseline polarity mapping.
3. **Actual DCS**: The true physical gradients applied to the scanner coils.

# Arguments
- `G_nominal::AbstractMatrix`: The input nominal gradient array. Must have one dimension of size 3 (spatial axes). 
- `geo::Geometry`: The `Geometry` object containing spatial transformations.

# Keyword Arguments
- `dt::Float64`: Dwell time in seconds. Default is `10e-6`.
- `t::AbstractVector`: Optional custom time vector in seconds. Overrides `dt` if provided.
- `boundaries::Vector{Int}`: Indices for drawing vertical dashed lines to separate segments.
- `gamma::Float64`: Gyromagnetic ratio in MHz/T. Default is `42.575575`.
- `title::String`: Figure title.
"""
function plt_grad(
    G_nominal::AbstractMatrix, 
    geo::Geometry; 
    dt::Float64        = 10e-6,
    t                  :: Union{Nothing, AbstractVector} = nothing,
    boundaries         :: Vector{Int} = Int[],
    gamma              :: Float64 = 42.575575, # MHz/T, which is equal to Hz/uT
    title              = "",
    width              = 24,  
    height             = 15,  
    fontsize_title     = 13,
    fontsize_label     = 12,
    color_facecolor    = "#FFFFFF", 
    color_label        = "#000000",
)
    ticker = pyimport("matplotlib.ticker")

    # 1. Auto-detect spatial dimension and transpose if necessary
    if size(G_nominal, 1) == 3
        G_work = G_nominal
    elseif size(G_nominal, 2) == 3
        G_work = transpose(G_nominal)
    else
        throw(ArgumentError("G_nominal must have exactly one dimension of size 3 (spatial axes). Found size $(size(G_nominal))."))
    end

    # 2. Define time vector
    nPoint = size(G_work, 2)
    if isnothing(t)
        t_vec = (0:nPoint-1) .* dt .* 1e3 # Convert to milliseconds (ms)
    else
        length(t) == nPoint || throw(ArgumentError("Length of custom time vector `t` ($(length(t))) must match the total number of gradient points ($nPoint)."))
        t_vec = t .* 1e3 # Convert to ms
    end

    # 3. Calculate accurate boundaries and start/end times for each segment
    boundary_idx = [b for b in boundaries if 1 <= b < nPoint]
    boundary_times = [t_vec[b] for b in boundary_idx]
    
    # Each segment's local time resets at these start_times
    start_times = isempty(boundary_idx) ? [t_vec[1]] : [t_vec[1]; [t_vec[b+1] for b in boundary_idx]]
    end_times   = isempty(boundary_idx) ? [t_vec[end]] : [boundary_times; t_vec[end]]

    # 4. Conversion factor: (Hz/m) to (mT/m)
    conv_factor = gamma * 1000.0

    # 5. Delegate Core Mathematical Transformations
    G_dcs_input  = G_work ./ conv_factor
    G_rps_design = (geo.R_Nominal_RPS * G_work) ./ conv_factor
    G_physical   = grad_Nominal2DCS(G_work, geo) ./ conv_factor

    # 6. Calculate Shared Y-Axis Limits for consistency
    all_data = [G_dcs_input; G_rps_design; G_physical]
    y_min, y_max = minimum(all_data), maximum(all_data)
    y_pad = max((y_max - y_min) * 0.1, 1e-3) # 10% padding
    shared_ylim = (y_min - y_pad, y_max + y_pad)

    # 7. Visualization Setup: 3 rows, 1 column, shared X-axis
    fig, axs = plt.subplots(3, 1, figsize=(width/2.53999863, height/2.53999863), sharex=true)
    fig.patch.set_facecolor(color_facecolor)
    
    # Premium Color Palette
    colors = ["#EF476F", "#06D6A0", "#118AB2"] # Modern Red, Green, Blue
    labels = ["Gx", "Gy", "Gz"]
    
    # Helper to plot multi-axis data beautifully
    function plot_axes!(ax, data, title_str, is_last)
        ax.set_facecolor(color_facecolor)
        
        ax.set_axisbelow(true) 

        # Draw segment boundaries
        for bt in boundary_times
            ax.axvline(bt, color=color_label, linestyle=":", alpha=0.4, linewidth=1.5)
        end
        
        for i in 1:3
            # Main Line
            ax.plot(t_vec, data[i, :], label=labels[i], color=colors[i], linewidth=1.5)
            # Area Fill
            ax.fill_between(t_vec, 0, data[i, :], color=colors[i], alpha=0.1)
        end
        
        ax.set_title(title_str, fontsize=fontsize_title, color=color_label, pad=0)
        ax.set_ylabel("[mT/m]", fontsize=fontsize_label, color=color_label)
        
        # Apply shared Y-axis limits
        ax.set_ylim(shared_ylim)
        
        # X-axis label logic: dynamic local time formatting for the bottom plot
        if is_last
            ax.set_xlabel("Local Time [ms]", fontsize=fontsize_label, color=color_label)
            
            # Custom Formatter: Reset time to 0 at the start of each segment
            formatter = ticker.FuncFormatter((val, pos) -> begin
                idx = searchsortedlast(start_times, val)
                idx = clamp(idx, 1, length(start_times))
                local_val = val - start_times[idx]
                
                # Prevent floating point noise (-0.00001 becoming negative)
                if local_val < 0.0 && local_val > -1e-5
                    local_val = 0.0
                end
                return string(round(local_val, digits=3))
            end)
            ax.xaxis.set_major_formatter(formatter)
        else
            ax.tick_params(labelbottom=false) # Hide x labels for top rows
        end
        
        # Despine
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        ax.spines["bottom"].set_color(color_label)
        ax.spines["left"].set_color(color_label)
        
        ax.tick_params(colors=color_label, labelsize=fontsize_label-2)
    end
    
    # Plotting the three stages vertically
    plot_axes!(axs[1], G_dcs_input, "Nominal Gradient (DCS - Transversal)", false)
    plot_axes!(axs[2], G_rps_design, "Nominal Gradient (RPS - Logical)", false)
    plot_axes!(axs[3], G_physical, "Actual Gradient Waveform (DCS)", true)
    
    # Add a borderless legend to the first plot
    axs[1].legend(fontsize=fontsize_label-2, frameon=false, loc="upper right", handlelength=1.0)
    
    # =========================================================================
    # Add Segment Number Badges (1, 2, 3...) to the top plot (Top-Left)
    # (Only draw if there are multiple segments)
    # =========================================================================
    if length(start_times) > 1
        for (i, (st, et)) in enumerate(zip(start_times, end_times))
            # Optional tiny margin so the badge doesn't stick exactly to the dashed line
            margin = (et - st) * 0.01 
            
            axs[1].text(st + margin, shared_ylim[2], "$i",
                ha="left", va="top",
                fontsize=fontsize_label - 2, fontweight="bold",
                color=color_facecolor,
                bbox=Dict("facecolor"=>color_label, "edgecolor"=>"none", "boxstyle"=>"round,pad=0.2", "alpha"=>0.6),
                zorder=10)
        end
    end
    
    # Global title rendering
    if !isempty(title)
        fig.suptitle(title, color=color_label, fontsize=fontsize_title + 4, fontweight="bold")
    end
    
    fig.tight_layout(h_pad=0.5)
    
    return fig
end

"""
    plt_grad(G_nominal::AbstractArray{T, 3}, geo::Geometry; dim_dynamic::Int=3, kwargs...) where T

Overload for 3D gradient arrays (e.g., `[Spatial, Time, Dynamic]`).
Automatically permutes and concatenates the array along the `dim_dynamic` axis into a 2D matrix, 
and passes it to the core 2D `plt_grad` function.

Automatically calculates segment boundaries (vertical dashed lines) for each dynamic segment.
"""
function plt_grad(
    G_nominal::AbstractArray{T, 3}, 
    geo::Geometry; 
    dim_dynamic::Int = 1,
    kwargs...
) where T
    # 1. Identify dimensions safely
    # Find the spatial dimension (length 3), strictly excluding the dynamic dimension
    spatial_dim = findfirst(i -> size(G_nominal, i) == 3 && i != dim_dynamic, 1:3)
    
    if isnothing(spatial_dim)
        # Fallback if both dynamic and spatial dims happen to be length 3
        spatial_dim = findfirst(==(3), size(G_nominal))
        if isnothing(spatial_dim)
            throw(ArgumentError("G_nominal must have exactly one dimension of size 3 (spatial axes)."))
        end
    end

    time_dim = setdiff(1:3, [spatial_dim, dim_dynamic])[1]

    # 2. Permute to standardize order: (Spatial, Time, Dynamic)
    G_perm = permutedims(G_nominal, (spatial_dim, time_dim, dim_dynamic))
    
    n_spatial, n_time, n_dyn = size(G_perm)

    # 3. Reshape into 2D: (Spatial, Time * Dynamic) 
    # Because Julia is column-major, this perfectly appends them in order!
    G_2D = reshape(G_perm, 3, :)

    # 4. Calculate boundary indices (every `n_time` points)
    boundaries = [i * n_time for i in 1:(n_dyn-1)]

    # 5. Hand over to the base 2D function
    return plt_grad(G_2D, geo; boundaries=boundaries, kwargs...)
end