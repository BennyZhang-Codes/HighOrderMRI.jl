"""
    _plot_girf_engine(G_input, G_act; kwargs...)

Internal pure plotting engine. Handles 2D matrices (3, nPoint) and visualizes 
them with boundaries, dynamic local times, and premium aesthetics.
"""
function _plot_girf_engine(
    G_input::AbstractMatrix, 
    G_act::AbstractMatrix;
    dt::Float64        = 10e-6,
    t                  :: Union{Nothing, AbstractVector} = nothing,
    boundaries         :: Vector{Int} = Int[],
    gamma              :: Float64 = 42.575575, # MHz/T, which is equal to Hz/uT
    title              = "",
    width              = 24,  
    height             = 10,  
    fontsize_title     = 13,
    fontsize_label     = 12,
    color_facecolor    = "#FFFFFF", 
    color_label        = "#000000",
    color_grid         = "#E0E0E0"
)
    ticker = pyimport("matplotlib.ticker")

    # 1. Define time vector
    nPoint = size(G_input, 2)
    if isnothing(t)
        t_vec = (0:nPoint-1) .* dt .* 1e3 # Convert to milliseconds (ms)
    else
        length(t) == nPoint || throw(ArgumentError("Length of custom time vector `t` must match gradient points ($nPoint)."))
        t_vec = t .* 1e3 # Convert to ms
    end

    # 2. Calculate accurate boundaries and start/end times for each segment
    boundary_idx = [b for b in boundaries if 1 <= b < nPoint]
    boundary_times = [t_vec[b] for b in boundary_idx]
    
    start_times = isempty(boundary_idx) ? [t_vec[1]] : [t_vec[1]; [t_vec[b+1] for b in boundary_idx]]
    end_times   = isempty(boundary_idx) ? [t_vec[end]] : [boundary_times; t_vec[end]]

    # 3. Unit Conversion: (Hz/m) to (mT/m)
    conv_factor  = gamma * 1000.0
    G_input_plot = G_input ./ conv_factor
    G_act_plot   = G_act ./ conv_factor

    # 4. Calculate Shared Y-Axis Limits for consistency
    all_data = [G_input_plot; G_act_plot]
    y_min, y_max = minimum(all_data), maximum(all_data)
    y_pad = max((y_max - y_min) * 0.1, 1e-3) # 10% padding
    shared_ylim = (y_min - y_pad, y_max + y_pad)

    # 5. Visualization Setup: 2 rows, 1 column, shared X-axis
    fig, axs = plt.subplots(2, 1, figsize=(width/2.53999863, height/2.53999863), sharex=true)
    fig.patch.set_facecolor(color_facecolor)
    
    # Premium Color Palette
    colors = ["#EF476F", "#06D6A0", "#118AB2"] # Modern Red, Green, Blue
    labels = ["Gx", "Gy", "Gz"]
    
    # Helper to plot multi-axis data beautifully
    function plot_axes!(ax, data, title_str, is_last)
        ax.set_facecolor(color_facecolor)
        ax.set_axisbelow(true) 
        ax.grid(true, linestyle="--", linewidth=0.5, color=color_grid, alpha=0.8)

        # Draw segment boundaries
        for bt in boundary_times
            ax.axvline(bt, color=color_label, linestyle=":", alpha=0.4, linewidth=1.5)
        end
        
        for i in 1:3
            ax.plot(t_vec, data[i, :], label=labels[i], color=colors[i], linewidth=1.5)
            ax.fill_between(t_vec, 0, data[i, :], color=colors[i], alpha=0.1)
        end
        
        ax.set_title(title_str, fontsize=fontsize_title, color=color_label, pad=0)
        ax.set_ylabel("[mT/m]", fontsize=fontsize_label, color=color_label)
        ax.set_ylim(shared_ylim)
        
        # X-axis label logic: dynamic local time formatting for the bottom plot
        if is_last
            ax.set_xlabel("Local Time [ms]", fontsize=fontsize_label, color=color_label)
            
            formatter = ticker.FuncFormatter((val, pos) -> begin
                idx = searchsortedlast(start_times, val)
                idx = clamp(idx, 1, length(start_times))
                local_val = val - start_times[idx]
                
                if local_val < 0.0 && local_val > -1e-5
                    local_val = 0.0
                end
                return string(round(local_val, digits=3))
            end)
            ax.xaxis.set_major_formatter(formatter)
        else
            ax.tick_params(labelbottom=false) # Hide x labels for top rows
        end
        
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        ax.spines["bottom"].set_color(color_label)
        ax.spines["left"].set_color(color_label)
        
        ax.tick_params(colors=color_label, labelsize=fontsize_label-2)
    end
    
    # Plotting the two stages vertically
    plot_axes!(axs[1], G_input_plot, "Nominal Gradient (DCS)", false)
    plot_axes!(axs[2], G_act_plot, "GIRF-Predicted Gradient (DCS)", true)
    
    axs[1].legend(fontsize=fontsize_label-2, frameon=false, loc="upper right", handlelength=1.0)
    
    # Add Segment Number Badges (1, 2, 3...) to the top plot (Top-Left)
    if length(start_times) > 1
        for (i, (st, et)) in enumerate(zip(start_times, end_times))
            margin = (et - st) * 0.01 
            axs[1].text(st + margin, shared_ylim[2], "$i",
                ha="left", va="top",
                fontsize=fontsize_label - 2, fontweight="bold",
                color=color_facecolor,
                bbox=Dict("facecolor"=>color_label, "edgecolor"=>"none", "boxstyle"=>"round,pad=0.2", "alpha"=>0.6),
                zorder=10)
        end
    end
    
    if !isempty(title)
        fig.suptitle(title, color=color_label, fontsize=fontsize_title + 4, fontweight="bold")
    end
    
    fig.tight_layout(h_pad=0.5)
    return fig
end

# ==============================================================================
# 2D Matrix Overload (Single Segment)
# ==============================================================================
"""
    plt_grad(G_DCS_Nominal::AbstractMatrix, model::GIRFModel; kwargs...)

Visualizes the Gradient Impulse Response Function (GIRF) prediction results.
"""
function plt_grad(
    G_DCS_Nominal::AbstractMatrix, 
    model::GIRFModel; 
    rbw::Float64 = 1.0,
    kwargs...
)
    dim_spatial = size(G_DCS_Nominal, 1) == 3 ? 1 : 2
    dim_time = dim_spatial == 1 ? 2 : 1
    
    # 1. Prediction
    G_act_all = apply_girf(G_DCS_Nominal, model; dim_spatial=dim_spatial, dim_time=dim_time, rbw=rbw)
    G_DCS_act, _, _ = unpack_girf_channels(G_act_all; dim_spatial=dim_spatial)

    # 2. Standardize layout (3, N)
    G_in_work  = dim_spatial == 1 ? G_DCS_Nominal : transpose(G_DCS_Nominal)
    G_act_work = dim_spatial == 1 ? G_DCS_act     : transpose(G_DCS_act)

    return _plot_girf_engine(G_in_work, G_act_work; kwargs...)
end

# ==============================================================================
# 3D Array Overload (Multi-Segment: Predict FIRST, Concatenate LATER)
# ==============================================================================
"""
    plt_grad(G_DCS_Nominal::AbstractArray{T, 3}, model::GIRFModel; dim_dynamic::Int=1, kwargs...) where T

Overload for 3D nominal gradient arrays.
Predicts each physical shot independently using multi-threading to prevent Gibbs 
ringing at dead-time boundaries, and then concatenates them for a seamless visualization.
"""
function plt_grad(
    G_DCS_Nominal::AbstractArray{T, 3}, 
    model::GIRFModel; 
    dim_dynamic::Int = 1,
    rbw::Float64 = 1.0,
    kwargs...
) where T
    # 1. Identify dimensions safely
    spatial_dim = findfirst(i -> size(G_DCS_Nominal, i) == 3 && i != dim_dynamic, 1:3)
    
    if isnothing(spatial_dim)
        spatial_dim = findfirst(==(3), size(G_DCS_Nominal))
        if isnothing(spatial_dim)
            throw(ArgumentError("G_DCS_Nominal must have exactly one dimension of size 3 (spatial axes)."))
        end
    end
    time_dim = setdiff(1:3, [spatial_dim, dim_dynamic])[1]

    # -------------------------------------------------------------------------
    # 2. 🌟 INDEPENDENT PREDICTION 🌟
    # Utilizes the multi-threading loop inside apply_girf to predict each 
    # dynamic shot completely isolated from one another!
    # -------------------------------------------------------------------------
    G_act_all = apply_girf(G_DCS_Nominal, model; dim_spatial=spatial_dim, dim_time=time_dim, rbw=rbw)
    G_DCS_act, _, _ = unpack_girf_channels(G_act_all; dim_spatial=spatial_dim)

    # 3. Permute to standardize order: (Spatial, Time, Dynamic)
    G_nom_perm = permutedims(G_DCS_Nominal, (spatial_dim, time_dim, dim_dynamic))
    G_act_perm = permutedims(G_DCS_act,     (spatial_dim, time_dim, dim_dynamic))
    
    n_spatial, n_time, n_dyn = size(G_nom_perm)

    # 4. Concatenate for Visualization (Flatten to 2D)
    G_nom_2D = reshape(G_nom_perm, 3, :)
    G_act_2D = reshape(G_act_perm, 3, :)

    # 5. Calculate boundary indices
    boundaries = [i * n_time for i in 1:(n_dyn-1)]

    # 6. Call Base Engine
    return _plot_girf_engine(G_nom_2D, G_act_2D; boundaries=boundaries, kwargs...)
end