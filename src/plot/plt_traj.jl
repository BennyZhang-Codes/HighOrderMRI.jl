"""
    fig = plt_traj(ktraj::AbstractArray{<:Real, 2}; kwargs...)

# Description
    Plots a first-order k-trajectory using matplotlib 

# Arguments
- `ktraj::AbstractArray{<:Real, 2}`: `[3, nPoint]` trajectory in unit [m⁻¹] to be plotted 

# Keywords
- `width`: (`::Real`, `=5`) figure's width
- `height`: (`::Real`, `=5`) figure's height
- `markersize`: (`::Real`, `=0.5`) size of the markers
- `cmap`: (`::String`, `="gray"`) colormap to be used for plotting
- `fontsize_label`: (`::Integer`, `=6`) font size of the labels
- `fontsize_ticklabel`: (`::Integer`, `=4`) font size of the tick labels
- `color_facecolor`: (`::String`, `="#ffffff"`) background color of the figure
- `color_label`: (`::String`, `="#000000"`) color of the labels and tick labels
- `linewidth`: (`::Real`, `=0.5`) width of the lines
- `ticklength`: (`::Real`, `=1.5`) length of the ticks
- `pad_label`: (`::Real`, `=2`) padding of the labels
- `pad_labeltick`: (`::Real`, `=2`) padding of the tick labels

# Returns
- `Figure`: a PyObject representing the figure

# Examples
```julia-repl
julia> fig = plt_image(rand(100, 100))
julia> fig.savefig("123.png",bbox_inches="tight", pad_inches=0, transparent=true)
```
"""
function plt_traj(
    ktraj::AbstractArray{<:Real, 2};
    width              = 8        ,
    height             = 8        ,
    markersize         = 0.5      ,
    cmap               = "viridis",
    xlabel             = L"k_x \quad [m^{-1}]",
    ylabel             = L"k_y \quad [m^{-1}]",
    fontsize_label     = 7        ,
    fontsize_ticklabel = 6        ,
    color_facecolor    = "#1F1F1F",
    color_label        = "#CCCCCC",
    linewidth          = 0.5      ,
    ticklength         = 1.5      ,
    pad_label          = 2        ,
    pad_labeltick      = 2        ,
    )
    matplotlib.rc("mathtext", default="regular")
    nPoint, nTerm = size(ktraj)
    @assert nTerm == 3 "ktraj should be a 2D array with 3 rows for [kx, ky, kz]"
    kx, ky, kz = ktraj[:, 1], ktraj[:, 2], ktraj[:, 3]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width/2.53999863, height/2.53999863), facecolor=color_facecolor)
    ax.set_aspect("equal", "box")
    ax.set_facecolor(color_facecolor)
    ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, 
        color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
    for spine in ax.spines  # "left", "right", "bottom", "top"
        ax.spines[spine].set_visible(false)
    end
    ax.scatter(kx, ky, linewidths=linewidth, marker=".", s=markersize, c=collect(1:nPoint), cmap=cmap)

    ax.set_xlabel(xlabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)

    fig.tight_layout(pad=0.1, w_pad=0, h_pad=0)
    return fig
end

function plt_traj(
    trajs::AbstractVector{T};
    labels             = nothing  ,
    xlabel             = L"k_x \quad [m^{-1}]",
    ylabel             = L"k_y \quad [m^{-1}]",
    width              = 8        ,      
    height             = 8        ,  
    fontsize_label     = 7        ,
    fontsize_legend    = 7        ,
    fontsize_ticklabel = 6        ,
    color_facecolor    = "#1F1F1F",
    color_label        = "#CCCCCC",
    linewidth          = 0.5      ,
    ticklength         = 1.5      ,
    pad_label          = 2        ,
    pad_labeltick      = 2        ,
) where T<:AbstractArray{<:Real, 2}
    matplotlib.rc("mathtext", default="regular")
    if !isnothing(labels)
        @assert length(trajs) == length(labels) "labels should have the same length as ys"
    else
        labels = ["traj $(idx)" for idx = 1:length(trajs)] 
    end
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width/2.53999863, height/2.53999863), facecolor=color_facecolor)
    ax.set_aspect("equal", "box")
    ax.set_facecolor(color_facecolor)
    ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, 
        color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
    for spine in ax.spines  # "left", "right", "bottom", "top"
        # ax.spines[spine].set_color(color_label)
        # ax.spines[spine].set_linewidth(linewidth)
        ax.spines[spine].set_visible(false)
    end

    for (traj, label) in zip(trajs, labels)
        ax.plot(traj[:,1], traj[:,2], label=label, linewidth=linewidth)
    end
    ax.set_xlabel(xlabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.legend(fontsize=fontsize_legend, labelcolor=color_label, scatteryoffsets=[0.5],
            # loc="upper left", bbox_to_anchor=(0.01, 1.00), ncols=5, 
            frameon=false, handlelength=1, handletextpad=0.5, columnspacing=1)
    fig.tight_layout(pad=0.1, w_pad=0, h_pad=0)
    return fig
end
