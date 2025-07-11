"""
    fig = plt_scatter(ys::Vector{AbstractArray{Float64}}; labels=nothing, xs=nothing, xlabel="", ylabel="", width=15, height=8, fontsize_label=7, fontsize_legend=7, fontsize_ticklabel=6, color_facecolor="#1F1F1F", color_label="#CCCCCC", linewidth=0.5, ticklength=1.5, pad_label=2, pad_labeltick=2)

# Description
    Plots multiple scatter plots on the same figure.

# Arguments
- `ys::Vector{AbstractArray{Float64}}`: a vector of arrays of y-values to plot

# Keywords
- `xs`: a vector of arrays of x-values to plot. If `nothing`, the x-values are assumed to be the same for all scatter plots.
- `labels`: a vector of labels for each scatter plot. If `nothing`, the labels are not shown.
- `xlabel`: the label for the x-axis.
- `ylabel`: the label for the y-axis.
- `width`: the width of the figure in inches.
- `height`: the height of the figure in inches.
- `fontsize_label`: the font size for the labels.
- `fontsize_legend`: the font size for the legend.
- `fontsize_ticklabel`: the font size for the tick labels.
- `color_facecolor`: the background color of the figure.
- `color_label`: the color of the labels.
- `linewidth`: the width of the markers.
- `ticklength`: the length of the ticks.
- `pad_label`: the padding for the labels.
- `pad_labeltick`: the padding for the tick labels.

# Returns
- `Figure`: a PyObject representing the figure

# Examples
```julia-repl
julia> fig = plt_scatter([rand(100), rand(100)])
"""
function plt_scatter( 
    ys::AbstractVector{T};
    xs                 = nothing  ,
    labels             = nothing  ,
    xlabel             = ""       ,
    ylabel             = ""       ,
    width              = 15       ,      
    height             = 8        ,  
    axis_equal         = false    ,
    fontsize_label     = 7        ,
    fontsize_legend    = 7        ,
    fontsize_ticklabel = 6        ,
    color_facecolor    = "#1F1F1F",
    color_label        = "#CCCCCC",
    linewidth          = 0.5      ,
    ticklength         = 1.5      ,
    pad_label          = 2        ,
    pad_labeltick      = 2        ,
    marker             = "o"      ,
    marker_size        = 1.0      ,
    marker_linewidth   = 0.5      ,
    ) where T<:AbstractVector{<:Real}
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width/2.53999863, height/2.53999863), facecolor=color_facecolor, squeeze=false)
    ax = ax[1]
    if axis_equal
        ax.set_aspect(1)
    end
    ax.set_facecolor(color_facecolor)
    ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, 
        color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
    for spine in ax.spines  # "left", "right", "bottom", "top"
        ax.spines[spine].set_visible(false)
    end
    if isnothing(labels)
        labels = ["" for i=1:length(ys)]
    elseif length(labels) < length(ys)
        labels = ["" for i=1:length(ys)]
    end

    if isnothing(xs)
        for idx = eachindex(ys)
            ax.scatter(1:length(ys[idx]), ys[idx], label=labels[idx], linewidth=linewidth)
        end
    else
        if length(xs) == length(ys)
            for idx = eachindex(ys)
                ax.scatter(xs[idx], ys[idx], label=labels[idx], linewidth=linewidth)
            end
        else
            for idx = eachindex(ys)
                ax.scatter(xs[1], ys[idx], label=labels[idx], linewidth=linewidth)
            end
        end
    end

    ax.set_xlabel(xlabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.legend(fontsize=fontsize_legend, labelcolor=color_label, scatteryoffsets=[0.5],
            frameon=false, handlelength=1, handletextpad=0.5, columnspacing=1)

    fig.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    return fig
end