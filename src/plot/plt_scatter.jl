"""
    plt_scatter(
        ys::AbstractVector{T};
        xs = nothing,
        labels = nothing,
        xlabel = "",
        ylabel = "",
        width = 15,
        height = 8,
        axis_equal = false,
        fontsize_label = 7,
        fontsize_legend = 7,
        fontsize_ticklabel = 6,
        color_facecolor = "#1F1F1F",
        color_label = "#CCCCCC",
        linewidth = 0.5,
        ticklength = 1.5,
        pad_label = 2,
        pad_labeltick = 2,
        marker = "o",
        marker_size = 1.0,
        marker_linewidth = 0.5) where T <: AbstractVector{<:Real}

Plot a vector of real-valued vectors as scatter series and return the Matplotlib
figure. Without `xs`, coordinates are Julia indices `1:length(y)`; with `xs`,
provide one coordinate vector per series or a singleton collection reused for
all series. Each x/y pair must have matching lengths.

`width` and `height` are approximately centimetres; font sizes are points.
Currently `marker`, `marker_size`, and `marker_linewidth` are accepted but not
passed to `scatter`. The general `linewidth` is passed. Do not rely on the
three marker-specific keywords changing the figure.
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