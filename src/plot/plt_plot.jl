

"""
    plt_plot(
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
        pad_labeltick = 2) where T <: AbstractVector{<:Real}

Plot a vector of real-valued vectors using Matplotlib and return the figure.
`xs` is a collection of x-coordinate vectors: use one per curve, or a single
vector wrapped in a collection to reuse it. Coordinate vectors are sliced to
the y length and must be long enough. Without `xs`, Matplotlib supplies its
zero-based index coordinates. Missing/short `labels` are replaced with blanks.

`width` and `height` are approximately centimetres: the implementation divides
by 2.53999863 before passing inches to Matplotlib. Font sizes are points.
Use `PyPlot.close(fig)` after saving when creating many figures.
"""
function plt_plot(
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
) where T<:AbstractVector{<:Real}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width/2.53999863, height/2.53999863),facecolor=color_facecolor, squeeze=false)
    ax = ax[1]
    if axis_equal
        ax.set_aspect(1)
    end
    ax.set_facecolor(color_facecolor)
    ax.tick_params(axis="both", length=ticklength, width=linewidth, pad=pad_labeltick, 
        color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel)
    for spine in ax.spines  # "left", "right", "bottom", "top"
        # ax.spines[spine].set_linewidth(linewidth)
        ax.spines[spine].set_visible(false)
    end
    if isnothing(labels)
        labels = ["" for i=1:length(ys)]
    elseif length(labels) < length(ys)
        labels = ["" for i=1:length(ys)]
    end


    if isnothing(xs)
        for idx = eachindex(ys)
            ax.plot(ys[idx], label=labels[idx], linewidth=linewidth)
        end
    else
        if length(xs) == length(ys)
            for idx = eachindex(ys)
                ax.plot(xs[idx][1:length(ys[idx])], ys[idx], label=labels[idx], linewidth=linewidth)
            end
        else
            for idx = eachindex(ys)
                ax.plot(xs[1][1:length(ys[idx])], ys[idx], label=labels[idx], linewidth=linewidth)
            end
        end
    end
    ax.set_xlabel(xlabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label, color=color_label, labelpad=pad_label)
    ax.legend(fontsize=fontsize_legend, labelcolor=color_label, scatteryoffsets=[0.5],
            # loc="upper left", bbox_to_anchor=(0.01, 1.00), ncols=5, 
            frameon=false, handlelength=1, handletextpad=0.5, columnspacing=1)

    fig.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    return fig
end
