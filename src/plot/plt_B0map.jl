"""
    fig = plt_B0map(img::Array{<:Real, 2}; kwargs...)

# Description
    Plots an B0map using matplotlib 

# Arguments
- `img`: (`::Array{<:Real, 2}`) `[nX, nY]`, image array to be plotted

# Keywords
- `title`: (`::Integer`, `=""`) figure's suptitle
- `width`: (`::Real`, `=6`) figure's width
- `height`: (`::Real`, `=5`) figure's height
- `vmax`: (`::Real`, `=100`) maximum value to be used for window width/ window level
- `vmin`: (`::Real`, `=0`) minimum value to be used for window width/ window level
- `cmap`: (`::String`, `="gray"`) colormap to be used for plotting
- `fontsize_label`: (`::Integer`, `=10`) font size of the labels
- `fontsize_ticklabel`: (`::Integer`, `=8`) font size of the tick labels
- `pad_labeltick`: (`::Integer`, `=2`) padding between tick labels and the colorbar
- `linewidth`: (`::Real`, `=0.5`) width of the colorbar's outline
- `ticklength`: (`::Real`, `=1.5`) length of the colorbar's ticks
- `color_facecolor`: (`::String`, `="#1F1F1F"`) background color of the figure
- `color_label`: (`::String`, `="#CCCCCC"`) color of the labels

# Returns
- `Figure`: a PyObject representing the figure

# Examples
```julia-repl
julia> fig = plt_B0map(rand(100, 100))
julia> fig.savefig("B0map.png",bbox_inches="tight", pad_inches=0, transparent=true)
```
"""
function plt_B0map(
    b0map::AbstractArray{<:Real, 2};
    title              = ""       ,
    width              = 5        ,
    height             = 5        ,
    vmax               = 100      ,
    vmin               = -100     ,
    cmap               = "jet"    ,
    fontsize_label     = 7        ,
    fontsize_ticklabel = 6        ,
    pad_labeltick      = 2        ,
    linewidth          = 0.5      ,
    ticklength         = 1.5      ,
    color_facecolor    = "#1F1F1F",
    color_label        = "#CCCCCC",
    )
    fig = plt.figure(figsize=(width/0.75/2.53999863, height/0.9/2.53999863), facecolor=color_facecolor)

    ax    = fig.add_axes([0, 0, 0.75, 0.9])
    cax   = fig.add_axes([0.77, 0, 0.05*0.75, 0.9])
    ax.set_title(title, fontsize=fontsize_label, color=color_label)
    ax.set_facecolor(color_facecolor)
    ax.tick_params(axis="both", bottom=false, top=false, left=false, right=false, labelbottom=false, labeltop=false, labelleft=false, labelright=false)
    for spine in ax.spines  # "left", "right", "bottom", "top"
        ax.spines[spine].set_visible(false)
    end

    ai = ax.imshow(b0map, cmap=cmap, vmin=vmin, vmax=vmax)

    cb = fig.colorbar(ai, cax=cax)
    cb.set_label("ΔB₀ [Hz]", color=color_label, size=fontsize_ticklabel)
    cb.ax.tick_params(color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel,length=ticklength, width=linewidth, pad=pad_labeltick)
    cb.outline.set_visible(false)
    # cb.set_ticks([-120.0, -60.0, 0.0, 60.0, 120.0])
    cb.update_ticks()
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    return fig
end