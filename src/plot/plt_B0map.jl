"""
    fig = plt_B0map(b0map::AbstractArray{<:Real, 2}; kwargs...)

# Description
Plots an B0map using matplotlib with a precise colorbar layout.

# Arguments
- `b0map`: (`::AbstractArray{<:Real, 2}`) `[nX, nY]`, image array to be plotted

# Keywords
- `title`: (`::String`, `=""`) figure's suptitle
- `width`: (`::Real`, `=5`) figure's width (unit in centimeters roughly). The height is automatically scaled based on the image's aspect ratio.
- `vmax`: (`::Real`, `=100`) maximum value to be used for window width/ window level
- `vmin`: (`::Real`, `=-100`) minimum value to be used for window width/ window level
- `cmap`: (`::String`, `="jet"`) colormap to be used for plotting
- `fontsize_label`: (`::Integer`, `=7`) font size of the labels
- `fontsize_ticklabel`: (`::Integer`, `=6`) font size of the tick labels
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
julia> fig.savefig("B0map.png", bbox_inches="tight", pad_inches=0, transparent=true)
```
"""
function plt_B0map(
    b0map::AbstractArray{<:Real, 2};
    title              = ""       ,
    width              = 2        ,
    vmax               = 100      ,
    vmin               = -100     ,
    cmap               = "jet"    ,
    fontsize_label     = 9        ,
    fontsize_ticklabel = 8        ,
    pad_labeltick      = 2        ,
    linewidth          = 0.5      ,
    ticklength         = 1.5      ,
    color_facecolor    = "#1F1F1F",
    color_label        = "#CCCCCC",
    )
    nX, nY = size(b0map)
    img_height = (nX/nY) * width
    
    fig_width_in  = (width / 0.75) / 2.53999863
    fig_height_in = (img_height / 0.9) / 2.53999863

    fig = plt.figure(figsize=(fig_width_in, fig_height_in), facecolor=color_facecolor)

    ax    = fig.add_axes([0.05, 0.05, 0.75, 0.9])
    cax   = fig.add_axes([0.82, 0.05, 0.05*0.75, 0.9])
    
    ax.set_title(title, fontsize=fontsize_label, color=color_label)
    ax.set_facecolor(color_facecolor)
    ax.tick_params(axis="both", bottom=false, top=false, left=false, right=false, labelbottom=false, labeltop=false, labelleft=false, labelright=false)
    for spine in ["left", "right", "bottom", "top"]
        ax.spines[spine].set_visible(false)
    end

    ai = ax.imshow(b0map, cmap=cmap, vmin=vmin, vmax=vmax)

    cb = fig.colorbar(ai, cax=cax)
    cb.set_label("ΔB₀ [Hz]", color=color_label, size=fontsize_ticklabel)
    cb.ax.tick_params(color=color_label, labelcolor=color_label, labelsize=fontsize_ticklabel, length=ticklength, width=linewidth, pad=pad_labeltick)
    cb.outline.set_visible(false)
    cb.update_ticks()
    
    # fig.tight_layout() is not used here because we manually positioned axes via add_axes
    return fig
end


"""
    fig = plt_B0map(b0map::AbstractArray{<:Real, 3}; dim=1, nRow=nothing, nCol=nothing, kwargs...)

Plots a sequence of B0map arrays as a mosaic using matplotlib by overloading `plt_B0map`.
It stitches the 3D array into a 2D mosaic and then calls the 2D `plt_B0map` method.

# Arguments
- `b0map`: (`::AbstractArray{<:Real, 3}`) image array to be plotted
- `dim`: (`::Int`, `=1`) dimension along which to plot the images

# Keywords
- `nRow`: (`::Union{Integer, Nothing}`, `=nothing`) number of rows in the mosaic grid
- `nCol`: (`::Union{Integer, Nothing}`, `=nothing`) number of columns in the mosaic grid
- `width`: (`::Real`, `=5`) width per subplot (unit in centimeters roughly). The total height is auto-scaled.
- `kwargs...`: other keywords passed to the 2D `plt_B0map` function

# Returns
- `Figure`: a PyObject representing the figure

# Examples
```julia-repl
julia> b0maps = rand(10, 100, 100)
julia> fig = plt_B0map(b0maps, dim=1, nRow=2, nCol=5)
julia> fig.savefig("B0map_mosaic.png", bbox_inches="tight", pad_inches=0, transparent=true)
```
"""
function plt_B0map(
    b0map::AbstractArray{<:Real, 3};
    dim                = 1        ,
    nRow               = nothing  ,
    nCol               = nothing  ,
    width              = 2        , # width of a single subplot
    kwargs...
    )
    
    # Determine the total number of frames in the image sequence
    nFrame = size(b0map, dim)
    
    # Automatically calculate the number of rows and columns if not specified
    if nRow === nothing || nCol === nothing
        nRow, nCol = get_factors(nFrame)
    end
    
    # Call mosaic to stitch the 3D sequence into a 2D matrix
    stitched_map = mosaic(b0map; dim=dim, nRow=nRow, nCol=nCol)
    
    # Calculate total dimensions (only width is needed now)
    total_width = width * nCol

    # The underlying 2D plt_B0map will auto-scale the height based on the stitched_map's aspect ratio
    # and precisely mount the colorbar adjacent to it.
    fig = plt_B0map(
        stitched_map;
        width              = total_width,    
        kwargs...
    )
    
    return fig
end


function plt_B0map(
    b0map::Vector{<:AbstractArray};
    nRow               = nothing  ,
    nCol               = nothing  ,
    width              = 2        , # width of a single subplot
    kwargs...
    )
    
    # Determine the total number of frames in the image sequence
    nFrame = size(b0map, 1)
    
    # Automatically calculate the number of rows and columns if not specified
    if nRow === nothing || nCol === nothing
        nRow, nCol = get_factors(nFrame)
    end
    
    # Call mosaic to stitch the 3D sequence into a 2D matrix
    stitched_map = mosaic(b0map; nRow=nRow, nCol=nCol)
    
    # Calculate total dimensions (only width is needed now)
    total_width = width * nCol

    # The underlying 2D plt_B0map will auto-scale the height based on the stitched_map's aspect ratio
    # and precisely mount the colorbar adjacent to it.
    fig = plt_B0map(
        stitched_map;
        width              = total_width,    
        kwargs...
    )
    
    return fig
end