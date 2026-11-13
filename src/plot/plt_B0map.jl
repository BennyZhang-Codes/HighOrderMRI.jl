"""
    plt_B0map(
        b0map::AbstractArray{<:Real, 2};
        title = "",
        width = 2,
        vmax = 100,
        vmin = -100,
        cmap = "jet",
        fontsize_label = 9,
        fontsize_ticklabel = 8,
        pad_labeltick = 2,
        linewidth = 0.5,
        ticklength = 1.5,
        color_facecolor = "#1F1F1F",
        color_label = "#CCCCCC")

Display a real matrix interpreted as a static field map in Hz, with a labelled
colorbar, and return the Matplotlib figure. Values are not converted. Width is
approximately the image-panel width in centimetres; the figure additionally
reserves margins and colorbar space. Color limits are the supplied values and
font sizes are points. This helper does not save or close the figure.
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
    plt_B0map(b0map::AbstractArray{<:Real, 3}; dim = 1, nRow = nothing, nCol = nothing, width = 2, kwargs...)

Tile a 3D real field-map array using `mosaic`, then call the matrix `plt_B0map`
method. `dim` selects the slice axis. If either grid size is nothing, both are
chosen from the frame count. Width is approximately centimetres per tile column;
other keywords are forwarded. Values must already be in Hz. Return the figure.
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


"""
    plt_B0map(b0map::Vector{<:AbstractArray}; nRow = nothing, nCol = nothing, width = 2, kwargs...)

Tile equally sized real field maps from a nonempty vector; elements are 2D or
3D with singleton third axes. If either grid size is nothing, both are chosen
from the frame count. Width is approximately centimetres per tile column.
Remaining keywords are forwarded to the matrix method. Return its figure.
"""
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