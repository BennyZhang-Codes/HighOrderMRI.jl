"""
    plt_image(
        img::AbstractArray{<:Real, 2};
        title = "",
        width = 5,
        vmaxp = 100,
        vminp = 0,
        vmax = nothing,
        vmin = nothing,
        cmap = "gray",
        fontsize_title = 10,
        color_facecolor = "#ffffff",
        color_label = "#000000")

Display a real matrix and return the Matplotlib figure. Width is approximately
centimetres; height follows the matrix aspect ratio. No magnitude conversion
is performed, so pass `abs.(image)` for a complex MRI image.

If either `vmin` or `vmax` is nothing, the implementation recomputes both from
`vminp`/`vmaxp` quantiles. Supply both explicit limits to retain either one.
Font sizes are points; the function does not save or close its figure.
"""
function plt_image(
    img::AbstractArray{<:Real, 2};
    title              = ""       ,
    width              = 5        ,
    vmaxp              = 100      ,
    vminp              = 0        ,
    vmax               = nothing  ,
    vmin               = nothing  ,
    cmap               = "gray"   ,
    fontsize_title     = 10       ,
    color_facecolor    = "#ffffff",
    color_label        = "#000000",
    )
    nX, nY = size(img)
    if vmax === nothing || vmin === nothing
        vmax=quantile(reshape(img, :), vmaxp/100)
        vmin=quantile(reshape(img, :), vminp/100)
    end

    # Auto-scaling height based on the image's aspect ratio and the provided width
    # Note: size(img, 1) is nX (height), size(img, 2) is nY (width)
    # So aspect ratio height/width = nX/nY
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width/2.53999863, (nX/nY)*width/2.53999863), facecolor=color_facecolor)
    ax.set_title(title, fontsize=fontsize_title, color=color_label)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    return fig
end


"""
    plt_image(imgs::AbstractArray{<:Real, 3}; dim = 1, nRow = nothing, nCol = nothing, width = 2, kwargs...)

Tile a real 3D array using `mosaic`, then call the matrix `plt_image` method.
`dim` selects the slice axis. If either grid size is nothing, both `nRow` and
`nCol` are replaced by factors of the frame count. Width is approximately
centimetres per tile column; the final matrix plot receives `width*nCol`.
Remaining keywords are forwarded to the matrix method. Return its figure.
"""
function plt_image(
    imgs::AbstractArray{<:Real, 3};
    dim                = 1        ,
    nRow               = nothing  ,
    nCol               = nothing  ,
    width              = 2        , # width of a single subplot
    kwargs...
    )
    
    # Determine the total number of frames in the image sequence
    nFrame = size(imgs, dim)
    
    # Automatically calculate the number of rows and columns if not specified
    if nRow === nothing || nCol === nothing
        nRow, nCol = get_factors(nFrame)
    end
    
    # Call mosaic to stitch the 3D data to a 2D matrix
    stitched_img = mosaic(imgs; dim=dim, nRow=nRow, nCol=nCol)
    
    # Calculate total dimensions (only width is needed now)
    total_width = width * nCol

    # The underlying 2D plt_image will auto-scale the height based on the stitched_img's aspect ratio
    fig = plt_image(
        stitched_img;
        width              = total_width,    
        kwargs...
    )
    
    return fig
end

"""
    plt_image(imgs::Vector{<:AbstractArray}; nRow = nothing, nCol = nothing, width = 2, kwargs...)

Tile a nonempty vector of equally sized real images and call matrix `plt_image`.
Elements must be 2D or 3D with a singleton third axis. If either grid size is
nothing, both are chosen from the frame count. Width is approximately centimetres
per tile column. Remaining keywords reach the matrix method. Return its figure.
"""
function plt_image(
    imgs::Vector{<:AbstractArray}; 
    nRow               = nothing  ,
    nCol               = nothing  ,
    width              = 2        , # width of a single subplot
    kwargs...
    )
    
    # Determine the total number of frames in the image sequence
    nFrame = size(imgs, 1)
    
    # Automatically calculate the number of rows and columns if not specified
    if nRow === nothing || nCol === nothing
        nRow, nCol = get_factors(nFrame)
    end
    
    # Call mosaic to stitch the 3D data to a 2D matrix
    stitched_img = mosaic(imgs; nRow=nRow, nCol=nCol)
    
    # Calculate total dimensions (only width is needed now)
    total_width = width * nCol

    # The underlying 2D plt_image will auto-scale the height based on the stitched_img's aspect ratio
    fig = plt_image(
        stitched_img;
        width              = total_width,    
        kwargs...
    )
    
    return fig
end


