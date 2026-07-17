"""
    fig = plt_image(img::AbstractArray{<:Real, 2}; kwargs...)

# Description
Plots an image array using matplotlib.

# Arguments
- `img`: (`::AbstractArray{<:Real, 2}`) `[nX, nY]`, image array to be plotted

# Keywords
- `title`: (`::String`, `=""`) figure's suptitle
- `width`: (`::Real`, `=5`) figure's width (unit in centimeters roughly). The height is automatically scaled based on the image's aspect ratio.
- `vmaxp`: (`::Real`, `=100`) percentile of maximum value to be used for window width/ window level
- `vminp`: (`::Real`, `=0`) percentile of minimum value to be used for window width/ window level
- `vmax`: (`::Real`, `=nothing`) maximum value to be used for window width/ window level
- `vmin`: (`::Real`, `=nothing`) minimum value to be used for window width/ window level
- `cmap`: (`::String`, `="gray"`) colormap to be used for plotting
- `fontsize_title`: (`::Integer`, `=10`) font size of the title
- `color_facecolor`: (`::String`, `="#ffffff"`) background color of the figure
- `color_label`: (`::String`, `="#000000"`) color of the labels

# Returns
- `Figure`: a PyObject representing the figure

# Examples
```julia-repl
julia> fig = plt_image(rand(100, 100))
julia> fig.savefig("123.png", bbox_inches="tight", pad_inches=0, transparent=true)
```
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
    fig = plt_image(imgs::AbstractArray{<:Real, 3}; dim=1, nRow=nothing, nCol=nothing, kwargs...)

Plots a sequence of image array as a mosaic using matplotlib by overloading `plt_image`.
It stitches the 3D array into a 2D mosaic and then calls the 2D `plt_image` method.

# Arguments
- `imgs`: (`::AbstractArray{<:Real, 3}`) image array to be plotted
- `dim`: (`::Int`, `=1`) dimension along which to plot the images

# Keywords
- `nRow`: (`::Union{Integer, Nothing}`, `=nothing`) number of rows in the mosaic grid
- `nCol`: (`::Union{Integer, Nothing}`, `=nothing`) number of columns in the mosaic grid
- `width`: (`::Real`, `=2`) width per subplot (unit in centimeters roughly). The total height is auto-scaled.
- `kwargs...`: other keywords passed to the 2D `plt_image` function

# Returns
- `Figure`: a PyObject representing the figure

# Examples
```julia-repl
julia> imgs = rand(10, 100, 100)
julia> fig = plt_image(imgs, dim=1, nRow=2, nCol=5)
julia> fig.savefig("123.png", bbox_inches="tight", pad_inches=0, transparent=true)
```
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


