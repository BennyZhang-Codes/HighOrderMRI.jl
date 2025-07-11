
"""
    fig = plt_image(img::Array{<:Real, 2}; kwargs...)

# Description
    Plots an image array using matplotlib 

# Arguments
- `img`: (`::Array{<:Real, 2}`) `[nX, nY]`, image array to be plotted

# Keywords
- `title`: (`::Integer`, `=""`) figure's suptitle
- `width`: (`::Real`, `=5`) figure's width
- `height`: (`::Real`, `=5`) figure's height
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
julia> fig.savefig("123.png",bbox_inches="tight", pad_inches=0, transparent=true)
```
"""
function plt_image(
    img::AbstractArray{<:Real, 2};
    title              = ""       ,
    width              = 5        ,
    height             = 5        ,
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

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width/2.53999863, (nY/nX)*height/2.53999863), facecolor=color_facecolor)
    ax.set_title(title, fontsize=fontsize_title, color=color_label)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    return fig
end

"""
    fig = plt_images(imgs::Array{<:Real, 3}; title="", width=5, height=5, vmaxp=100, vminp=0, cmap="gray", fontsize_title=10, color_facecolor="#ffffff")
    Plots a sequence of image array using matplotlib 

# Arguments
- `imgs`: (`::Array{<:Real, 3}`) `[nFrame, nX, nY]`, image array to be plotted
- `dim`: (`::Int`) dimension along which to plot the images

# Keywords
- `title`: (`::String`, `=""`) figure's suptitle
- `width`: (`::Real`, `=2`) figure's width (unit in minimeters)
- `height`: (`::Real`, `=2`) figure's height (unit in minimeters)
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
julia> imgs = rand(10, 100, 100)
julia> fig = plt_images(imgs)
julia> fig.savefig("123.png",bbox_inches="tight", pad_inches=0, transparent=true)
```
"""
function plt_images(
    imgs::AbstractArray{<:Real, 3};
    dim                = 1        ,
    nRow               = nothing  ,
    nCol               = nothing  ,
    title              = ""       ,
    width              = 2        ,
    height             = 2        ,
    vmaxp              = 100      ,
    vminp              = 0        ,
    vmax               = nothing  ,
    vmin               = nothing  ,
    cmap               = "gray"   ,
    fontsize_title     = 10       ,
    color_facecolor    = "#ffffff",
    color_label        = "#000000",
    )
    @assert dim in [1, 2, 3] "dim of image sequence should be 1, 2, or 3"
    if dim == 2
        imgs = permutedims(imgs, [2, 1, 3])
    elseif dim == 3
        imgs = permutedims(imgs, [3, 1, 2])
    end

    nFrame, nX, nY = size(imgs)
    if nRow === nothing || nCol === nothing
        nRow, nCol = get_factors(nFrame)
    end
    if vmax === nothing || vmin === nothing
        vmin, vmax = quantile(reshape(imgs, :), vminp/100), quantile(reshape(imgs, :), vmaxp/100)
    end
    
    fig, axs = plt.subplots(nrows=nRow, ncols=nCol, figsize=(width*nCol/2.53999863, (nX/nY)*height*nRow/2.53999863), facecolor=color_facecolor)
	axs = reshape(axs, nRow, nCol) # to keep axs as a 2D array
    
    fig.suptitle(title, fontsize=fontsize_title, color=color_label)
    for frame = 1:nFrame
        row, col = Int(ceil(frame/nCol)), (frame-1)%nCol + 1
        ax = axs[row, col]
        ax.imshow(imgs[frame, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
    end
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    return fig
end



