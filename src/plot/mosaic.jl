"""
    mosaic(imgs; dim=1, nRow=4, nCol=4)

Create a 2D mosaic (tiled grid) from a 3D image volume or image sequence.

# Description
This function extracts 2D slices from the specified dimension of a 3D array 
and arranges them into a 2D grid. 
- If the grid size (`nRow * nCol`) is larger than the number of slices, the remaining spaces are padded with zeros (black boxes). 
- If the grid size is smaller, the image sequence is truncated to fit the grid.

# Arguments
- `imgs::AbstractArray{T, 3}`: The input 3D image volume.
- `dim::Int`: The dimension along which the image sequence/slices are stacked. 
  Must be 1, 2, or 3. Default is `1`.
- `nRow::Int`: The number of rows in the output mosaic grid. Default is `4`.
- `nCol::Int`: The number of columns in the output mosaic grid. Default is `4`.

# Returns
- `show_res::Matrix{T}`: A 2D matrix containing the tiled images.

# Example
```julia
# Create a dummy 3D volume (e.g., 16 slices of 128x128 images stacked along Z)
volume = rand(128, 128, 16)

# Create a 4x4 mosaic of the slices
grid_img = mosaic(volume; dim=3, nRow=4, nCol=4)
"""
function mosaic(imgs; dim=1, nRow=4, nCol=4)
    @assert dim in [1, 2, 3] "dim of image sequence should be 1, 2, or 3"
    if dim == 2
        imgs = permutedims(imgs, [2, 1, 3])
    elseif dim == 3
        imgs = permutedims(imgs, [3, 1, 2])
    end
    nImages = size(imgs, 1)

    # zero padding if needed
    if nRow * nCol > nImages
        pad_imgs = zeros(eltype(imgs), nRow * nCol - nImages, size(imgs,2), size(imgs,3))
        imgs = cat(imgs, pad_imgs; dims=1)
    elseif nRow * nCol < nImages
        imgs = imgs[1:(nRow*nCol), :, :]
    end

    # create mosaic
    h, w = size(imgs,2), size(imgs,3)
    show_res = zeros(eltype(imgs), h*nRow, w*nCol)

    for r in 1:nRow
        row_imgs = imgs[nCol*(r-1) + 1, :, :]
        for c in 2:nCol
            row_imgs = hcat(row_imgs, imgs[nCol*(r-1) + c, :, :])
        end
        if r == 1
            show_res = row_imgs
        else
            show_res = vcat(show_res, row_imgs)
        end
    end
    return show_res
end


"""
    mosaic(imgs::Vector{<:AbstractArray}; nRow=4, nCol=4)

Overloads the `mosaic` function to support a vector of images.
Automatically handles strict 2D matrices as well as 3D arrays with a singleton third dimension (e.g., NxMx1).
"""
function mosaic(imgs::Vector{<:AbstractArray}; nRow=4, nCol=4)
    # 1. Safety check: Ensure the image vector is not empty
    isempty(imgs) && error("Image vector is empty!")
    
    # 2. Squeeze singleton dimensions (convert NxMx1 to NxM)
    # This perfectly handles cases where slices are technically 3D but effectively 2D
    processed_imgs = map(imgs) do img
        if ndims(img) == 3 && size(img, 3) == 1
            return dropdims(img, dims=3)
        elseif ndims(img) == 2
            return img
        else
            error("Elements must be strictly 2D arrays or 3D arrays with a singleton third dimension (NxMx1).")
        end
    end

    # 3. Safety check: Ensure all processed 2D slices have identical dimensions
    sz = size(processed_imgs[1])
    all(x -> size(x) == sz, processed_imgs) || error("All images must have the same spatial dimensions.")

    # 4. Stack the processed Vector into a clean 3D array
    imgs_3d = cat(processed_imgs..., dims=3)

    # 5. Reuse the original core mosaic logic
    return mosaic(imgs_3d; dim=3, nRow=nRow, nCol=nCol)
end