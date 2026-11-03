"""
    mosaic(imgs; dim = 1, nRow = 4, nCol = 4)

Arrange slices of a three-dimensional array as a row-major 2D tile grid. `dim`
selects the slice axis (1, 2 or 3); use positive integer `nRow` and `nCol`.
Too few slices are padded with zeros; excess slices are truncated. The output
has the input element type and tile height times nRow by tile width times nCol.
This function returns an array and does not create a figure.
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
    mosaic(imgs::Vector{<:AbstractArray}; nRow = 4, nCol = 4)

Tile a nonempty vector of identically sized images. Each element must be a
2D array or a 3D array with a singleton third axis, which is dropped. The
images are stacked and passed to the array `mosaic` method with `dim=3`.
Positive `nRow`/`nCol` control padding or truncation; return a 2D array.
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