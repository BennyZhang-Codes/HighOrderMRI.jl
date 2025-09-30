
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