# Normalization
function normalization(img::AbstractArray{T,2}) where T<:Real
    return (img.- minimum(img))./ (maximum(img) .- minimum(img))
end

# Standardization
function standardization(img::AbstractArray{T,2}) where T<:Real
    return (img.- mean(img))./ std(img; corrected=false)
end

"""
MSE
"""
function mymse(img::AbstractArray{T,2}, ref::AbstractArray{T,2}) where T<:Real
    return mean((img.- ref).^2)
end

"""
    out = get_center_range(x::Int64, x_range::Int64)

# Description
    This function returns the range of the center of the input range.

# Arguments
- `x`: (Int64) center of the range
- `x_range`: (Int64) range of the center

# Returns
- `out`: range of the center

```julia-repl
julia> out = get_center_range(200, 100)
```
"""
function get_center_range(x::Int64, x_range::Int64)
    center = Int64(floor(x/2))
    return center - Int64(floor(x_range/2))+1 : center + Int64(ceil(x_range/2))
end


"""
    out = get_center_crop(image::AbstractArray{T, 3}, out_x::Int64, out_y::Int64) where T <: Real

# Description
    This function crops the center of the input image to the specified size. If the input image is smaller than the output size, the output image will be padded with zeros.

# Arguments
- `image`: (AbstractArray{<:Number, 2}) input image array
- `out_x`: (Int64) output image width
- `out_y`: (Int64) output image height

# Returns
- `out`: croped image array

# Examples
```julia-repl
julia> out = get_center_crop(img, 100, 100)
```
"""
function  get_center_crop(image::AbstractArray{T}, out_x::Int64, out_y::Int64) where T <: Number
    nDim = ndims(image)
    @assert nDim == 2 || nDim == 3 "Input array must be 2D or 3D"
    @assert out_x > 0 && out_y > 0 "Output size must be positive"
    
    image = nDim == 2 ? reshape(image, size(image)..., 1) : image
    Nx = out_x
    Ny = out_y
    
    M, N, K = size(image)
    out = zeros(T, Nx, Ny, K)

    if Nx < M && Ny < N
        rangex = get_center_range(M, Nx)
        rangey = get_center_range(N, Ny)
        out = image[rangex, rangey, :]
    elseif Nx > M && Ny > N
        rangex = get_center_range(Nx, M)
        rangey = get_center_range(Ny, N)
        out[rangex, rangey, :] = image
    elseif Nx < M && Ny > N
        rangex = get_center_range(M, Nx)
        rangey = get_center_range(Ny, N)
        out[:, rangey, :] = image[rangex, :, :]
    elseif Nx > M && Ny < N
        rangex = get_center_range(Nx, M)
        rangey = get_center_range(N, Ny)
        out[rangex, :, :] = image[:, rangey, :]
    end
    out = nDim == 2 ?  out[:,:,1] : out
    return out
end
