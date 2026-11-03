# Normalization
"""
    normalization(img::AbstractArray{T, 2}) where T <: Real

Apply min-max normalization `(img - minimum(img)) / (maximum(img) - minimum(img))`
to a real matrix. This rescales image intensity; it does not preserve absolute
quantitative calibration. A constant image has a zero denominator and produces
nonfinite values in the current implementation. Empty/nonfinite inputs are not
handled by a separate validation policy.
"""
function normalization(img::AbstractArray{T,2}) where T<:Real
    return (img.- minimum(img))./ (maximum(img) .- minimum(img))
end

# Standardization
"""
    standardization(img::AbstractArray{T, 2}) where T <: Real

Subtract the mean and divide by the population standard deviation
`std(img; corrected=false)` of a real matrix. A constant image has zero standard
deviation and produces nonfinite values in the current implementation.
This helper does not preserve absolute intensity or perform robust outlier handling.
"""
function standardization(img::AbstractArray{T,2}) where T<:Real
    return (img.- mean(img))./ std(img; corrected=false)
end

"""
    mymse(img::AbstractArray{T, 2}, ref::AbstractArray{T, 2}) where T <: Real

Internal, unexported mean squared difference for two real matrices. The
implementation uses broadcasting and does not provide the explicit input checks
of the public reconstruction metrics.
"""
function mymse(img::AbstractArray{T,2}, ref::AbstractArray{T,2}) where T<:Real
    return mean((img.- ref).^2)
end

"""
    get_center_range(x::Int64, x_range::Int64)

Return `floor(x/2) - floor(x_range/2) + 1 : floor(x/2) + ceil(x_range/2)`
as an `Int64` unit range. Here `x` is the full axis length and `x_range` is the
requested length, not a centre coordinate. Use positive lengths; containment
and bounds are not validated. Even/odd cases follow this exact index convention.
"""
function get_center_range(x::Int64, x_range::Int64)
    center = Int64(floor(x/2))
    return center - Int64(floor(x_range/2))+1 : center + Int64(ceil(x_range/2))
end


"""
    get_center_crop(image::AbstractArray{T}, out_x::Int64, out_y::Int64) where T <: Number

Centre-crop or zero-pad the first two axes of a 2D or 3D numeric array.
The requested sizes must be positive `Int64` values. The third axis, when
present, is retained, and the output has the input element type.

# Current limitation

If either requested spatial size equals the corresponding input size, none of
the four copy branches executes: the result is all zeros. This includes an
unchanged requested size. Do not use this helper for that case without an
explicit caller-side alternative. The limitation is documented here, not fixed
by this documentation change.
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
