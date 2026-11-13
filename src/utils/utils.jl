include("GPUFunctions.jl")

include("ImageProcess.jl")
export normalization, standardization
export get_center_range, get_center_crop

include("ImageMetrics.jl")
export complex_alignment_scale, raw_complex_nrmse, aligned_complex_nrmse
export magnitude_nrmse, magnitude_ssim
export HO_img_scale, HO_MSE, HO_RMSE, HO_NRMSE, HO_SSIM

include("ImageResize.jl")
export imresize_real, imresize_complex

include("conversion.jl")
export grad2traj, traj2grad

include("io.jl")
export struct_to_dict

"""
    get_factors(num::Int64)::Tuple{Int64, Int64}

For a positive `Int64`, return an ascending pair `(a, b)` with `a*b == num`
selected near the square root. The pair satisfies `a <= b`, including square
numbers. Use positive input; the implementation has no explicit domain guard
for zero or negative values.
"""
function get_factors(num::Int64)::Tuple{Int64,Int64}
    num_sqrt = Int64(ceil(sqrt(num)))
    factor_a = num
    factor_b = 1

    for a in range(num_sqrt,1, step=-1)
        if num%a == 0
            factor_a = Int64(a)
            factor_b = Int64(num/a)
            break
        end
    end
    factor_a, factor_b = sort([factor_a, factor_b])
    return factor_a, factor_b
end

export get_factors


"""
    get_mask(image; threshold = 0)

Return a Boolean array of the input shape whose entries satisfy
`abs(image) > threshold`. The comparison is strict; values equal to the
threshold are excluded. This is an intensity threshold, not a geometry or
noise-estimation mask. NaN magnitudes compare false.
"""
function get_mask(image; threshold=0)
    image = abs.(image);
    mask = zeros(size(image));
    mask[image.>threshold] .= 1;
    mask = isone.(mask);
    return mask
end
export get_mask
