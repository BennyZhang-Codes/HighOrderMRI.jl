include("GPUFunctions.jl")

include("EncodingApproximations.jl")

include("ImageProcess.jl")
export normalization, standardization
export get_center_range, get_center_crop

# include("ImageMetrics.jl")
# export HO_img_scale, HO_MSE, HO_RMSE, HO_NRMSE, HO_SSIM

include("ImageResize.jl")
export imresize_real, imresize_complex

include("conversion.jl")
export grad2traj, traj2grad

"""
    factor_a, factor_b = get_factors(num::Int64)

# Description
    Get the factors of a number, Where a * b = num, make the difference between a and b as small as possible.

# Arguments
- `num::Int64`: The number whose factors are to be found.

# Returns
- `Tuple{Int64,Int64}`: (a, b), a < b. A tuple containing the two factors of the number.
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