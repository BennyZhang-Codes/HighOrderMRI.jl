import ImageQualityIndexes: assess_ssim
import LinearAlgebra: norm, dot

function HO_img_scale(x_ref::AbstractArray{<:Real}, x::AbstractArray{<:Real})
    # This is a little trick. We usually are not interested in simple scalings
    # and therefore "calibrate" them away
    alpha = norm(x)>0 ? (dot(vec(x_ref),vec(x))+dot(vec(x),vec(x_ref))) /
            (2*dot(vec(x),vec(x))) : 1.0
    x2 = x.*alpha
    return x2
end

# HO_RMSE(x_ref, x)
# HO_MSE(x_ref, x)
# HO_NRMSE(x_ref, x)
# HO_SSIM(x_ref, x)

function HO_NRMSE(x_ref::AbstractArray{<:Real}, x::AbstractArray{<:Real}; scale::Bool=true)
    x     = vec(x)
    x_ref = vec(x_ref)
    N = length(x)
    x = scale ? HO_img_scale(x_ref, x) : x
    
    RMS =  1.0/sqrt(N)*norm(vec(x_ref)-vec(x))
    NRMS = RMS/(maximum(abs.(x_ref))-minimum(abs.(x_ref)))
    return NRMS
end

function HO_MSE(x_ref::AbstractArray{<:Real}, x::AbstractArray{<:Real}; scale::Bool=true)
    N = length(vec(x))
    x = scale ? HO_img_scale(x_ref, x) : x
    mse = 1 / N * norm(x_ref-x)^2
    return mse
end

function HO_RMSE(x_ref::AbstractArray{<:Real}, x::AbstractArray{<:Real}; scale::Bool=true)
    mse = HO_MSE(x_ref, x; scale=scale)
    rmse = sqrt(mse)
    return rmse
end

function HO_SSIM(x_ref::AbstractArray{<:Real}, x::AbstractArray{<:Real}; scale::Bool=true)
    x = scale ? HO_img_scale(x_ref, x) : x
    ssim = assess_ssim(x_ref, x)
    return ssim
end



