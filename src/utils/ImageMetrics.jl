import ImageQualityIndexes: assess_ssim
import LinearAlgebra: dot, norm

function _check_metric_inputs(reconstruction::AbstractArray, reference::AbstractArray)
    size(reconstruction) == size(reference) || throw(DimensionMismatch( "reconstruction and reference must have identical sizes; got " * "$(size(reconstruction)) and $(size(reference))"))
    isempty(reference) && throw(ArgumentError("metric inputs must not be empty"))
    return nothing
end

function _reference_norm(reference::AbstractArray)
    denominator = norm(vec(reference))
    isfinite(denominator) || throw(DomainError(denominator, "reference must contain only finite values"))
    iszero(denominator) && throw(DomainError(denominator, "reference must have nonzero norm"))
    return denominator
end

"""
    complex_alignment_scale(reconstruction::AbstractArray{<:Number}, reference::AbstractArray{<:Number})

Return the least-squares scalar `alpha = dot(vec(x), vec(reference)) / dot(vec(x), vec(x))`
for `x = reconstruction`, using the conjugating inner product. Multiplying by
this scalar removes one global amplitude and phase difference.

Inputs must have the same nonempty shape and finite scalar products. A zero
reconstruction returns a scale of one because the scale is then immaterial.
This is an explicitly aligned comparison, not raw reconstruction accuracy.
"""
function complex_alignment_scale(
    reconstruction::AbstractArray{<:Number},
    reference::AbstractArray{<:Number},
)
    _check_metric_inputs(reconstruction, reference)

    values = float.(vec(reconstruction))
    target = float.(vec(reference))
    energy = real(dot(values, values))
    numerator = dot(values, target)

    isfinite(energy) && isfinite(numerator) || throw(DomainError((energy, numerator), "metric inputs must contain only finite values",))

    # The scale is immaterial when the reconstruction is identically zero.
    iszero(energy) && return one(numerator + energy)
    return numerator / energy
end

"""
    raw_complex_nrmse(reconstruction::AbstractArray{<:Number}, reference::AbstractArray{<:Number})

Return `norm(vec(reconstruction) - vec(reference)) / norm(vec(reference))`
without intensity scaling or global phase alignment. Inputs must have the same
nonempty shape. A zero/nonfinite reference norm or nonfinite error norm raises
an error. The output is a dimensionless scalar; no mask is applied internally.
"""
function raw_complex_nrmse(
    reconstruction::AbstractArray{<:Number},
    reference::AbstractArray{<:Number},
)
    _check_metric_inputs(reconstruction, reference)
    denominator = _reference_norm(reference)
    numerator = norm(vec(reconstruction) - vec(reference))
    isfinite(numerator) || throw(DomainError(numerator, "reconstruction must contain only finite values"))
    return numerator / denominator
end

"""
    aligned_complex_nrmse(
        reconstruction::AbstractArray{<:Number},
        reference::AbstractArray{<:Number};
        scale = complex_alignment_scale(reconstruction, reference))

Return complex NRMSE after multiplying the reconstruction by `scale`. By
default the scale is fitted by `complex_alignment_scale`. Reference shape,
nonempty size and nonzero finite norm are checked as for raw complex NRMSE.
Label this as an aligned result because amplitude and phase errors can be removed.
"""
function aligned_complex_nrmse(
    reconstruction::AbstractArray{<:Number},
    reference::AbstractArray{<:Number};
    scale=complex_alignment_scale(reconstruction, reference),
)
    _check_metric_inputs(reconstruction, reference)
    denominator = _reference_norm(reference)
    numerator = norm(scale .* vec(reconstruction) - vec(reference))
    isfinite(numerator) || throw(DomainError(numerator, "aligned reconstruction must contain only finite values"))
    return numerator / denominator
end

"""
    magnitude_nrmse(
        reconstruction::AbstractArray{<:Number},
        reference::AbstractArray{<:Number};
        align::Bool = false)

Return `norm(abs.(reconstruction) - abs.(reference)) / norm(reference)` after
vectorization. Phase differences are not measured. With `align=true`, first fit
one global complex scale, then compare magnitudes. Inputs require the same
nonempty shape and a nonzero finite reference norm; nonfinite errors are rejected.
"""
function magnitude_nrmse(
    reconstruction::AbstractArray{<:Number},
    reference::AbstractArray{<:Number};
    align::Bool=false,
)
    _check_metric_inputs(reconstruction, reference)
    denominator = _reference_norm(reference)
    values = vec(reconstruction)
    if align
        values = complex_alignment_scale(reconstruction, reference) .* values
    end
    numerator = norm(abs.(values) - abs.(vec(reference)))
    isfinite(numerator) || throw(DomainError(numerator, "reconstruction must contain only finite values"))
    return numerator / denominator
end

"""
    magnitude_ssim(
        reconstruction::AbstractArray{<:Number},
        reference::AbstractArray{<:Number};
        align::Bool = false,
        normalize::Bool = true)

Compute SSIM on magnitude images using `ImageQualityIndexes.assess_ssim`.
Both images are converted to floating-point host arrays with `Array` and `abs`.
If `align=true`, first fit a global complex scale. If `normalize=true`, divide
both magnitudes by the same peak magnitude of the reference; the reconstruction
is not normalized independently or clipped to a unit range.

Inputs must have the same nonempty shape, finite values, and nonzero reference
norm. Supported image dimensions and the SSIM window are delegated to the
dependency. Report alignment/normalization settings with the value.
"""
function magnitude_ssim(
    reconstruction::AbstractArray{<:Number},
    reference::AbstractArray{<:Number};
    align::Bool=false,
    normalize::Bool=true,
)
    _check_metric_inputs(reconstruction, reference)
    _reference_norm(reference)

    aligned_reconstruction = reconstruction
    if align
        aligned_reconstruction = complex_alignment_scale(reconstruction, reference) .* reconstruction
    end
    values = float.(abs.(Array(aligned_reconstruction)))
    target = float.(abs.(Array(reference)))

    all(isfinite, values) && all(isfinite, target) || throw(DomainError(nothing, "metric inputs must contain only finite values"))

    if normalize
        peak = maximum(target)
        iszero(peak) && throw(DomainError(peak, "reference magnitude must have a nonzero peak"))
        values ./= peak
        target ./= peak
    end

    return assess_ssim(values, target)
end

# Reference-first compatibility API retained for existing scripts.
"""
    HO_img_scale(reference::AbstractArray{<:Number}, reconstruction::AbstractArray{<:Number})

Compatibility API with reference-first argument order. Return the reconstruction
multiplied by `complex_alignment_scale(reconstruction, reference)`; this returns
an aligned image, not the fitted scalar.
"""
HO_img_scale(reference::AbstractArray{<:Number}, reconstruction::AbstractArray{<:Number}) = complex_alignment_scale(reconstruction, reference) .* reconstruction

"""
    HO_MSE(reference::AbstractArray{<:Number}, reconstruction::AbstractArray{<:Number}; scale::Bool = false)

Compatibility API with reference-first argument order. Return the squared
complex Euclidean error divided by the number of reference elements. With
`scale=true`, first apply `HO_img_scale`. Same nonempty shapes are checked;
without scaling this implementation does not separately reject nonfinite values
or require a nonzero reference norm.
"""
function HO_MSE(reference::AbstractArray{<:Number}, reconstruction::AbstractArray{<:Number}; scale::Bool=false)
    _check_metric_inputs(reconstruction, reference)
    values = scale ? HO_img_scale(reference, reconstruction) : reconstruction
    return norm(vec(reference) - vec(values))^2 / length(reference)
end

"""
    HO_RMSE(reference::AbstractArray{<:Number}, reconstruction::AbstractArray{<:Number}; scale::Bool = false)

Compatibility API with reference-first argument order. Return the square root
of `HO_MSE(reference, reconstruction; scale)`, retaining that function's
alignment choice and input behavior.
"""
HO_RMSE(reference::AbstractArray{<:Number}, reconstruction::AbstractArray{<:Number}; scale::Bool=false) = sqrt(HO_MSE(reference, reconstruction; scale))

"""
    HO_NRMSE(reference::AbstractArray{<:Number}, reconstruction::AbstractArray{<:Number}; scale::Bool = false)

Compatibility API with reference-first argument order. Dispatch to
`raw_complex_nrmse(reconstruction, reference)` when `scale=false`, or to
`aligned_complex_nrmse` when true. A zero reference norm is rejected.
"""
HO_NRMSE(reference::AbstractArray{<:Number}, reconstruction::AbstractArray{<:Number}; scale::Bool=false) =
    scale ? aligned_complex_nrmse(reconstruction, reference) : raw_complex_nrmse(reconstruction, reference)

"""
    HO_SSIM(reference::AbstractArray{<:Number}, reconstruction::AbstractArray{<:Number}; scale::Bool = false)

Compatibility API with reference-first argument order. Call
`magnitude_ssim(reconstruction, reference; align=scale)` with its default
reference-peak normalization. It compares magnitudes and returns a scalar SSIM.
"""
HO_SSIM(reference::AbstractArray{<:Number}, reconstruction::AbstractArray{<:Number}; scale::Bool=false) = magnitude_ssim(reconstruction, reference; align=scale)
