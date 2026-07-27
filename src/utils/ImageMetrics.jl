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
    complex_alignment_scale(reconstruction, reference)

Return the least-squares complex scalar `α` that minimizes
`norm(α * reconstruction - reference)`.

This alignment removes a global intensity and phase difference. It is intended
for explicitly labelled secondary metrics, not raw reconstruction accuracy.
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
    raw_complex_nrmse(reconstruction, reference)

Compute the raw complex normalized root-mean-square error

```math
\\frac{\\lVert x - x_\\mathrm{ref}\\rVert_2}
     {\\lVert x_\\mathrm{ref}\\rVert_2}.
```

No intensity scaling or global phase alignment is applied. This is the primary
accuracy metric for complex MRI reconstructions.
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
    aligned_complex_nrmse(reconstruction, reference;
                          scale=complex_alignment_scale(reconstruction, reference))

Compute complex NRMSE after applying one global least-squares complex scale.
This is an explicitly aligned secondary metric.
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
    magnitude_nrmse(reconstruction, reference; align=false)

Compute NRMSE between the magnitude images:

```math
\\frac{\\lVert |x| - |x_\\mathrm{ref}|\\rVert_2}
     {\\lVert |x_\\mathrm{ref}|\\rVert_2}.
```

Set `align=true` only for a secondary analysis that first applies the global
complex scale returned by [`complex_alignment_scale`](@ref).
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
    magnitude_ssim(reconstruction, reference; align=false, normalize=true)

Compute SSIM between magnitude images. Complex inputs are converted with
`abs`. By default, both images are divided by the peak magnitude of the
reference so floating-point SSIM inputs use a common unit intensity range.
The same scale is applied to both images, so reconstruction intensity errors
are retained.

Set `align=true` only for an explicitly aligned secondary analysis.
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
HO_img_scale(reference::AbstractArray{<:Number}, reconstruction::AbstractArray{<:Number}) = complex_alignment_scale(reconstruction, reference) .* reconstruction

function HO_MSE(reference::AbstractArray{<:Number}, reconstruction::AbstractArray{<:Number}; scale::Bool=false)
    _check_metric_inputs(reconstruction, reference)
    values = scale ? HO_img_scale(reference, reconstruction) : reconstruction
    return norm(vec(reference) - vec(values))^2 / length(reference)
end

HO_RMSE(reference::AbstractArray{<:Number}, reconstruction::AbstractArray{<:Number}; scale::Bool=false) = sqrt(HO_MSE(reference, reconstruction; scale))

HO_NRMSE(reference::AbstractArray{<:Number}, reconstruction::AbstractArray{<:Number}; scale::Bool=false) =
    scale ? aligned_complex_nrmse(reconstruction, reference) : raw_complex_nrmse(reconstruction, reference)

HO_SSIM(reference::AbstractArray{<:Number}, reconstruction::AbstractArray{<:Number}; scale::Bool=false) = magnitude_ssim(reconstruction, reference; align=scale)
