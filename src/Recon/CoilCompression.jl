export CoilCompressionTransform
export estimate_noise_covariance
export noise_prewhitening_scale_factor
export fit_coil_compression, apply_coil_compression, compress_coils

"""
    CoilCompressionTransform

Linear transform fitted by [`fit_coil_compression`](@ref). The
`compression_matrix` has size `(input_coils, virtual_coils)` and acts on the
coil dimension from the right. `singular_values` contains the complete
singular-value spectrum of the fitted calibration matrix after optional noise
prewhitening, and `retained_energy` is the fraction of its squared
singular-value energy kept by the selected virtual coils.

When `noise_covariance` is not `nothing`, the transform includes the whitening
and compression operations in one matrix. If `s` is
`prewhitening_scale_factor`, it satisfies
`compression_matrix' * (noise_covariance / s) * compression_matrix ≈ I`.
The default `s=1` recovers the usual uncorrected covariance convention.

Apply the same transform to acquired data and coil-sensitivity maps with
[`apply_coil_compression`](@ref).
"""
struct CoilCompressionTransform{T<:AbstractFloat}
    compression_matrix        :: Matrix{Complex{T}}
    singular_values           :: Vector{T}
    retained_energy           :: T
    noise_covariance          :: Union{Nothing,Matrix{Complex{T}}}
    prewhitening_scale_factor :: T
end

# Preserve the original four-argument constructor for callers that manually
# truncate or copy a transform. When a covariance is present, infer the scalar
# from C' * Ψ * C = sI so bandwidth-corrected transforms retain their metadata.
function CoilCompressionTransform(
    compression_matrix :: Matrix{Complex{T}},
    singular_values    :: Vector{T},
    retained_energy    :: T,
    noise_covariance   :: Union{Nothing,Matrix{Complex{T}}},
) where {T<:AbstractFloat}
    scale_factor = if isnothing(noise_covariance)
        one(T)
    else
        compressed_covariance = compression_matrix' * noise_covariance * compression_matrix
        T(real(tr(compressed_covariance)) / size(compression_matrix, 2))
    end
    return CoilCompressionTransform(compression_matrix, singular_values, retained_energy, noise_covariance, scale_factor)
end

Base.size(transform::CoilCompressionTransform) = size(transform.compression_matrix)
Base.size(transform::CoilCompressionTransform, dimension::Integer) = size(transform.compression_matrix, dimension)

Base.show(io::IO, transform::CoilCompressionTransform{T}) where {T} = begin
    method = isnothing(transform.noise_covariance) ? "SVD" : "noise-whitened SVD"
    scale  = transform.prewhitening_scale_factor
    scale_description = isone(scale) ? "" : ", prewhitening_scale_factor=$scale"
    println(io, ">>> CoilCompressionTransform{$T} <<<")
    println(io, "$(size(transform, 1)) => $(size(transform, 2))")
    println(io, "method          : $method")
    println(io, "retained_energy : $(transform.retained_energy)$scale_description")
end
"""
    noise_prewhitening_scale_factor(acquisition_dwell_time, noise_dwell_time; receiver_bandwidth_ratio=1,)

Compute the dimensionless correction used when a noise prescan and the MRI
acquisition have different sampling intervals or effective receiver
bandwidths:

```math
s = \\frac{T_{\\mathrm{acq}}}{T_{\\mathrm{noise}}} R_{\\mathrm{BW}}.
```

Here `receiver_bandwidth_ratio` is the scanner- or receiver-specific effective
noise-bandwidth correction. Both dwell times must use the same units. Pass the
result as `prewhitening_scale_factor` to [`fit_coil_compression`](@ref) or
[`compress_coils`](@ref). A value of one means no correction.

With a covariance `Ψ_noise` estimated from the noise prescan, the corrected
whitening treats `Ψ_acq = Ψ_noise / s` as the covariance of the acquired
MRI data. This follows the scale-factor convention used by ISMRMRD and mrpro;
it deliberately does not add an extra `sqrt(2)` factor because this package
defines covariance directly as `E[n' * n]` for complex noise.
"""
function noise_prewhitening_scale_factor(
    acquisition_dwell_time   :: Real    ,
    noise_dwell_time         :: Real    ;
    receiver_bandwidth_ratio :: Real = 1,
)
    acquisition_dwell, noise_dwell, bandwidth_ratio = promote(float(acquisition_dwell_time), float(noise_dwell_time), float(receiver_bandwidth_ratio))
    isfinite(acquisition_dwell) && acquisition_dwell > zero(acquisition_dwell) || throw(ArgumentError("acquisition_dwell_time must be finite and positive"))
    isfinite(noise_dwell) && noise_dwell > zero(noise_dwell) || throw(ArgumentError("noise_dwell_time must be finite and positive"))
    isfinite(bandwidth_ratio) && bandwidth_ratio > zero(bandwidth_ratio) || throw(ArgumentError("receiver_bandwidth_ratio must be finite and positive"))

    scale_factor = (acquisition_dwell / noise_dwell) * bandwidth_ratio
    isfinite(scale_factor) && scale_factor > zero(scale_factor) ||  throw(ArgumentError("the resulting prewhitening scale factor must be finite and positive"))
    return scale_factor
end

"""
    estimate_noise_covariance(noise_data; coil_dim=ndims(noise_data), center=true) -> Matrix

Estimate the complex receive-noise covariance from noise-only samples. All
dimensions other than `coil_dim` are flattened into independent observations.
With `center=true`, the per-coil sample mean is removed and the unbiased
`1 / (nObservation - 1)` normalization is used. With `center=false`, the
input is assumed to be zero mean and `1 / nObservation` is used.

Acquire `noise_data` with the same receiver gains and channel ordering as the
MRI data. Do not apply density compensation or coil compression before
covariance estimation. When the noise scan and MRI data have different dwell
times or effective receiver bandwidths, compute a correction with
[`noise_prewhitening_scale_factor`](@ref) and pass it alongside the covariance
to [`fit_coil_compression`](@ref) or [`compress_coils`](@ref).
"""
function estimate_noise_covariance(
    noise_data :: AbstractArray{Complex{T}}   ;
    coil_dim   :: Integer = ndims(noise_data) ,
    center     :: Bool    = true              ,
) where {T<:AbstractFloat}
    noise_matrix, _, _ = _coil_last_matrix(noise_data, coil_dim)
    all(isfinite, noise_matrix) || throw(ArgumentError("noise_data must contain only finite values"))
    n_observation = size(noise_matrix, 1)
    minimum_observations = center ? 2 : 1
    n_observation >= minimum_observations || 
        throw(ArgumentError("noise_data must contain at least $minimum_observations observations"))

    centered_noise = center ? noise_matrix .- mean(noise_matrix; dims = 1) : noise_matrix
    normalization = center ? n_observation - 1 : n_observation
    covariance = centered_noise' * centered_noise / T(normalization)
    return Matrix{Complex{T}}((covariance + covariance') / T(2))
end

function _noise_whitening_matrix(
    noise_covariance::Union{Nothing,AbstractMatrix},
    n_coil::Int,
    ::Type{T},
    prewhitening_scale_factor::Real,
) where {T<:AbstractFloat}
    identity_matrix = Matrix{Complex{T}}(I, n_coil, n_coil)
    scale_factor = T(prewhitening_scale_factor)
    isfinite(scale_factor) && scale_factor > zero(T) || throw(ArgumentError("prewhitening_scale_factor must be finite and positive"))
    if isnothing(noise_covariance)
        isone(scale_factor) || throw(ArgumentError("prewhitening_scale_factor requires noise_covariance"))
        return identity_matrix, nothing, scale_factor
    end

    size(noise_covariance) == (n_coil, n_coil) || throw(DimensionMismatch("noise_covariance must have size ($n_coil, $n_coil)"))
    covariance = Matrix{Complex{T}}(noise_covariance)
    all(isfinite, covariance) || throw(ArgumentError("noise_covariance must contain only finite values"))
    covariance_scale = max(norm(covariance), eps(T))
    isapprox(
        covariance,
        covariance';
        rtol = sqrt(eps(T)),
        atol = eps(T) * covariance_scale,
    ) || throw(ArgumentError("noise_covariance must be Hermitian"))
    covariance = Matrix{Complex{T}}((covariance + covariance') / T(2))

    factorization = try
        cholesky(Hermitian(covariance); check = true)
    catch error
        error isa PosDefException || rethrow()
        throw(ArgumentError("noise_covariance must be positive definite; use more noise samples or regularize it explicitly"))
    end
    whitening_matrix = sqrt(scale_factor) .* (factorization.U \ identity_matrix)
    return Matrix{Complex{T}}(whitening_matrix), covariance, scale_factor
end

function _coil_dimension(array::AbstractArray, coil_dim::Integer)
    1 <= coil_dim <= ndims(array) || throw(ArgumentError("coil_dim=$coil_dim must be between 1 and ndims(array)=$(ndims(array))"))
    return Int(coil_dim)
end

function _coil_last_matrix(array::AbstractArray{<:Complex}, coil_dim::Integer)
    dimension = _coil_dimension(array, coil_dim)
    permutation = [filter(!=(dimension), collect(1:ndims(array))); dimension]
    coil_last = permutedims(Array(array), permutation)
    return reshape(coil_last, :, size(coil_last, ndims(coil_last))), size(coil_last), permutation
end

"""
    fit_coil_compression(calibration_data; coil_dim=ndims(calibration_data), n_virtual_coils=nothing,
        energy_threshold=nothing, noise_covariance=nothing, prewhitening_scale_factor=1) -> CoilCompressionTransform

Fit an SVD-based coil-compression transform. All dimensions other than
`coil_dim` are treated as calibration observations. Pass either an explicit
`n_virtual_coils` or an `energy_threshold` in `(0, 1]`; specifying both or
neither is an error. The energy criterion selects the smallest rank whose
cumulative squared singular values reach the requested fraction.

When `noise_covariance` is supplied, the calibration data are first whitened
with a Cholesky factor and the returned matrix combines whitening with the
leading right singular vectors. This noise-whitened PCA is the globally
optimal rank-constrained approximation of the whitened calibration matrix in
the Frobenius norm. Without a covariance, identity noise covariance is
assumed and the method reduces to conventional software coil compression.

Set `prewhitening_scale_factor` when the noise scan and acquisition use
different dwell times or effective receiver bandwidths. For
`s=prewhitening_scale_factor`, the fitted right whitening matrix satisfies
`W' * (noise_covariance / s) * W ≈ I`. Compute `s` with
[`noise_prewhitening_scale_factor`](@ref). A non-unit value requires
`noise_covariance`.

The implementation follows software channel compression by Huang et al.
([doi:10.1016/j.mri.2007.04.010](https://doi.org/10.1016/j.mri.2007.04.010)),
with receive-noise conditioning following Kellman and McVeigh
([doi:10.1002/mrm.20713](https://doi.org/10.1002/mrm.20713)). The truncated
SVD optimality statement is limited to one global linear coil transform; it
does not claim equivalence to spatially varying geometric-decomposition coil
compression.

The returned transform is independent of array layout. It can therefore be
fitted on a calibration view and then applied to complete acquired data and
coil-sensitivity maps whose coil dimensions occur at different positions.
Inputs are copied to the CPU for the decomposition.
"""
function fit_coil_compression(
    calibration_data          :: AbstractArray{Complex{T}};
    coil_dim                  :: Integer = ndims(calibration_data),
    n_virtual_coils           :: Union{Nothing,Integer}        = nothing,
    energy_threshold          :: Union{Nothing,Real}           = nothing,
    noise_covariance          :: Union{Nothing,AbstractMatrix} = nothing,
    prewhitening_scale_factor :: Real = 1,
) where {T<:AbstractFloat}
    xor(isnothing(n_virtual_coils), isnothing(energy_threshold)) || throw(ArgumentError("specify exactly one of n_virtual_coils and energy_threshold"))

    calibration_matrix, _, _ = _coil_last_matrix(calibration_data, coil_dim)
    all(isfinite, calibration_matrix) || throw(ArgumentError("calibration_data must contain only finite values"))
    n_observation, n_coil = size(calibration_matrix)
    maximum_rank = min(n_observation, n_coil)

    whitening_matrix, effective_noise_covariance, effective_scale_factor = _noise_whitening_matrix(noise_covariance, n_coil, T, prewhitening_scale_factor)
    whitened_calibration = calibration_matrix * whitening_matrix
    decomposition = svd(whitened_calibration; full = false)
    singular_values = Vector{T}(decomposition.S)
    energy = sum(abs2, singular_values)
    energy > zero(T) || throw(ArgumentError("calibration_data must have nonzero signal energy"))

    rank = if !isnothing(n_virtual_coils)
        requested_rank = Int(n_virtual_coils)
        1 <= requested_rank <= maximum_rank || throw(ArgumentError("n_virtual_coils must be between 1 and min(nObservation, nCoil)=$maximum_rank"))
        requested_rank
    else
        threshold = T(energy_threshold)
        isfinite(threshold) && zero(T) < threshold <= one(T) || throw(ArgumentError("energy_threshold must be finite and in (0, 1]"))
        cumulative_energy = cumsum(abs2.(singular_values))
        something(findfirst(>=(threshold * energy), cumulative_energy), maximum_rank)
    end

    retained_energy = sum(abs2, @view(singular_values[1:rank])) / energy
    compression_matrix = whitening_matrix * @view(decomposition.V[:, 1:rank])
    return CoilCompressionTransform(
        Matrix{Complex{T}}(compression_matrix),
        singular_values,
        retained_energy,
        effective_noise_covariance,
        effective_scale_factor,
    )
end

"""
    apply_coil_compression(array, transform; coil_dim=ndims(array))

Apply `transform` along `coil_dim` and return an `Array` with that dimension
replaced by the virtual-coil count. The input coil count must equal
`size(transform, 1)`. Use the same fitted transform for acquired data and
coil-sensitivity maps so that every encoding operator retains the same signal
model after compression.
"""
function apply_coil_compression(
    array     :: AbstractArray{Complex{T}},
    transform :: CoilCompressionTransform ;
    coil_dim  :: Integer = ndims(array)   ,
) where T <: AbstractFloat
    array_matrix, coil_last_shape, permutation = _coil_last_matrix(array, coil_dim)
    size(array_matrix, 2) == size(transform, 1) || throw(DimensionMismatch("array has $(size(array_matrix, 2)) coils but transform expects $(size(transform, 1))"))

    compressed_matrix = array_matrix * transform.compression_matrix
    compressed_coil_last_shape = (coil_last_shape[1:(end-1)]..., size(transform, 2))
    compressed_coil_last = reshape(compressed_matrix, compressed_coil_last_shape)
    return permutedims(compressed_coil_last, invperm(permutation))
end

"""
    compress_coils(data, csm;
        data_coil_dim=ndims(data), csm_coil_dim=ndims(csm), n_virtual_coils=nothing,
        energy_threshold=nothing, noise_covariance=nothing, prewhitening_scale_factor=1
    ) -> compressed_data, compressed_csm, transform

Fit a coil-compression transform from `data` and apply it consistently to
both `data` and `csm`. Pass exactly one of `n_virtual_coils` and
`energy_threshold`. Supply `noise_covariance` for noise-prewhitened SVD, which
is preferred when a representative noise acquisition is available. Supply
`prewhitening_scale_factor` when its dwell time or effective receiver
bandwidth differs from the MRI acquisition. For
calibration-region fitting, call
[`fit_coil_compression`](@ref) on a calibration view and then apply the result
to both full arrays with [`apply_coil_compression`](@ref).
"""
function compress_coils(
    data                      :: AbstractArray{Complex{T}}                   ,
    csm                       :: AbstractArray{Complex{T}}                   ;
    data_coil_dim             :: Integer                       = ndims(data) ,
    csm_coil_dim              :: Integer                       = ndims(csm)  ,
    n_virtual_coils           :: Union{Nothing,Integer}        = nothing     ,
    energy_threshold          :: Union{Nothing,Real}           = nothing     ,
    noise_covariance          :: Union{Nothing,AbstractMatrix} = nothing     ,
    prewhitening_scale_factor :: Real                          = 1           ,
) where T <: AbstractFloat
    transform = fit_coil_compression(
        data;
        coil_dim = data_coil_dim,
        n_virtual_coils = n_virtual_coils,
        energy_threshold = energy_threshold,
        noise_covariance = noise_covariance,
        prewhitening_scale_factor = prewhitening_scale_factor,
    )
    compressed_data = apply_coil_compression(data, transform; coil_dim = data_coil_dim)
    compressed_csm  = apply_coil_compression(csm , transform; coil_dim = csm_coil_dim )
    return compressed_data, compressed_csm, transform
end
