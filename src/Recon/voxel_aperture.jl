"""
    voxel_aperture(grid::Grid{T}, k::AbstractArray{T}) where T <: AbstractFloat

Return the host vector `prod(sinc.(k .* voxel_spacing); dims=1)` for an
axis-aligned uniform box voxel. `k` contains prepared first-order coefficients
`[kx,ky,kz]` in cycles/metre, with shape `(3,nPoint)` or `(3,nSam,nDyn)`.
Grid spacings are metres. The result has precision `T` and sample-fast,
readout-next order, shared by all receive coils. Inputs are not modified.

This is the exact normalized box integral of the linear spatial phase. B0,
CSM and other spatial factors are treated as constant within each voxel;
variation of higher-order phase within a voxel is not integrated here.
This optional encoding model is distinct from sampling-density weights.

Pass the result, moved to the reconstruction backend when necessary, as
`encoding_weights` to [`recon_HOOp`](@ref). That wrapper applies the aperture
to the encoding while applying raw data weights exactly once. The default
operator and reconstruction behavior remain point-voxel encoding.
"""
function voxel_aperture(grid::Grid{T}, k::AbstractArray{T}) where T<:AbstractFloat
    ndims(k) in (2, 3) && size(k, 1) == 3 || throw(DimensionMismatch("Expected three first-order coefficients"))
    spacing = T[grid.Δx, grid.Δy, grid.Δz]
    all(x -> isfinite(x) && x > 0, spacing) || throw(ArgumentError("Voxel spacings must be finite and positive"))
    coefficients = reshape(Array(k), 3, :)
    all(isfinite, coefficients) || throw(ArgumentError("First-order coefficients must be finite"))
    return vec(prod(sinc.(coefficients .* spacing); dims=1))
end
