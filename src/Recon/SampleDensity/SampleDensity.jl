
export SampleDensity

"""
    SampleDensity(tr::AbstractArray{T,2}, shape::Tuple) where T
    returns the sampling density for non-Cartersian trajectories.

# Arguments
- `tr::AbstractArray{T,2}`: 2D k-space trajectory.
- `shape::Tuple`: Shape of the image to reconstruct.

# Returns
- `weights::Vector{Complex{T}}`: Sample density function (SDF) of the k-space trajectory.

# Example
```julia
julia> tr = randn(100,2);
julia> shape = (128,128);
julia> weights = SampleDensity(tr, shape);
```
"""
function SampleDensity(tr::AbstractArray{T,2}, shape::Tuple) where T
    C = maximum(2*abs.(tr[:]));  #Normalize k-space to -.5 to .5 for NUFFT
    tr = tr ./ C;
    weights = Vector{Complex{T}}(undef, size(tr,2))
    plan = plan_nfft(Float64.(tr), shape, m=2, σ=2)
    weights = Complex{T}.(sqrt.(sdc(plan, iters=10)))
    return weights
end