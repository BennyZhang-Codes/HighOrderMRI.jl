import MRIReco.MRIBase: samplingDensity
export samplingDensity

"""
    samplingDensity(tr::AbstractArray{T,2}, shape::Tuple) where T
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
julia> weights = samplingDensity(tr, shape);
```
"""
function samplingDensity(tr::AbstractArray{T,2}, shape::Tuple) where T
    C = maximum(2*abs.(tr[:]));  #Normalize k-space to -.5 to .5 for NUFFT
    tr = tr ./ C;
    plan = plan_nfft(Float64.(tr), shape, m=2, σ=2)
    weights = Complex{T}.(sqrt.(sdc(plan, iters=10)))
    return weights
end


# """
#     samplingDensity(acqData::AcquisitionData{T}, shape::Tuple) where T

# Overload sampling density calculation utilizing GPU SDC for non-Cartesian trajectories.
# Supports multiple contrasts/echoes.
# """
# function samplingDensity(acqData::AcquisitionData{T}, shape::Tuple) where T
#     @info "samplingDensity with GPU SDC (with SQRT!) for $(numContrasts(acqData)) contrasts..."
    
#     numContr = numContrasts(acqData)
#     weights = Array{Vector{Complex{T}}}(undef, numContr)
#     for echo = 1:numContr
#         tr = trajectory(acqData, echo)
#         if isCartesian(tr)
#             nodes = kspaceNodes(tr)[:, acqData.subsampleIndices[echo]]
#             weights[echo] = [Complex{T}(1.0 / sqrt(prod(shape))) for _ = 1:size(nodes, 2)]
#         else
#             nodes = kspaceNodes(tr) 
#             raw_w = calculate_sdc_pipe(T.(nodes), shape) 
#             w_sqrt = sqrt.(raw_w) 
#             weights[echo] = Complex{T}.(w_sqrt)
#         end
#     end
#     return weights
# end


# function calculate_sdc_pipe(k_traj, matrix_size; iters=15)
#     @info "Calculating Density Compensation (Pipe's Method on GPU)..."
    
#     plan = plan_nfft(CuArray, k_traj, matrix_size; m=3, σ=1.25)
    
#     n_pts = size(k_traj, 2)
#     w = CUDA.ones(Float32, n_pts)
#     w_c = CUDA.ones(ComplexF32, n_pts)
    
#     img_tmp = CUDA.zeros(ComplexF32, matrix_size)
#     k_tmp = CUDA.zeros(ComplexF32, n_pts)
    
#     for i in 1:iters
#         w_c .= complex.(w)
#         mul!(img_tmp, plan', w_c)      # Adjoint (Gridding)
#         mul!(k_tmp, plan, img_tmp)     # Forward (Degridding)
#         # Pipe's update rule
#         @. w = w / (abs(k_tmp) + 1e-6)
#     end
    
#     w ./= (sum(w) / length(w))
#     w_cpu = Array(w) 
    
#     CUDA.unsafe_free!(w)
#     CUDA.unsafe_free!(w_c)
#     CUDA.unsafe_free!(img_tmp)
#     CUDA.unsafe_free!(k_tmp)

#     return w_cpu
# end