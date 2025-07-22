export fastHighOrderOp

# Prepare NUFFT operator using MRIReco.jl (or your preferred NUFFT package)
using NFFT
using LinearAlgebra
using CUDA
using NNlib, cuDNN

"""
# A Julia implementation of the expanded signal encoding model.
- This implementation using GPU with CUDA.jl to accelerate the calculation
- Fast Higher Order Encoding Operator based on the paper by Wilm et al. and adapted from Corey Baron's MATMRI package
"""
mutable struct fastHighOrderOp{T,F1,F2} <: fHOOp{T}
    nrow::Int
    ncol::Int
    symmetric::Bool
    hermitian::Bool
    prod!::Function
    tprod!::F1
    ctprod!::F2
    nprod::Int
    ntprod::Int
    nctprod::Int
    args5::Bool
    use_prod5!::Bool
    allocated5::Bool
    Mv5::Vector{T}
    Mtu5::Vector{T}
end
LinearOperators.storage_type(op::fastHighOrderOp) = typeof(op.Mv5)


"""
    fastHighOrderOp(grid::Grid{T}, kspha::AbstractArray{T, 2}, times::AbstractVector{T}; kwargs...)

# Description
    generates a `fastHighOrderOp` which approximately evaluates the MRI Fourier HighOrder encoding operator.

# Arguments:
* `grid::Grid{T}`                   - grid object.
* `kspha::AbstractArray{T, 2}`      - [nSam, nTerm], Coefficients of field dynamics.
* `times::AbstractVector{T}`        - [nSam], time points for trajectory.

# Keywords:
* `fieldmap::Matrix{T}`             - [nX, nY], fieldmap for off-resonance correction.
* `csm::Array{Complex{T}, 3}`       - [nX, nY, nCha], coil sensitivity map.
* `recon_terms::String`             - digits flag (e.g. "111") to indicate terms to be used in the HOOp.
* `k_nominal::AbstractArray{T, 2}`  - [nSam, 3], nominal kspace trajectory.
* `kspha_dt`                        - [nSam, nTerm], time-derivative of the coefficients of field dynamics.
* `nBlock::Int64`                   - split trajectory into `nBlock` blocks to avoid memory overflow.
* `use_gpu::Bool`                   - use GPU for HighOrder encoding/decoding(default: `true`).
* `verbose::Bool`                   - print progress information(default: `false`).
"""
function fastHighOrderOp(
    grid::Grid{T},
    kspha::AbstractArray{T,2},
    times::AbstractVector{T};
    fieldmap::AbstractArray{T,2} = zeros(T, (grid.nX, grid.nY)),
    csm::Array{Complex{T},3} = ones(Complex{T}, (grid.nX, grid.nY)..., 1),
    recon_terms::String = nothing,
    k_nominal::AbstractArray{T,2} = kspha[2:4, :],
    kspha_dt = nothing,
    nBlock::Int64 = 50,
    use_gpu::Bool = true,
    verbose::Bool = false,
) where {T<:AbstractFloat}

    nX, nY, nZ = grid.nX, grid.nY, grid.nZ
    nTerm, nSam = size(kspha)
    nCha = size(csm, 3)
    nRow = nSam * nCha
    nCol = nVox = prod(grid.matrixSize)
    if verbose
        @info "fastHighOrderOp nRow=$nRow, nCol=$nCol, nSam=$nSam, nCha=$nCha, nBlock=$nBlock, use_gpu=$use_gpu"
    end

    @assert nTerm in [9, 16] "kspha must have 9 or 16 terms (row) for up to 2nd or 3rd order terms"
    @assert size(k_nominal, 1) == 3 "k_nominal must have 3 terms (row) for kx, ky, kz"
    @assert size(fieldmap) == (nX, nY) "FieldMap must have same size as $((nX, nY)) in grid"
    @assert size(csm)[1:2] == (nX, nY) "Coil-SensitivityMap must have same size as $((nX, nY)) in grid"

    # prepare data 
    kspha = prep_kspha(kspha, k_nominal, nTerm; recon_terms = recon_terms)
    csm = reshape(csm, nX * nY, nCha)      # [nX * nY, nCha]
    fieldmap = vec(fieldmap)                  # [nVox]

    use_gpu = false

    # since we modify kspha in place, import it new again before running

    kspha_stitched = kspha'
    kspha = -kspha_stitched

    # set up the undersampling scheme
    num_L = 20 # number of basis functions to use, i.e., the number of singular values to keep
    reduc_fac_space = 2 # reduction factor in space (linear interpolation by meanpooling and upsampling)
    reduc_fac_time = 300 # reduction factor in time (linear interpolation by meanpooling and upsampling)
    bf_choice = collect(5:9) # note, we intentionally skip the zeroth order term since we can directly demodulate the data by the zeroth order phase term
    num_basis = length(bf_choice)

    # compute basis functions (solid harmonics)
    bf = basisfunc_spha(grid.x, grid.y, grid.z, bf_choice)
    
    # for ii = 1:size(bf,2)
    #     bf[:,ii] = rotl90(reshape(bf[:,ii],nX,nY))[:]
    # end

    s_bf_orig = size(bf)

    # put the basis functions on GPU
    if use_gpu
        bf = bf |> gpu
    end

    s_bf_orig = size(bf)

    # put the basis functions on GPU
    if use_gpu
        bf = bf |> gpu
    end

    # subsample the basis functions by meanpooling
    pooled_bf = meanpool(reshape(bf, s_bf_orig[1], 1, num_basis), (reduc_fac_space^3,))

    if use_gpu
        CUDA.unsafe_free!(bf)
    end

    s_kspha_orig = size(kspha)

    # put kspha on GPU
    if use_gpu
        kspha = reshape(kspha[:, bf_choice], s_kspha_orig[1], 1, num_basis) |> gpu
    else
        kspha = reshape(kspha[:, bf_choice], s_kspha_orig[1], 1, num_basis)
    end

    pooled_kspha = meanpool(kspha, (reduc_fac_time,))

    if use_gpu
        CUDA.unsafe_free!(kspha)
    end

    # make the small version of E
    E_lite = cispi.(-2 * dropdims(pooled_bf, dims = 2) * dropdims(pooled_kspha, dims = 2)')

    U_e, S_e, V_e = svd(E_lite) # There may be better ways to compute such a decomposition! Remember, we don't need the singular values alone, we just need an orthogonal decomposition that approximates the encoding operator.

    if use_gpu
        CUDA.unsafe_free!(E_lite)
    end

    bls = reshape(U_e[:, 1:num_L], size(U_e, 1), 1, num_L)
    bls_linear_fullsize = upsample_linear(bls; size = s_bf_orig[1])
    bls_fullsize = reshape(bls_linear_fullsize, nX, nY, nZ, 1, num_L)

    cls = reshape((V_e.*conj(S_e)')[:, 1:num_L], size(V_e, 1), 1, num_L)
    cls_fullsize = upsample_linear(cls; size = s_kspha_orig[1])

    if use_gpu
        bls_fullsize = bls_fullsize |> cpu
        cls_fullsize = cls_fullsize |> cpu
        U_e = U_e |> cpu
        S_e = S_e |> cpu
        V_e = V_e |> cpu
    end

    bls = dropdims(bls_fullsize, dims = 4)
    cls = repeat(dropdims(cls_fullsize, dims = 2), nCha,1)

    kspha_nufft = kspha_stitched[:, 2:4]

    # Prepare k-space trajectory for NUFFT (typically only 1st order: x and y, i.e., rows 2 and 3)
    ktraj_nufft = kspha_nufft[:, 1:2]
    ktraj_nufft = permutedims(ktraj_nufft, (2, 1))

    traj_input = Trajectory("custom", convert_rad_per_m_to_nfft(ktraj_nufft, 0.001), times, 0.0, 0.0, 1, 25000, 1, false, false)

    encoding_op = FieldmapNFFTOp((nX, nY), traj_input, -2 * pi * 1im .* reshape(fieldmap, nX, nY))

    full_op = ∘(DiagOp(encoding_op,nCha),SensitivityOp(reshape(csm,:,nCha),1))

    shape = (nX, nY)

    if isnothing(kspha_dt)
        func_prod = (res, xm) -> (res .= prod_fastHighOrderOp(xm, bls, cls, full_op, shape, reshape(csm,:,nCha); use_gpu = use_gpu, verbose = verbose))
    else # for calculation of Bx (2023, https://doi.org/10.1002/mrm.29460)
        @assert size(kspha_dt) == size(kspha) "kspha_dt must have same size as kspha"
        func_prod =
            (res, xm) -> (
                res .= prod_dt_fastHighOrderOp(
                    xm,
                    bf,
                    nVox,
                    nSam,
                    nCha,
                    kspha,
                    kspha_dt,
                    times,
                    fieldmap,
                    csm;
                    nBlock = nBlock,
                    parts = parts,
                    use_gpu = use_gpu,
                    verbose = verbose,
                )
            )
    end
    func_ctprod = (res, ym) -> (res .= ctprod_fastHighOrderOp(ym, bls, cls, full_op, shape, reshape(conj.(csm),:,nCha); use_gpu = use_gpu, verbose = verbose))

    return fastHighOrderOp{Complex{T},Nothing,Function}(
        nRow,
        nCol,
        false,
        false,
        func_prod,
        nothing,
        func_ctprod,
        0,
        0,
        0,
        false,
        false,
        false,
        Complex{T}[],
        Complex{T}[],
    )
end


# function prep_kspha(
#     kspha::AbstractArray{T,2},
#     k_nominal::AbstractArray{T,2},
#     nTerm::Int64;
#     recon_terms::String=nothing,
#     verbose::Bool=false,
# ) where T<:AbstractFloat
#     if isnothing(recon_terms)
#         recon_terms = nTerm == 9 ? "111" : "1111"
#     end
#     if nTerm == 9
#         @assert length(recon_terms) == 3 "recon_terms must be 3 digits for up to 2nd order terms"
#         t0 = Bool(parse(Int64, recon_terms[1]))
#         t1 = Bool(parse(Int64, recon_terms[2]))
#         t2 = Bool(parse(Int64, recon_terms[3]))
#         t3 = false
#     elseif nTerm == 16
#         @assert length(recon_terms) == 4 "recon_terms must be 4 digits for up to 3rd order terms"
#         t0 = Bool(parse(Int64, recon_terms[1]))
#         t1 = Bool(parse(Int64, recon_terms[2]))
#         t2 = Bool(parse(Int64, recon_terms[3]))
#         t3 = Bool(parse(Int64, recon_terms[4]))
#     else
#         @error "nTerm must be 9 or 16"
#     end

#     if t0 == false
#         kspha[1, :] = kspha[1, :] .* 0
#     end
#     if t1 == false
#         kspha[2:4, :] = k_nominal[:, :]
#     end
#     if t2 == false
#         kspha[5:9, :] = kspha[5:9, :] .* 0
#     end
#     if t3 == false && nTerm == 16
#         kspha[10:16, :] = kspha[10:16, :] .* 0
#     end
#     if verbose
#         @info "kspha prepared for flag: $(recon_terms)" zeroth = t0 first = t1 second = t2 third = t3
#     end
#     return kspha
# end


"""
    prod_dt_fastHighOrderOp
    for calculation of Bx (2023, https://doi.org/10.1002/mrm.29460)
"""
function prod_dt_fastHighOrderOp(
    x::AbstractVector{T},
    bf::AbstractArray{D,2},
    nVox::Int64,
    nSam::Int64,
    nCha::Int64,
    kspha::AbstractArray{D,2},
    kspha_dt::AbstractArray{D,2},
    times::AbstractVector{D},
    fieldmap::AbstractVector{D},
    csm::AbstractArray{Complex{D},2};
    nBlock::Int64 = 1,
    parts::Vector{UnitRange{Int64}} = [1:nSam],
    use_gpu::Bool = false,
    verbose::Bool = false,
) where {D<:AbstractFloat,T<:Union{Real,Complex}}
    x = Vector(x)
    if verbose
        @info "fastHighOrderOp prod_dt nBlock=$nBlock, use_gpu=$use_gpu"
    end
    if use_gpu
        x = x |> gpu
        out = CUDA.zeros(Complex{D}, nSam, nCha)
    else
        out = zeros(Complex{D}, nSam, nCha)
    end
    progress_bar = Progress(nBlock)
    for (block, p) in enumerate(parts)
        ϕ = @view(times[p]) .* fieldmap' .+ (bf * @view(kspha[:, p]))'
        e = exp.(2 * 1im * pi * ϕ) .* (bf * @view(kspha_dt[:, p]))' .* (2 * 1im * pi)
        out[p, :] = e * (x .* csm)
        if verbose
            next!(progress_bar, showvalues = [(:nBlock, block)])
        end
        if use_gpu
            CUDA.unsafe_free!(ϕ)
            CUDA.unsafe_free!(e)
        end
    end
    if use_gpu
        CUDA.unsafe_free!(x)
    end
    out = out ./ sqrt(nVox)
    if use_gpu
        out = out |> cpu
    end
    return vec(out)
end


"""
    Forward operator for fastHighOrderOp
"""
function prod_fastHighOrderOp(
    x::AbstractVector{T}, 
    bls::AbstractArray{T,4},
    cls::AbstractArray{T,2},
    encoding_op::AbstractLinearOperator{T},
    shape,
    csm;
    use_gpu::Bool = false,
    verbose::Bool = false,
) where {D<:AbstractFloat,T<:Union{Real,Complex}}

    # Perform the multiplication with the basis functions and coefficients
    # This is a simplified version of the multiplication
    nL = size(bls, 4)

    x = reshape(x, shape[1], shape[2], 1)

    result = zeros(Complex{Float64}, size(cls))
    # print(size(x))
    # print(size(bls))
    # print(size(cls))
    # print(size(encoding_op))
    for l = 1:nL
        @views result[:, l] = cls[:, l] .* reshape(encoding_op * dropdims(conj.(bls[:, :, 1, l]) .* x, dims = 3)[:], size(cls, 1))
    end
    return vec(sum(result, dims = 2))
end

"""
    Adjoint operator for fastHighOrderOp
"""
function ctprod_fastHighOrderOp(
    y::AbstractVector{T},
    bls::AbstractArray{T,4},
    cls::AbstractArray{T,2},
    encoding_op::AbstractLinearOperator{T},
    shape,
    csmC;
    use_gpu::Bool = false,
    verbose::Bool = false,
) where {D<:AbstractFloat,T<:Union{Real,Complex}}

    # Perform the multiplication with the basis functions and coefficients
    # This is a simplified version of the multiplication
    nL = size(bls, 4)

    result = zeros(Complex{Float64}, size(bls, 1), size(bls, 2), nL)
    for l = 1:nL
        @views result[:, :, l] = dropdims(bls, dims = 3)[:, :, l] .* reshape(encoding_op' * (conj.(cls[:, l]) .* y), (size(bls, 1), size(bls, 2)))
    end
    return vec(sum(result, dims = 3))
end


function Base.adjoint(op::fastHighOrderOp{T}) where {T}
    return LinearOperator{T}(
        op.ncol, 
        op.nrow, 
        op.symmetric, 
        op.hermitian, 
        op.ctprod!, 
        nothing, 
        op.prod!)
end

function convert_rad_per_m_to_nfft(x_rad_per_m, L::Float64)
    x_nfft = (x_rad_per_m .* L)
    # Wrap to [-0.5, 0.5)
    return mod.(x_nfft .+ 0.5, 1.0) .- 0.5
end

function Base.copy(S::fastHighOrderOp{T}) where T
    deepcopy(S)
end

function Base.copy(S::LinearOperator{T}) where T
    deepcopy(S)
end

