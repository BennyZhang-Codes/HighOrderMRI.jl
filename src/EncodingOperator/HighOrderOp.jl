export HighOrderOp

"""
# A Julia implementation of the expanded signal encoding model.
- This implementation using GPU with CUDA.jl to accelerate the calculation.
- If the GPU memory is not enough, the calculation can be divided into blocks.
"""
mutable struct HighOrderOp{T,F1,F2} <: HOOp{T}
    nrow       :: Int
    ncol       :: Int
    symmetric  :: Bool
    hermitian  :: Bool
    prod!      :: Function
    tprod!     :: F1
    ctprod!    :: F2
    nprod      :: Int
    ntprod     :: Int
    nctprod    :: Int
    args5      :: Bool
    use_prod5! :: Bool
    allocated5 :: Bool
    Mv5        :: Vector{T}
    Mtu5       :: Vector{T}
end
LinearOperators.storage_type(op::HighOrderOp) = typeof(op.Mv5)


"""
    HighOrderOp(grid::Grid{T}, kspha::AbstractArray{T, 2}, times::AbstractVector{T}; kwargs...)

# Description
    generates a `HighOrderOp` which explicitely evaluates the MRI Fourier HighOrder encoding operator.

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
function HighOrderOp(
    grid        :: Grid{T}                                                          ,
    kspha       :: AbstractArray{T, 2}                                              , 
    times       :: AbstractVector{T}                                                ;
    fieldmap    :: AbstractArray{T, 2}  = zeros(T,(grid.nX, grid.nY))               , 
    csm         :: Array{Complex{T}, 3} = ones(Complex{T},(grid.nX, grid.nY)..., 1) , 
    recon_terms :: String               = nothing                                   ,
    k_nominal   :: AbstractArray{T, 2}  = kspha[2:4, :]                             ,
    kspha_dt                            = nothing                                   ,
    nBlock      :: Int64                = 50                                        , 
    use_gpu     :: Bool                 = true                                      , 
    verbose     :: Bool                 = false                                     , 
    ) where {T<:AbstractFloat}

    nX, nY, nZ = grid.nX, grid.nY, grid.nZ
    nTerm, nSam = size(kspha)
    nCha = size(csm, 3)
    nRow = nSam * nCha
    nCol = nVox = prod(grid.matrixSize)
    if verbose
        @info "HighOrderOp nRow=$nRow, nCol=$nCol, nSam=$nSam, nCha=$nCha, nBlock=$nBlock, use_gpu=$use_gpu"
    end

    @assert nTerm              in [9, 16]   "kspha must have 9 or 16 terms (row) for up to 2nd or 3rd order terms"
    @assert size(k_nominal, 1) == 3         "k_nominal must have 3 terms (row) for kx, ky, kz"
    @assert size(fieldmap)     == (nX, nY)  "FieldMap must have same size as $((nX, nY)) in grid"
    @assert size(csm)[1:2]     == (nX, nY)  "Coil-SensitivityMap must have same size as $((nX, nY)) in grid"
    
    # prepare data 
    kspha    = prep_kspha(kspha, k_nominal, nTerm; recon_terms=recon_terms)
    csm      = reshape(csm, nX*nY, nCha)      # [nX * nY, nCha]
    fieldmap = vec(fieldmap)                  # [nVox]

    # divide the calculation into blocks (nBlock) to avoid memory overflow
    nBlock = nBlock > nSam  ? nSam  : nBlock  # nBlock must be <= k
    n      = nSam ÷ nBlock                    # number of sampless per block
    parts  = [n for i=1:nBlock]               # number of samples per block
    parts  = [1+n*(i-1):n*i for i=1:nBlock]
    if nSam%nBlock!= 0
        push!(parts, n*nBlock+1:nSam)
    end
    
    # if use_gpu, move all the variables to GPU
    if use_gpu
        kspha       = kspha       |> gpu
        kspha_dt    = kspha_dt    |> gpu
        grid        = grid        |> gpu
        times       = times       |> gpu
        fieldmap    = fieldmap    |> gpu
        csm         = csm         |> gpu
    end

    # compute basis functions (spherical harmonics)
    bf = basisfunc_spha(grid.x, grid.y, grid.z, collect(1:nTerm))

    if isnothing(kspha_dt)
        func_prod = (res,xm)->(res .= prod_HighOrderOp(xm, bf, nVox, nSam, nCha, kspha, times, fieldmap, csm; 
                                            nBlock=nBlock, parts=parts, use_gpu=use_gpu, verbose=verbose))
    else # for calculation of Bx (2023, https://doi.org/10.1002/mrm.29460)
        @assert size(kspha_dt) == size(kspha) "kspha_dt must have same size as kspha"
        func_prod = (res,xm)->(res .= prod_dt_HighOrderOp(xm, bf, nVox, nSam, nCha, kspha, kspha_dt, times, fieldmap, csm; 
                                            nBlock=nBlock, parts=parts, use_gpu=use_gpu, verbose=verbose))
    end
    func_ctprod = (res,ym)->(res .= ctprod_HighOrderOp(ym, bf, nVox, nSam, nCha, kspha, times, fieldmap, csm; 
                                            nBlock=nBlock, parts=parts, use_gpu=use_gpu, verbose=verbose))
    
    return HighOrderOp{Complex{T},Nothing,Function}(
                        nRow, nCol, 
                        false, false,
                        func_prod, nothing, func_ctprod,
                        0, 0, 0, 
                        false, false, false, 
                        Complex{T}[], Complex{T}[])
end


function prep_kspha(
    kspha         :: AbstractArray{T, 2} , 
    k_nominal     :: AbstractArray{T, 2} , 
    nTerm         :: Int64               ;
    recon_terms   :: String = nothing    ,
    verbose       :: Bool   = false      , 
    ) where T<:AbstractFloat
    if isnothing(recon_terms)
        recon_terms = nTerm == 9 ? "111" : "1111"
    end
    if nTerm == 9
        @assert length(recon_terms) == 3 "recon_terms must be 3 digits for up to 2nd order terms"
        t0 = Bool(parse(Int64, recon_terms[1]))
        t1 = Bool(parse(Int64, recon_terms[2]))
        t2 = Bool(parse(Int64, recon_terms[3]))
        t3 = false
    elseif nTerm == 16
        @assert length(recon_terms) == 4 "recon_terms must be 4 digits for up to 3rd order terms"
        t0 = Bool(parse(Int64, recon_terms[1]))
        t1 = Bool(parse(Int64, recon_terms[2]))
        t2 = Bool(parse(Int64, recon_terms[3]))
        t3 = Bool(parse(Int64, recon_terms[4]))
    else
        @error "nTerm must be 9 or 16"
    end

    if t0 == false
        kspha[1, :] = kspha[1, :] .* 0
    end
    if t1 == false
        kspha[2:4, :] = k_nominal[:, :]
    end
    if t2 == false
        kspha[5:9, :] = kspha[5:9, :] .* 0
    end
    if t3 == false && nTerm == 16
        kspha[10:16, :] = kspha[10:16, :] .* 0
    end
    if verbose
        @info "kspha prepared for flag: $(recon_terms)" zeroth=t0 first=t1 second=t2 third=t3
    end
    return kspha
end


"""
    prod_dt_HighOrderOp
    for calculation of Bx (2023, https://doi.org/10.1002/mrm.29460)
"""
function prod_dt_HighOrderOp(
    x         :: AbstractVector{T}                   , 
    bf        :: AbstractArray{D, 2}                 ,
    nVox      :: Int64                               ,
    nSam      :: Int64                               , 
    nCha      :: Int64                               ,
    kspha     :: AbstractArray{D, 2}                 , 
    kspha_dt  :: AbstractArray{D, 2}                 ,
    times     :: AbstractVector{D}                   , 
    fieldmap  :: AbstractVector{D}                   ,
    csm       :: AbstractArray{Complex{D}, 2}        ;
    nBlock    :: Int64                    = 1        , 
    parts     :: Vector{UnitRange{Int64}} = [1:nSam] , 
    use_gpu   :: Bool                     = false    , 
    verbose   :: Bool                     = false    ,
    ) where {D<:AbstractFloat, T<:Union{Real,Complex}}
    x = Vector(x)
    if verbose
        @info "HighOrderOp prod_dt nBlock=$nBlock, use_gpu=$use_gpu"
    end

    # CPU fallback (simple / clear)
    if !use_gpu
        out = zeros(Complex{D}, nSam, nCha)
        progress_bar = Progress(nBlock)
        for (block, p) = enumerate(parts)
            ϕ = @view(times[p]) .* fieldmap' .+ (bf * @view(kspha[:,p]))'
            # derivative factor (bf * kspha_dt[:,p])' times 2iπ
            der = (bf * @view(kspha_dt[:,p]))' .* (2im*pi)
            e = exp.(2im*pi*ϕ) .* der
            out[p, :] = e * (x .* csm)
            if verbose
                next!(progress_bar, showvalues=[(:nBlock, block)])
            end
        end
        out .= out ./ sqrt(nVox)
        return vec(out)
    end

    # --- GPU-optimized path: reuse buffers, minimize allocations ---
    bf_gpu      = isa(bf, CuArray)       ? bf       : cu(bf)
    kspha_gpu   = isa(kspha, CuArray)    ? kspha    : cu(kspha)
    kspha_dt_gpu= isa(kspha_dt, CuArray) ? kspha_dt : cu(kspha_dt)
    times_gpu   = isa(times, CuArray)    ? times    : cu(times)
    field_gpu   = isa(fieldmap, CuArray) ? fieldmap : cu(fieldmap)
    csm_gpu     = isa(csm, CuArray)      ? csm      : cu(csm)
    x_gpu       = isa(x, CuArray)        ? x        : cu(x)

    # precompute x .* csm once (nVox x nCha)
    x_csm = x_gpu .* csm_gpu  # (nVox, nCha)

    out_gpu = CUDA.zeros(Complex{D}, nSam, nCha)

    # temporaries: layout (lenp, nVox)
    max_block = maximum(length.(parts))
    phi_gpu = CUDA.zeros(D, max_block, nVox)
    e_gpu   = CUDA.zeros(Complex{D}, max_block, nVox)

    progress_bar = Progress(nBlock)
    for (block, p) = enumerate(parts)
        lenp = length(p)

        ks_block    = view(kspha_gpu, :, p)       # (nTerm, lenp)
        ks_block_dt = view(kspha_dt_gpu, :, p)    # (nTerm, lenp)

        # tmp = bf * ks_block  -> (nVox, lenp)
        tmp    = bf_gpu * ks_block
        tmp_dt = bf_gpu * ks_block_dt

        # write transpose into phi_gpu[1:lenp, :]  -> (lenp, nVox)
        phi_view = view(phi_gpu, 1:lenp, :)
        phi_view .= permutedims(tmp)

        # add outer product times_block * field_gpu' -> (lenp, nVox)
        times_block = view(times_gpu, p)         # (lenp,)
        @. phi_view .+= times_block * field_gpu'

        # compute complex exponentials (lenp x nVox)
        @. e_gpu[1:lenp, :] = cispi(2.0 * phi_view)

        # multiply by derivative term (tmp_dt' is lenp x nVox) and 2im*pi
        # elementwise multiply: e .= e .* (tmp_dt' * (2im*pi))
        @. e_gpu[1:lenp, :] .*= permutedims(tmp_dt) * (2im*pi)

        # compute out_block = e * (x .* csm)  -> (lenp x nCha)
        out_block = e_gpu[1:lenp, :] * x_csm   # (lenp x nCha)

        # write into global output
        @views out_gpu[p, :] .= out_block

        if verbose
            next!(progress_bar, showvalues=[(:nBlock, block)])
        end
    end

    # normalization
    out_gpu .= out_gpu ./ sqrt(nVox)

    # bring result back to CPU
    CUDA.@sync out_cpu = Array(out_gpu)
    return vec(out_cpu)
end


""""
    Forward operator for HighOrderOp
"""
function prod_HighOrderOp(
    x         :: AbstractVector{T}                   , 
    bf        :: AbstractArray{D, 2}                 ,
    nVox      :: Int64                               ,
    nSam      :: Int64                               , 
    nCha      :: Int64                               ,
    kspha     :: AbstractArray{D, 2}                 , 
    times     :: AbstractVector{D}                   , 
    fieldmap  :: AbstractVector{D}                   ,
    csm       :: AbstractArray{Complex{D}, 2}        ;
    nBlock    :: Int64                    = 1        , 
    parts     :: Vector{UnitRange{Int64}} = [1:nSam] , 
    use_gpu   :: Bool                     = false    , 
    verbose   :: Bool                     = false    ,
    ) where {D<:AbstractFloat, T<:Union{Real,Complex}}
    x = Vector(x)
    if verbose
        @info "HighOrderOp prod nBlock=$nBlock, use_gpu=$use_gpu"
    end

    if !use_gpu
        # fallback to original (CPU) implementation for simplicity / clarity
        out = zeros(Complex{D}, nSam, nCha)
        progress_bar = Progress(nBlock)
        for (block, p) = enumerate(parts)
            ϕ = @view(times[p]) .* fieldmap' .+ (bf * @view(kspha[:,p]))'
            e = exp.(2*1im*pi*ϕ)
            out[p, :] =  e * (x .* csm)
            if verbose
                next!(progress_bar, showvalues=[(:nBlock, block)])
            end
        end
        out .= out ./ sqrt(nVox)
        return vec(out)
    end

    # --- GPU path: minimize allocations, reuse buffers per-block ---
    # ensure GPU arrays
    bf_gpu      = isa(bf, CuArray)      ? bf      : cu(bf)
    kspha_gpu   = isa(kspha, CuArray)   ? kspha   : cu(kspha)
    times_gpu   = isa(times, CuArray)   ? times   : cu(times)
    field_gpu   = isa(fieldmap, CuArray) ? fieldmap : cu(fieldmap)
    csm_gpu     = isa(csm, CuArray)     ? csm     : cu(csm)
    x_gpu       = isa(x, CuArray)       ? x       : cu(x)

    # precompute x .* csm once (nVox x nCha)
    x_csm = x_gpu .* csm_gpu  # CuArray (nVox, nCha)

    # prepare output on GPU
    out_gpu = CUDA.zeros(Complex{D}, nSam, nCha)

    # determine largest block size to preallocate temporaries
    max_block = maximum(length.(parts))

    # phi: samples x voxels (Float) for the largest block, e: complex samples x voxels
    phi_gpu = CUDA.zeros(D, max_block, nVox)
    e_gpu   = CUDA.zeros(Complex{D}, max_block, nVox)

    progress_bar = Progress(nBlock)
    for (block, p) = enumerate(parts)
        lenp = length(p)
        # kspha block (nTerm x lenp)
        ks_block = view(kspha_gpu, :, p)  # lazy view on GPU

        # tmp = bf * ks_block  --> (nVox x lenp)
        tmp = bf_gpu * ks_block  # CuArray (nVox, lenp)

        # copy transpose into phi_gpu[1:lenp, :]
        # phi = times[p] .* fieldmap' .+ (bf * kspha[:,p])'
        # tmp' is (lenp x nVox); write into phi_gpu view
        phi_view = view(phi_gpu, 1:lenp, :)
        phi_view .= permutedims(tmp)             # copy tmp' into phi_view
        # add outer product times_block * fieldmap'
        times_block = view(times_gpu, p)         # lenp
        # broadcasting outer product and add into phi_view
        @. phi_view .+= times_block * field_gpu' # efficient GPU broadcast

        # compute complex exponentials (samples x voxels)
        @. e_gpu[1:lenp, :] = cispi(2.0 * phi_view)

        # compute out_block = e * (x .* csm)  -> (lenp x nCha)
        # note: x_csm is (nVox x nCha), e_gpu[1:lenp,:] is (lenp x nVox)
        out_block = e_gpu[1:lenp, :] * x_csm   # CuArray (lenp x nCha)

        # write into global output
        @views out_gpu[p, :] .= out_block

        if verbose
            next!(progress_bar, showvalues=[(:nBlock, block)])
        end
    end

    # normalization
    out_gpu .= out_gpu ./ sqrt(nVox)

    # bring result back to CPU
    CUDA.@sync out_cpu = Array(out_gpu)
    return vec(out_cpu)
end


"""
    Adjoint of prod_HighOrderOp
"""
function ctprod_HighOrderOp(
    y         :: AbstractVector{T}                   , 
    bf        :: AbstractArray{D, 2}                 ,
    nVox      :: Int64                               , 
    nSam      :: Int64                               , 
    nCha      :: Int64                               ,
    kspha     :: AbstractArray{D, 2}                 , 
    times     :: AbstractVector{D}                   , 
    fieldmap  :: AbstractVector{D}                   ,
    csm       :: AbstractArray{Complex{D}, 2}        ;
    nBlock    :: Int64                    = 1        , 
    parts     :: Vector{UnitRange{Int64}} = [1:nSam] , 
    use_gpu   :: Bool                     = false    , 
    verbose   :: Bool                     = false    ,
    ) where {D<:AbstractFloat, T<:Union{Real,Complex}}

    # CPU fallback (kept simple / clear)
    if !use_gpu
        csmC = conj.(csm)
        ymat = reshape(y, nSam, nCha)
        out = zeros(Complex{D}, nVox, nCha)
        progress_bar = Progress(nBlock)
        for (block, p) = enumerate(parts)
            ϕ =  fieldmap .* @view(times[p])' .+ (bf * @view(kspha[:,p]))
            e = exp.(2im*pi*ϕ)
            out .+= conj(e) * ymat[p, :]
            if verbose
                next!(progress_bar, showvalues=[(:nBlock, block)])
            end
        end
        out .= out ./ sqrt(nVox)
        out .= out .* csmC
        return vec(sum(out, dims=2))
    end

    # --- GPU-optimized path: minimize allocations, reuse temporaries ---
    # ensure GPU arrays
    bf_gpu      = isa(bf, CuArray)      ? bf      : cu(bf)
    kspha_gpu   = isa(kspha, CuArray)   ? kspha   : cu(kspha)
    times_gpu   = isa(times, CuArray)   ? times   : cu(times)
    field_gpu   = isa(fieldmap, CuArray) ? fieldmap : cu(fieldmap)
    csm_gpu     = isa(csm, CuArray)     ? csm     : cu(csm)
    y_gpu       = isa(y, CuArray)       ? reshape(y, nSam, nCha) : cu(reshape(y, nSam, nCha))

    csmC_gpu = conj.(csm_gpu)

    # prepare output on GPU
    out_gpu = CUDA.zeros(Complex{D}, nVox, nCha)

    # determine largest block size to preallocate temporaries (lenp x nVox layout)
    max_block = maximum(length.(parts))
    phi_gpu = CUDA.zeros(D, max_block, nVox)               # (lenp, nVox)
    e_gpu   = CUDA.zeros(Complex{D}, max_block, nVox)      # (lenp, nVox)

    progress_bar = Progress(nBlock)
    for (block, p) = enumerate(parts)
        lenp = length(p)
        ks_block = view(kspha_gpu, :, p)          # (nTerm, lenp)

        # tmp = bf * ks_block  -> (nVox, lenp)
        tmp = bf_gpu * ks_block                  # (nVox, lenp)

        # write transpose into phi_gpu[1:lenp, :]  -> (lenp, nVox)
        phi_view = view(phi_gpu, 1:lenp, :)
        phi_view .= permutedims(tmp)             # copy tmp' into phi_view

        # add outer product times_block * field_gpu' -> (lenp, nVox)
        times_block = view(times_gpu, p)         # (lenp,)
        @. phi_view .+= times_block * field_gpu'  # broadcasted outer-product

        # compute complex exponentials (lenp x nVox)
        @. e_gpu[1:lenp, :] = cispi(2.0*phi_view)

        # accumulate: out_gpu += conj(e_block)' * y_block  where conj+transpose = adjoint
        e_block = view(e_gpu, 1:lenp, :)                 # (lenp, nVox)
        y_block = view(y_gpu, p, :)                      # (lenp, nCha)
        out_gpu .+= adjoint(e_block) * y_block           # (nVox, nCha)

        if verbose
            next!(progress_bar, showvalues=[(:nBlock, block)])
        end
    end

    # normalization and coil-multiplication
    out_gpu ./= sqrt(nVox)
    out_gpu .*= csmC_gpu

    # sum across coils and return to CPU
    sum_gpu = sum(out_gpu, dims=2)   # (nVox, 1)
    CUDA.@sync out_cpu = Array(sum_gpu)
    return vec(out_cpu)
end


function Base.adjoint(op::HighOrderOp{T}) where T
  return LinearOperator{T}(
                            op.ncol, 
                            op.nrow, 
                            op.symmetric, 
                            op.hermitian,
                            op.ctprod!, 
                            nothing, 
                            op.prod!)
end