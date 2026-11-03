export HighOrderOp

"""
    HighOrderOp

Explicit array-based MRI encoding operator on CPU or CUDA. It evaluates the
selected field phase directly, with forward sign `exp(+2πim * phase)` and
normalization `1/sqrt(count(mask))`. Input vectors span the full image grid;
output vectors order samples first and receive coils second. The ordinary
adjoint conjugates both phase and coil sensitivities.
"""
mutable struct HighOrderOp{T,F1,F2,S<:AbstractVector{T}} <: HOOp{T}
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
    Mv         :: S
    Mtu        :: S
end
LinearOperators.storage_type(op::HighOrderOp) = typeof(op.Mv)


"""
    HighOrderOp(grid::Grid{T},
        kspha::AbstractArray{T, 2},
        times::AbstractVector{T};
        fieldmap::AbstractArray{T} = zeros(T, grid.matrixSize...),
        csm::AbstractArray{Complex{T}} = ones(Complex{T}, grid.matrixSize..., 1),
        mask::AbstractArray{Bool} = trues(grid.matrixSize...),
        concomitant = nothing,
        recon_terms::Union{Nothing, AbstractString} = nothing,
        k_nominal::AbstractArray{T, 2} = kspha[2:4, :],
        kspha_dt = nothing,
        nBlock::Int64 = 50,
        arrayType::Type{<:AbstractArray} = CuArray,
        verbose::Bool = false) where T <: AbstractFloat

Construct an explicit encoding operator from integrated field coefficients
`kspha` of size `(9,nSam)` or `(16,nSam)`, and `times` of length `nSam`.
Use metres for grid coordinates, seconds for times, Hz for `fieldmap`, and
cycles/m^p for order-p coefficients (cycles for the constant term). The
constructor does not integrate fields or convert radians to cycles.

`fieldmap` and `mask` match `grid.matrixSize`; `csm` has an additional final
coil dimension. For `nZ == 1`, 2D maps/masks and `(nX,nY,nCha)` coil maps are
accepted. The mask must contain at least one voxel. Both precision and device
placement of solver inputs must match the chosen backend.

`recon_terms=nothing` selects all orders (`"111"` for 9 terms, `"1111"` for
16). A zero disables its order; disabling first-order error substitutes the
three rows of `k_nominal` rather than setting the Fourier trajectory to zero.
The default nominal rows are copied from `kspha`, so a distinct nominal
trajectory must be supplied for this replacement to change the model.

`concomitant`, when supplied, has shape `(4,nSam)` and contains integrated
coefficients in cycles/m² for `z²`, `x²+y²`, `xz`, `yz`, in the same frame and
origin as the grid. It is independent of `recon_terms` and ignored when
`kspha_dt` is supplied.

`nBlock` is a positive block count, capped at `nSam`; a remainder uses an
additional final block. `arrayType=Array` selects CPU computation; `CuArray`
selects CUDA. `verbose=false` does not suppress every setup message.

With `kspha_dt`, the forward product evaluates the trajectory-derivative
expression used for delay estimation. The current adjoint still evaluates the
ordinary encoding adjoint: do not use this derivative object for adjoint-based
solves or normal products. The derivative does not include a concomitant
coefficient derivative.

Known precision limitation: the array-based CUDA product helpers use `gpu`,
which can downcast Float64 input vectors to Float32 internally. Use the CPU
path when an independently checked Float64 explicit reference is required.

The returned operator has size `(nSam*nCha, prod(grid.matrixSize))` and complex
element type `Complex{T}`. It represents the full image grid, including zero
columns outside the mask; the adjoint is a backprojection, not an inverse.
"""
function HighOrderOp(
    grid        :: Grid{T}                                                             ,
    kspha       :: AbstractArray{T, 2}                                                 , 
    times       :: AbstractVector{T}                                                   ;
    fieldmap    :: AbstractArray{T}          = zeros(T, grid.matrixSize...)            , 
    csm         :: AbstractArray{Complex{T}} = ones(Complex{T}, grid.matrixSize..., 1) , 
    mask        :: AbstractArray{Bool}       = trues(grid.matrixSize...)               ,
    concomitant                              = nothing                                 ,
    recon_terms :: Union{Nothing,AbstractString} = nothing                            ,
    k_nominal   :: AbstractArray{T, 2}       = kspha[2:4, :]                           ,
    kspha_dt                                 = nothing                                 ,
    nBlock      :: Int64                     = 50                                      , 
    arrayType   :: Type{<:AbstractArray}     = CuArray                                 ,
    verbose     :: Bool                      = false                                   , 
    ) where {T<:AbstractFloat}

    is_gpu = arrayType <: CuArray
    (is_gpu || arrayType <: Array) || throw(ArgumentError("HighOrderOp supports arrayType=Array or CuArray"))

    nX, nY, nZ = grid.nX, grid.nY, grid.nZ
    nTerm, nSam = size(kspha)
    nCha = size(csm)[end]
    nRow = nSam * nCha
    nCol = prod(grid.matrixSize)
    nVox = sum(mask)

    fieldmap = ndims(fieldmap) == 2 ? reshape(fieldmap, nX, nY, 1) : fieldmap
    csm      = ndims(csm) == 3      ? reshape(csm, nX, nY, 1, nCha) : csm
    mask     = ndims(mask) == 2     ? reshape(mask, nX, nY, 1) : mask

    @info "HighOrderOp nRow=$nRow [nSam*nCha=$nSam*$nCha], nCol=$nCol [prod(matrixSize=$((nX,nY,nZ))], nVox in mask=$nVox, nBlock=$nBlock, arrayType=$arrayType"

    @assert nTerm              in [9, 16]       "kspha must have 9 or 16 terms (row) for up to 2nd or 3rd order terms"
    @assert size(k_nominal, 1) == 3             "k_nominal must have 3 terms (row) for kx, ky, kz"
    @assert size(fieldmap)     == (nX, nY, nZ)  "FieldMap must have same size as $((nX, nY, nZ)) in grid"
    @assert size(csm)[1:3]     == (nX, nY, nZ)  "Coil-SensitivityMap must have same size as $((nX, nY, nZ)) in grid"
    @assert size(mask)         == (nX, nY, nZ)  "Mask must have same size as $((nX, nY, nZ)) in grid"

    mask     = vec(mask)                               # [prod(MatrixSize)]

    # prepare data 
    kspha    = prep_kspha(kspha, k_nominal, nTerm; recon_terms=recon_terms)
    csm      = reshape(csm, :, nCha)[mask.!=0, :]      # [nVox, nCha]
    fieldmap = vec(fieldmap)[mask.!=0]                 # [nVox]

    # divide the calculation into blocks (nBlock) to avoid memory overflow
    nBlock = nBlock > nSam  ? nSam  : nBlock  # nBlock must be <= k
    n      = nSam ÷ nBlock                    # number of sampless per block
    parts  = [n for i=1:nBlock]               # number of samples per block
    parts  = [1+n*(i-1):n*i for i=1:nBlock]
    if nSam%nBlock!= 0
        push!(parts, n*nBlock+1:nSam)
    end
    
    # compute basis functions (spherical harmonics)
    bf = basisfunc_spha(grid.x[mask.!=0], grid.y[mask.!=0], grid.z[mask.!=0], collect(1:nTerm))
    bf, kspha = prep_concomitant(bf, kspha, grid, mask; concomitant=isnothing(kspha_dt) ? concomitant : nothing)

    if is_gpu
        kspha    = arrayType(kspha)
        bf       = arrayType(bf)
        times    = arrayType(times)
        fieldmap = arrayType(fieldmap)
        csm      = arrayType(csm)
        if !isnothing(kspha_dt)
            kspha_dt = arrayType(kspha_dt)
        end
    end


    if isnothing(kspha_dt)
        func_prod = (res,xm)->begin
            values = prod_HighOrderOp(xm, mask, bf, nVox, nSam, nCha, kspha, times, fieldmap, csm; nBlock=nBlock, parts=parts, use_gpu=is_gpu, verbose=verbose)
            copyto!(res, arrayType(values))
            res
        end
    else # for calculation of Bx (2023, https://doi.org/10.1002/mrm.29460)
        @assert size(kspha_dt) == size(kspha) "kspha_dt must have same size as kspha"
        func_prod = (res,xm)->begin
            values = prod_dt_HighOrderOp(xm, mask, bf, nVox, nSam, nCha, kspha, kspha_dt, times, fieldmap, csm; nBlock=nBlock, parts=parts, use_gpu=is_gpu, verbose=verbose)
            copyto!(res, arrayType(values))
            res
        end
    end
    func_ctprod = (res,ym)->begin
        values = ctprod_HighOrderOp(ym, mask, bf, nVox, nSam, nCha, kspha, times, fieldmap, csm; nBlock=nBlock, parts=parts, use_gpu=is_gpu, verbose=verbose)
        copyto!(res, arrayType(values))
        res
    end

    Mv = Mtu = arrayType(Vector{Complex{T}}(undef, 0))
    
    return HighOrderOp{Complex{T},Nothing,Function,typeof(Mv)}(
                        nRow, nCol, 
                        false, false,
                        func_prod, nothing, func_ctprod,
                        0, 0, 0, 
                        Mv, Mtu)
end

"""
    prod_dt_HighOrderOp
    for calculation of Bx (2023, https://doi.org/10.1002/mrm.29460)
"""
function prod_dt_HighOrderOp(
    x         :: AbstractVector{T}                   , 
    mask      :: AbstractVector{Bool}                ,   # [prod(MatrixSize)]
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
    x = Vector(x)[mask.!=0]
    if verbose
        @info "HighOrderOp prod_dt nBlock=$nBlock, use_gpu=$use_gpu"
    end
    if use_gpu
        x   = x |> gpu
        out = CUDA.zeros(Complex{D}, nSam, nCha)
    else
        out = zeros(Complex{D}, nSam, nCha)
    end
    progress_bar = Progress(nBlock)
    for (block, p) = enumerate(parts)
        ϕ = @view(times[p]) .* fieldmap' .+ (bf * @view(kspha[:,p]))'
        e = exp.(2*1im*pi*ϕ) .* (bf * @view(kspha_dt[:,p]))' .* (2*1im*pi)
        out[p, :] =  e * (x .* csm)
        if verbose
            next!(progress_bar, showvalues=[(:nBlock, block)])
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
    Forward operator for HighOrderOp
"""
function prod_HighOrderOp(
    x         :: AbstractVector{T}                   , 
    mask      :: AbstractVector{Bool}                ,   # [prod(MatrixSize)]
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
    x = Vector(x)[mask.!=0]
    if verbose
        @info "HighOrderOp prod nBlock=$nBlock, use_gpu=$use_gpu"
    end
    if use_gpu
        x   = x |> gpu
        out = CUDA.zeros(Complex{D}, nSam, nCha)
    else
        out = zeros(Complex{D}, nSam, nCha)
    end
    progress_bar = Progress(nBlock)
    for (block, p) = enumerate(parts)
        ϕ = @view(times[p]) .* fieldmap' .+ (bf * @view(kspha[:,p]))'
        e = exp.(2*1im*pi*ϕ)
        out[p, :] =  e * (x .* csm)
        if verbose
            next!(progress_bar, showvalues=[(:nBlock, block)])
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
    Adjoint of prod_HighOrderOp
"""
function ctprod_HighOrderOp(
    y         :: AbstractVector{T}                   , 
    mask      :: AbstractVector{Bool}                ,   # [prod(MatrixSize)]
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
    csmC = conj.(csm)
    y    = reshape(y, nSam, nCha)

    if verbose
        @info "HighOrderOp ctprod nBlock=$nBlock, use_gpu=$use_gpu"
    end
    
    if use_gpu
        y   = y |> gpu
        out = CUDA.zeros(Complex{D}, nVox, nCha)
    else
        out = zeros(Complex{D}, nVox, nCha)
    end

    progress_bar = Progress(nBlock)
    for (block, p) = enumerate(parts)
        ϕ =  fieldmap .* @view(times[p])' .+ (bf * @view(kspha[:,p]))
        e = exp.(2*1im*pi*ϕ)
        out +=  conj(e) * y[p, :]
        if verbose
            next!(progress_bar, showvalues=[(:nBlock, block)])
        end
        if use_gpu
            CUDA.unsafe_free!(ϕ)
            CUDA.unsafe_free!(e)
        end
    end
    if use_gpu
        CUDA.unsafe_free!(y)
    end
    out = out ./ sqrt(nVox)
    out = out .* csmC
    if use_gpu
        out = out |> cpu
    end
    x = zeros(Complex{D}, size(mask))
    x[mask] .= vec(sum(out, dims=2))
    return x
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
