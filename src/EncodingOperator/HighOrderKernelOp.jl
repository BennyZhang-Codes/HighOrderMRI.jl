export HighOrderKernelOp

"""
Kernel-based implementation of the explicit high-order signal encoding model.

The forward phase convention is `exp(+2πim * phase)`, the operator
normalization is `1 / sqrt(nVox)`, and the vectorized output ordering is
samples first and then channels.

- Supports 2D or 3D reconstruction with up to third-order dynamic fields.
- Supports CUDA kernel acceleration and multiple GPUs.
- Supports multiple receive channels, static off-resonance, and masking.
"""
mutable struct HighOrderKernelOp{T,F1,F2,S<:AbstractVector{T}} <: HOOp{T}
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
    Mv         :: S         # Dynamically adapted CPU/GPU Vector
    Mtu        :: S         # Dynamically adapted CPU/GPU Vector
end
LinearOperators.storage_type(op::HighOrderKernelOp) = typeof(op.Mv)

mutable struct HighOrderKernelGPUWorkspace{T<:AbstractFloat}
    gpu_id   :: Int
    voxels   :: UnitRange{Int}
    x        :: CuVector{Complex{T}}
    csm      :: CuMatrix{Complex{T}}
    fieldmap :: CuVector{T}
    bf       :: CuMatrix{T}
    times    :: CuVector{T}
    kspha    :: CuMatrix{T}
    signal   :: CuMatrix{Complex{T}}
    image    :: CuVector{Complex{T}}
end

function split_highorder_voxel_ranges(nVox::Int, nGPU::Int)
    nGPU <= nVox || throw(ArgumentError("number of GPUs must not exceed number of masked voxels"))
    nBase, nExtra = divrem(nVox, nGPU)
    ranges = Vector{UnitRange{Int}}(undef, nGPU)
    first_voxel = 1
    for i = 1:nGPU
        nLocal = nBase + (i <= nExtra)
        ranges[i] = first_voxel:first_voxel + nLocal - 1
        first_voxel += nLocal
    end
    return ranges
end

function HighOrderKernelGPUWorkspace(gpu_id::Int, voxels::UnitRange{Int}, ::Type{T}, nSam::Int, nCha::Int, nChaPad::Int, times, kspha, fieldmap, bf, csm) where {T<:AbstractFloat}
    CUDA.device!(gpu_id)
    nLocal = length(voxels)
    csm_local = zeros(Complex{T}, nLocal, nChaPad)
    @views csm_local[:, 1:nCha] .= csm[voxels, :]
    return HighOrderKernelGPUWorkspace{T}(gpu_id, voxels, CUDA.zeros(Complex{T}, nLocal), CuArray(csm_local), CuArray(fieldmap[voxels]), CuArray(bf[voxels, :]), CuArray(times), CuArray(kspha), CUDA.zeros(Complex{T}, nSam, nChaPad), CUDA.zeros(Complex{T}, nLocal))
end

function run_highorder_gpu_tasks!(f, workspaces)
    tasks = [Threads.@spawn begin CUDA.device!(workspace.gpu_id); f(workspace) end for workspace in workspaces]
    foreach(fetch, tasks)
    return nothing
end


"""
    HighOrderKernelOp(grid::Grid{T}, kspha::AbstractArray{T, 2}, times::AbstractVector{T}; kwargs...)

# Description

Construct a `HighOrderKernelOp` that explicitly evaluates the MRI high-order
Fourier encoding operator.

# Arguments

* `grid::Grid{T}`                      - Cartesian reconstruction grid.
* `kspha::AbstractArray{T, 2}`         - [nTerm, nSam], coefficients of field
  dynamics. `nTerm` must be `9` (up to second order) or `16` (up to third
  order).
* `times::AbstractVector{T}`           - [nSam], sampling time points.

# Keywords

* `fieldmap::AbstractArray{T}`          - [nX, nY, nZ], off-resonance map.
  [nX, nY] is accepted when `nZ == 1`.
* `csm::AbstractArray{Complex{T}}`      - [nX, nY, nZ, nCha], complex coil
  sensitivity maps. [nX, nY, nCha] is accepted when `nZ == 1`.
* `mask::AbstractArray{Bool}`           - [nX, nY, nZ], reconstruction mask.
  [nX, nY] is accepted when `nZ == 1`.
* `recon_terms::Union{Nothing, AbstractString}` - Binary order-selection
  string. Use three digits for `nTerm == 9` and four digits for
  `nTerm == 16`; the digits select zeroth-, first-, second-, and third-order
  terms. The default is `"111"` or `"1111"`.
* `k_nominal::AbstractArray{T, 2}`      - [3, nSam], nominal first-order
  trajectory ordered as [kx, ky, kz]. It replaces the measured first-order
  coefficients when first-order error is disabled by `recon_terms`.
* `kspha_dt`                            - [nTerm, nSam], optional time
  derivative of the field-dynamic coefficients.
* `nBlock::Int64`                       - Number of trajectory blocks used by
  the derivative path to limit temporary memory.
* `arrayType::Type{<:AbstractArray}`     - Compute backend. The kernel
  implementation requires `CuArray` (the default); solver vectors stay on the
  CPU while persistent workspaces execute on the selected GPUs.
* `gpus::Vector{Int}`                   - Zero-based CUDA device IDs.
* `verbose::Bool`                       - Print progress information; default
  is `false`.
"""
function HighOrderKernelOp(
    grid        :: Grid{T}                                                             ,
    kspha       :: AbstractArray{T, 2}                                                 , 
    times       :: AbstractVector{T}                                                   ;
    fieldmap    :: AbstractArray{T}          = zeros(T, grid.matrixSize...)            , 
    csm         :: AbstractArray{Complex{T}} = ones(Complex{T}, grid.matrixSize..., 1) , 
    mask        :: AbstractArray{Bool}       = trues(grid.matrixSize...)               ,
    recon_terms :: Union{Nothing,AbstractString} = nothing                            ,
    k_nominal   :: AbstractArray{T, 2}       = kspha[2:4, :]                           ,
    kspha_dt                                 = nothing                                 ,
    nBlock      :: Int64                     = 50                                      , 
    arrayType   :: Type{<:AbstractArray}     = CuArray                                 ,
    gpus        :: Vector{Int}               = [0]                                     ,
    verbose     :: Bool                      = false                                   , 
    ) where {T<:AbstractFloat}

    arrayType <: CuArray || throw(ArgumentError("HighOrderKernelOp requires arrayType=CuArray"))
    isempty(gpus) && throw(ArgumentError("gpus must contain at least one CUDA device id"))
    length(unique(gpus)) == length(gpus) || throw(ArgumentError("gpus contains duplicate device ids: $gpus"))

    nX, nY, nZ = grid.nX, grid.nY, grid.nZ
    nTerm, nSam = size(kspha)
    nCha = size(csm)[end]
    nRow = nSam * nCha
    nCol = prod(grid.matrixSize)
    nVox = sum(mask)

    fieldmap = ndims(fieldmap) == 2 ? reshape(fieldmap, nX, nY, 1) : fieldmap
    csm      = ndims(csm) == 3      ? reshape(csm, nX, nY, 1, nCha) : csm
    mask     = ndims(mask) == 2     ? reshape(mask, nX, nY, 1) : mask

    @info "HighOrderKernelOp nRow=$nRow [nSam*nCha=$nSam*$nCha], nCol=$nCol [prod(matrixSize=$((nX,nY,nZ))], nVox in mask=$nVox, gpus=$gpus"

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

    nVox > 0 || throw(ArgumentError("mask must contain at least one voxel"))
    nChaPad = 32 * cld(nCha, 32)
    ranges = split_highorder_voxel_ranges(nVox, length(gpus))
    warn_if_insufficient_gpu_worker_threads(length(gpus); operation=:explicit_highorder_kernel)
    workspace_tasks = [Threads.@spawn HighOrderKernelGPUWorkspace(gpus[i], ranges[i], T, nSam, nCha, nChaPad, times, kspha, fieldmap, bf, csm) for i in eachindex(gpus)]
    workspaces = HighOrderKernelGPUWorkspace{T}[fetch(task) for task in workspace_tasks]
    forward_partial = zeros(Complex{T}, nSam, nCha)

    if isnothing(kspha_dt)
        func_prod = (res,xm)->begin
            values = prod_HighOrderKernelOp(xm, mask, workspaces, forward_partial, nSam, nCha, nTerm, nVox; verbose=verbose)
            copyto!(res, values)
            res
        end
    else # for calculation of Bx (2023, https://doi.org/10.1002/mrm.29460)
        @assert size(kspha_dt) == size(kspha) "kspha_dt must have same size as kspha"
        func_prod = (res,xm)->begin
            values = prod_dt_HighOrderOp(xm, mask, bf, nVox, nSam, nCha, kspha, kspha_dt, times, fieldmap, csm; nBlock=nBlock, parts=parts, use_gpu=false, verbose=verbose)
            copyto!(res, values)
            res
        end
    end
    func_ctprod = (res,ym)->begin
        values = ctprod_HighOrderKernelOp(ym, mask, workspaces, nSam, nCha, nTerm, nVox; verbose=verbose)
        copyto!(res, values)
        res
    end

    Mv = Mtu = Vector{Complex{T}}(undef, 0)
    
    return HighOrderKernelOp{Complex{T},Nothing,Function,typeof(Mv)}(
                        nRow, nCol, 
                        false, false,
                        func_prod, nothing, func_ctprod,
                        0, 0, 0, 
                        Mv, Mtu)
end


"""
    Forward operator for HighOrderKernelOp
"""
function prod_HighOrderKernelOp(
    x               :: AbstractVector{T},
    mask            :: AbstractVector{Bool},
    workspaces      :: AbstractVector{<:HighOrderKernelGPUWorkspace{D}},
    forward_partial :: Matrix{Complex{D}},
    nSam            :: Int64,
    nCha            :: Int64,
    nTerm           :: Int64,
    nVox            :: Int64;
    verbose         :: Bool = false,
    ) where {D<:AbstractFloat, T<:Union{Real,Complex}}
    if verbose @info "HighOrderOp: Kernel-based prod" end
    t_total = @elapsed begin
        x_masked = Vector(x)[mask]
        out = zeros(Complex{D}, nSam, nCha)
        run_highorder_gpu_tasks!(workspaces) do workspace
            copyto!(workspace.x, 1, x_masked, first(workspace.voxels), length(workspace.voxels))
            run_kernel_prod_workspace!(workspace.signal, workspace.x, workspace.csm, workspace.times, workspace.fieldmap, workspace.bf, workspace.kspha, nSam, nCha, nTerm, length(workspace.voxels))
            CUDA.device_synchronize(; blocking=true)
        end
        for workspace in workspaces
            CUDA.device!(workspace.gpu_id)
            copyto!(forward_partial, 1, workspace.signal, 1, length(forward_partial))
            out .+= forward_partial
        end
        out ./= sqrt(nVox)
    end
    if verbose println("runtime: $(round(t_total, digits=5)) [s]") end
    return vec(out)
end


"""
    Adjoint of prod_HighOrderKernelOp
"""
function ctprod_HighOrderKernelOp(
    y          :: AbstractVector{T},
    mask       :: AbstractVector{Bool},
    workspaces :: AbstractVector{<:HighOrderKernelGPUWorkspace{D}},
    nSam       :: Int64,
    nCha       :: Int64,
    nTerm      :: Int64,
    nVox       :: Int64;
    verbose    :: Bool = false,
    ) where {D<:AbstractFloat, T<:Union{Real,Complex}}
    if verbose @info "HighOrderOp: Kernel-based ctprod" end
    t_total = @elapsed begin
        y_matrix = reshape(y, nSam, nCha)
        x_masked = zeros(Complex{D}, nVox)
        run_highorder_gpu_tasks!(workspaces) do workspace
            fill!(workspace.signal, zero(Complex{D}))
            copyto!(workspace.signal, 1, y_matrix, 1, length(y_matrix))
            run_kernel_ctprod!(workspace.image, workspace.signal, workspace.csm, workspace.times, workspace.fieldmap, workspace.bf, workspace.kspha, nSam, nCha, nTerm, length(workspace.voxels))
            CUDA.device_synchronize(; blocking=true)
            copyto!(x_masked, first(workspace.voxels), workspace.image, 1, length(workspace.voxels))
        end
        x_masked ./= sqrt(nVox)
    end
    if verbose println("runtime: $(round(t_total, digits=5)) [s]") end
    x = zeros(Complex{D}, size(mask))
    x[mask] .= x_masked
    return x
end


function Base.adjoint(op::HighOrderKernelOp{T}) where T
  return LinearOperator{T}(
                            op.ncol, 
                            op.nrow, 
                            op.symmetric, 
                            op.hermitian,
                            op.ctprod!, 
                            nothing, 
                            op.prod!)
end
