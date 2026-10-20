export explicit_gpu_normal

mutable struct ExplicitGPUState{T<:AbstractFloat}
    backend::Symbol
    phase_backend::Symbol
    gemm_backend::Symbol
    nSam::Int
    nVox::Int
    nCha::Int
    block_size::Int
    forward_threads::Int
    adjoint_threads::Int
    mask_idx::CuVector{Int32}
    csm::CuMatrix{Complex{T}}
    bf::CuMatrix{T}
    kspha::CuMatrix{T}
    times::CuVector{T}
    fieldmap::CuVector{T}
    x::CuVector{Complex{T}}
    image::CuVector{Complex{T}}
    signal::CuMatrix{Complex{T}}
    Z::CuMatrix{Complex{T}}
    Q::CuMatrix{Complex{T}}
    P::CuMatrix{Complex{T}}
    phase_cache::CuMatrix{Complex{T}}
    phase::CuMatrix{T}
    Y::CuMatrix{Complex{T}}
    released::Bool
end

function explicit_gemm!(state::ExplicitGPUState, transA::Char, transB::Char, α, A, B, β, C)
    if state.gemm_backend === :cublas
        return CUDA.cuBLAS.gemm!(transA, transB, α, A, B, β, C)
    end
    left = transA == 'N' ? A : transA == 'T' ? transpose(A) : adjoint(A)
    right = transB == 'N' ? B : transB == 'T' ? transpose(B) : adjoint(B)
    return mul!(C, left, right, α, β)
end

function activate_explicit_gpu!(state::ExplicitGPUState)
    state.released && throw(ArgumentError("operator is closed"))
    CUDA.device!(CUDA.device(state.x))
    return nothing
end

"""
    ExplicitGPUOp(grid, kspha, times; backend=:blocked, kwargs...)

Internal single-GPU implementation of `HighOrderKernelOp` resident backends.
Use the public `HighOrderKernelOp(...; gpus=[id], backend=...)` constructor.
`backend=:direct` reuses the existing CUDA encoding kernels, `:blocked` builds
one complex phase block and reuses it for both products of the weighted normal,
and `:cached` retains the complete complex phase matrix on the GPU while
applying it in sample blocks. `:cached_full` retains the same matrix and uses
one full matrix product per forward or adjoint, with a full sample-by-coil
workspace. Both cached modes generate their initial matrix in sample blocks.

All paths use positive forward phase in cycles and `1/sqrt(sum(mask))`.
`block_size` counts sample rows. `phase_backend=:kernel` fuses phase and complex
exponential; `:gemm` uses real GEMM followed by the same exponential. The latter
allocates an additional real phase block. `gemm_backend=:cublas` explicitly calls
cuBLAS for the real and complex matrix products; `:auto` uses Julia's `mul!`
dispatch and is retained for comparisons with the generic GPU matrix path.
The blocked/cached kernels do not
enable fast math; the reused legacy direct kernels retain their `@fastmath`.
The object owns mutable scratch space and is not reentrant. Call `close` after
the last use.
Model arrays are prepared on the host during construction. Pass GPU data and
weights to `recon_HOOp` so that CG vectors remain on the GPU. Input and output
vectors must belong to the GPU selected during construction. Applications and
`close` select that GPU even if the calling task has since changed devices.

# Arguments

- `grid::Grid{T}`: reconstruction grid, with `T` equal to `Float32` or `Float64`.
- `kspha`: `(nTerm, nSam)` field coefficients, with 9 or 16 terms.
- `times`: `nSam` sample times. Sample order and time origins are preserved.

# Keywords

- `fieldmap`, `csm`, `mask`, `recon_terms`, `k_nominal`: the same spatial
  inputs and order-selection convention as [`HighOrderKernelOp`](@ref).
- `backend`: `:direct`, `:blocked` (default), `:cached`, or `:cached_full`.
- `gpus=nothing`: use the current GPU, or pass one visible CUDA ID such as
  `gpus=[2]`. The public constructor selects the internal multi-GPU type for multiple IDs.
- `phase_cache_bytes=0`: optional byte limit for a prefix of complex phase
  blocks with `backend=:blocked`. The cache is generated during construction
  and reused by forward, adjoint, and normal applications. The limit counts
  cached `Complex{T}` entries only, rounds down to complete sample blocks,
  and applies separately to each GPU. Model, scratch, solver, and library
  storage require additional memory. Zero preserves full recomputation.
- `block_size=320`: sample rows per block, including cache construction;
  unlike the legacy `nBlock`, this is a block size, not a number of blocks.
- `phase_backend=:kernel`: fused phase generation, or `:gemm` for real GEMM.
- `gemm_backend=:cublas`: explicit cuBLAS calls, or `:auto` for generic dispatch.
- `forward_threads=256`, `adjoint_threads=512`: direct-kernel thread counts.
  The Float64 adjoint is capped at 128 threads to fit the register budget.
"""
mutable struct ExplicitGPUOp{T,F,G} <: HOOp{Complex{T}}
    nrow::Int
    ncol::Int
    symmetric::Bool
    hermitian::Bool
    prod!::F
    tprod!::Nothing
    ctprod!::G
    nprod::Int
    ntprod::Int
    nctprod::Int
    Mv::CuVector{Complex{T}}
    Mtu::CuVector{Complex{T}}
    state::ExplicitGPUState{T}
end

LinearOperators.storage_type(op::ExplicitGPUOp) = typeof(op.Mv)

function explicit_phase_kernel!(P, phase, bf, coefficients, times, fieldmap,
                                first_sample, rows, use_phase::Bool)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= rows * size(P, 2)
        row = (i - 1) % rows + 1
        voxel = (i - 1) ÷ rows + 1
        sample = first_sample + row - 1
        T = eltype(times)
        ϕ = times[sample] * fieldmap[voxel]
        if use_phase
            ϕ += phase[row, voxel]
        else
            for term in axes(bf, 2)
                ϕ = muladd(coefficients[term, sample], bf[voxel, term], ϕ)
            end
        end
        s, c = sincos(T(2π) * ϕ)
        P[row, voxel] = Complex{T}(c, s)
    end
    return nothing
end

function explicit_phase_block!(state::ExplicitGPUState, rows::UnitRange{Int})
    if last(rows) <= size(state.phase_cache, 1)
        return @view state.phase_cache[rows, :]
    end
    n = length(rows)
    P = @view state.P[1:n, :]
    use_phase = state.phase_backend === :gemm
    if use_phase
        T = eltype(state.times)
        explicit_gemm!(state, 'T', 'T', one(T), @view(state.kspha[:, rows]),
            state.bf, zero(T), @view(state.phase[1:n, :]))
    end
    @cuda threads=256 blocks=cld(n * state.nVox, 256) explicit_phase_kernel!(
        P, state.phase, state.bf, state.kspha, state.times, state.fieldmap,
        first(rows), n, use_phase)
    return P
end

function explicit_scatter_kernel!(out, masked, indices)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(indices)
        @inbounds out[indices[i]] = masked[i]
    end
    return nothing
end

function explicit_gather_kernel!(masked, x, indices)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(indices)
        @inbounds masked[i] = x[indices[i]]
    end
    return nothing
end

function explicit_gather!(state::ExplicitGPUState, x)
    @cuda threads=256 blocks=cld(state.nVox, 256) explicit_gather_kernel!(
        state.x, x, state.mask_idx)
    return nothing
end

function explicit_reduce_coils_kernel!(out, Q, csm, scale)
    voxel = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if voxel <= length(out)
        value = zero(eltype(out))
        for coil in axes(Q, 2)
            @inbounds value += conj(csm[voxel, coil]) * Q[voxel, coil]
        end
        @inbounds out[voxel] = value * scale
    end
    return nothing
end

function explicit_scatter!(out, state::ExplicitGPUState)
    fill!(out, zero(eltype(out)))
    @cuda threads=256 blocks=cld(state.nVox, 256) explicit_scatter_kernel!(
        out, state.image, state.mask_idx)
    return out
end

function explicit_direct_forward!(state::ExplicitGPUState{T}, x) where {T}
    explicit_gather!(state, x)
    threads = state.forward_threads
    bytes = size(state.bf, 2) * sizeof(T) + 64 * cld(threads, 32) * sizeof(T)
    for first_coil in 1:32:state.nCha
        coils = first_coil:first_coil+31
        @cuda threads=threads blocks=state.nSam shmem=bytes CUDA_kernel_prod_HighOrderOp!(
            @view(state.signal[:, coils]), state.x, @view(state.csm[:, coils]),
            state.times, state.fieldmap, state.bf, state.kspha,
            Int32(state.nSam), Int32(32), Int32(size(state.bf, 2)), Int32(state.nVox))
    end
    return nothing
end

function explicit_direct_adjoint!(state::ExplicitGPUState{T}) where {T}
    fill!(state.image, zero(Complex{T}))
    threads = state.adjoint_threads
    bytes = size(state.bf, 2) * sizeof(T) + 64 * cld(threads, 32) * sizeof(T)
    for first_coil in 1:32:state.nCha
        coils = first_coil:first_coil+31
        @cuda threads=threads blocks=state.nVox shmem=bytes CUDA_kernel_ctprod_HighOrderOp!(
            state.image, @view(state.signal[:, coils]), @view(state.csm[:, coils]),
            state.times, state.fieldmap, state.bf, state.kspha,
            Int32(state.nSam), Int32(32), Int32(size(state.bf, 2)), Int32(state.nVox))
    end
    return nothing
end

function explicit_gpu_forward!(out, state::ExplicitGPUState{T}, x) where {T}
    activate_explicit_gpu!(state)
    scale = inv(sqrt(T(state.nVox)))
    if state.backend === :direct
        explicit_direct_forward!(state, x)
        reshape(out, state.nSam, state.nCha) .=
            @view(state.signal[:, 1:state.nCha]) .* scale
    else
        explicit_gather!(state, x)
        state.Z .= @view(state.csm[:, 1:state.nCha]) .* state.x
        output = reshape(out, state.nSam, state.nCha)
        step = state.backend === :cached_full ? state.nSam : state.block_size
        for first_sample in 1:step:state.nSam
            rows = first_sample:min(first_sample+step-1, state.nSam)
            P = state.backend in (:cached, :cached_full) ? @view(state.P[rows, :]) : explicit_phase_block!(state, rows)
            explicit_gemm!(state, 'N', 'N', Complex{T}(scale), P, state.Z,
                zero(Complex{T}), @view(output[rows, :]))
        end
    end
    return out
end

function explicit_gpu_adjoint!(out, state::ExplicitGPUState{T}, y) where {T}
    activate_explicit_gpu!(state)
    scale = inv(sqrt(T(state.nVox)))
    if state.backend === :direct
        fill!(state.signal, zero(Complex{T}))
        @views state.signal[:, 1:state.nCha] .= reshape(y, state.nSam, state.nCha)
        explicit_direct_adjoint!(state)
        state.image .*= scale
    else
        fill!(state.Q, zero(Complex{T}))
        input = reshape(y, state.nSam, state.nCha)
        step = state.backend === :cached_full ? state.nSam : state.block_size
        for first_sample in 1:step:state.nSam
            rows = first_sample:min(first_sample+step-1, state.nSam)
            P = state.backend in (:cached, :cached_full) ? @view(state.P[rows, :]) : explicit_phase_block!(state, rows)
            explicit_gemm!(state, 'C', 'N', one(Complex{T}), P, @view(input[rows, :]),
                one(Complex{T}), state.Q)
        end
        @cuda threads=256 blocks=cld(state.nVox, 256) explicit_reduce_coils_kernel!(
            state.image, state.Q, state.csm, scale)
    end
    return explicit_scatter!(out, state)
end

function explicit_gpu_normal!(out, state::ExplicitGPUState{T}, D, x) where {T}
    activate_explicit_gpu!(state)
    if state.backend === :direct
        explicit_direct_forward!(state, x)
        signal = @view state.signal[:, 1:state.nCha]
        signal .*= D
        explicit_direct_adjoint!(state)
        state.image ./= T(state.nVox)
    else
        explicit_gather!(state, x)
        state.Z .= @view(state.csm[:, 1:state.nCha]) .* state.x
        fill!(state.Q, zero(Complex{T}))
        step = state.backend === :cached_full ? state.nSam : state.block_size
        for first_sample in 1:step:state.nSam
            rows = first_sample:min(first_sample+step-1, state.nSam)
            P = state.backend in (:cached, :cached_full) ? @view(state.P[rows, :]) : explicit_phase_block!(state, rows)
            Y = @view state.Y[1:length(rows), :]
            explicit_gemm!(state, 'N', 'N', one(Complex{T}), P, state.Z, zero(Complex{T}), Y)
            Y .*= @view D[rows, :]
            explicit_gemm!(state, 'C', 'N', one(Complex{T}), P, Y, one(Complex{T}), state.Q)
        end
        @cuda threads=256 blocks=cld(state.nVox, 256) explicit_reduce_coils_kernel!(
            state.image, state.Q, state.csm, inv(T(state.nVox)))
    end
    return explicit_scatter!(out, state)
end

function ExplicitGPUOp(grid::Grid{T}, kspha::AbstractMatrix{T}, times::AbstractVector{T};
    fieldmap=zeros(T, grid.matrixSize...), csm=ones(Complex{T}, grid.matrixSize..., 1),
    mask=trues(grid.matrixSize...), recon_terms=nothing, k_nominal=kspha[2:4, :],
    backend::Symbol=:blocked, block_size::Int=320, phase_backend::Symbol=:kernel,
    gemm_backend::Symbol=:cublas, gpus=nothing, phase_cache_bytes::Integer=0,
    forward_threads::Int=256, adjoint_threads::Int=512,
) where {T<:AbstractFloat}
    if gpus !== nothing
        ids = Int.(collect(gpus))
        length(ids) == 1 || throw(ArgumentError(
            "ExplicitGPUOp requires one GPU; use ExplicitMultiGPUOp for multiple GPUs"))
        0 <= only(ids) < length(collect(CUDA.devices())) || throw(ArgumentError("invalid visible GPU ID"))
        CUDA.device!(only(ids))
    end
    backend in (:direct, :blocked, :cached, :cached_full) || throw(ArgumentError("unknown explicit backend $backend"))
    phase_cache_bytes >= 0 || throw(ArgumentError("phase_cache_bytes must be nonnegative"))
    phase_cache_bytes == 0 || backend === :blocked || throw(ArgumentError(
        "phase_cache_bytes applies to backend=:blocked; cached backends already retain the complete phase matrix"))
    phase_backend in (:kernel, :gemm) || throw(ArgumentError("unknown phase backend $phase_backend"))
    gemm_backend in (:cublas, :auto) || throw(ArgumentError("gemm_backend must be :cublas or :auto"))
    block_size > 0 || throw(ArgumentError("block_size must be positive"))
    forward_threads in (64, 128, 256, 512) || throw(ArgumentError("unsupported forward_threads"))
    adjoint_threads in (64, 128, 256, 512) || throw(ArgumentError("unsupported adjoint_threads"))
    nTerm, nSam = size(kspha)
    nTerm in (9, 16) || throw(ArgumentError("kspha requires 9 or 16 terms"))
    length(times) == nSam || throw(DimensionMismatch("times and kspha"))
    length(mask) == prod(grid.matrixSize) || throw(DimensionMismatch("mask and grid"))
    length(fieldmap) == length(mask) || throw(DimensionMismatch("fieldmap and mask"))
    nCha = size(csm, ndims(csm))
    length(csm) == length(mask)*nCha || throw(DimensionMismatch("csm and grid"))
    indices = findall(vec(mask))
    nVox = length(indices)
    nVox > 0 || throw(ArgumentError("mask must contain voxels"))
    nSam > 0 && nCha > 0 || throw(ArgumentError("samples and coils must be positive"))
    B = min(block_size, nSam)
    # Cache complete sample blocks only. The limit applies to complex P bytes,
    # with all remaining model, scratch, and solver storage checked separately.
    cache_rows = min(nSam, B*(phase_cache_bytes ÷ (sizeof(Complex{T})*nVox*B)))
    # The reused adjoint keeps 32 complex coil accumulators per thread. Its
    # Float64 specialization exceeds the register budget at 512 threads.
    effective_adjoint_threads = T === Float64 ? min(adjoint_threads, 128) : adjoint_threads
    if backend in (:cached, :cached_full) || cache_rows > 0
        # Include the complete complex P, CSM/Z/Q, image buffers, application
        # workspace, real phase block, model arrays, and an allowance for CG.
        # The remaining 20% is reserved for libraries and allocator overhead.
        workspace_rows = backend === :cached_full ? nSam : B
        phase_elements = backend in (:cached, :cached_full) ? nSam*nVox : (B+cache_rows)*nVox
        complex_elements = phase_elements + 3*nVox*nCha + 2*nVox + workspace_rows*nCha
        real_elements = nTerm*(nVox+nSam) + nSam + nVox
        phase_backend === :gemm && (real_elements += B*nVox)
        solver_elements = 8*length(mask) + 3*nSam*nCha
        estimate = sizeof(Complex{T})*(complex_elements+solver_elements) +
            sizeof(T)*real_elements + sizeof(Int32)*nVox
        estimate < 0.8 * CUDA.free_memory() || throw(ArgumentError(
            "estimated complex phase storage plus workspaces needs $estimate bytes; insufficient GPU memory"))
    end
    prepared = prep_kspha(kspha, k_nominal, nTerm; recon_terms)
    bf = basisfunc_spha(grid.x[indices], grid.y[indices], grid.z[indices], collect(1:nTerm))
    nPad = backend === :direct ? 32*cld(nCha, 32) : nCha
    sensitivities = zeros(Complex{T}, nVox, nPad)
    sensitivities[:, 1:nCha] .= reshape(csm, :, nCha)[indices, :]
    state = ExplicitGPUState{T}(
        backend, phase_backend, gemm_backend, nSam, nVox, nCha, B, forward_threads, effective_adjoint_threads,
        CuArray(Int32.(indices)), CuArray(sensitivities), CuArray(bf), CuArray(prepared),
        CuArray(times), CuArray(vec(fieldmap)[indices]), CUDA.zeros(Complex{T}, nVox),
        CUDA.zeros(Complex{T}, nVox), CUDA.zeros(Complex{T}, backend===:direct ? (nSam,nPad) : (0,0)),
        CUDA.zeros(Complex{T}, backend===:direct ? (0,0) : (nVox,nCha)),
        CUDA.zeros(Complex{T}, backend===:direct ? (0,0) : (nVox,nCha)),
        CUDA.zeros(Complex{T}, backend===:direct ? (0,0) : (backend in (:cached,:cached_full) ? nSam : B,nVox)),
        CUDA.zeros(Complex{T}, cache_rows>0 ? (cache_rows,nVox) : (0,0)),
        CUDA.zeros(T, phase_backend===:gemm && backend!==:direct ? (B,nVox) : (0,0)),
        CUDA.zeros(Complex{T}, backend===:direct ? (0,0) : (backend===:cached_full ? nSam : B,nCha)), false)
    initialize_rows = backend in (:cached, :cached_full) ? nSam : cache_rows
    if initialize_rows > 0
        destination = backend in (:cached, :cached_full) ? state.P : state.phase_cache
        # Write each block to its final cache location without a second complex block.
        for first_sample in 1:B:initialize_rows
            rows = first_sample:min(first_sample+B-1, initialize_rows)
            if phase_backend === :gemm
                explicit_gemm!(state, 'T', 'T', one(T), @view(state.kspha[:, rows]),
                    state.bf, zero(T), @view(state.phase[1:length(rows), :]))
            end
            @cuda threads=256 blocks=cld(length(rows)*nVox,256) explicit_phase_kernel!(
                @view(destination[rows, :]), state.phase, state.bf, state.kspha,
                state.times, state.fieldmap, first_sample, length(rows), phase_backend===:gemm)
        end
    end
    forward = (out, x) -> explicit_gpu_forward!(out, state, x)
    backward = (out, y) -> explicit_gpu_adjoint!(out, state, y)
    return ExplicitGPUOp{T,typeof(forward),typeof(backward)}(
        nSam*nCha, prod(grid.matrixSize), false, false, forward, nothing, backward,
        0, 0, 0, CUDA.zeros(Complex{T},0), CUDA.zeros(Complex{T},0), state)
end

function Base.adjoint(op::ExplicitGPUOp{T}) where {T}
    return LinearOperator{Complex{T}}(op.ncol, op.nrow, false, false,
        op.ctprod!, nothing, op.prod!; S=CuVector{Complex{T}})
end

function LinearOperatorCollection.normalOperator(op::ExplicitGPUOp{T}, weights=nothing; kwargs...) where {T}
    activate_explicit_gpu!(op.state)
    # normalOperator(W ∘ A) supplies WᴴW, which must never be squared here.
    raw = weights isa LinearOperatorCollection.WeightingOp ? weights.weights : weights
    D = raw === nothing ? ones(T, op.state.nSam*op.state.nCha) : vec(Array(raw))
    if length(D) == op.state.nSam
        D = repeat(D, op.state.nCha)
    end
    length(D) == op.nrow || throw(DimensionMismatch("normal weights"))
    all(isfinite, D) || throw(ArgumentError("normal weights must be finite"))
    maximum(abs, imag.(D)) <= 10eps(T)*max(one(T), maximum(abs,D)) ||
        throw(ArgumentError("normal weights must already be real nonnegative WᴴW"))
    minimum(real.(D)) >= zero(T) || throw(ArgumentError("normal weights must be nonnegative"))
    diagonal = CuArray(reshape(T.(real.(D)), op.state.nSam, op.state.nCha))
    normal = (out, x) -> explicit_gpu_normal!(out, op.state, diagonal, x)
    return LinearOperator{Complex{T}}(op.ncol, op.ncol, false, true,
        normal, nothing, normal; S=CuVector{Complex{T}})
end

"""
    explicit_gpu_normal(op; weights=nothing)

Create the data normal operator `Aᴴ * D * A` for a resident
`HighOrderKernelOp` (`backend=:direct`, `:blocked`, `:cached`, or `:cached_full`), with `D = abs2.(weights)`. Raw sample weights may have one
entry per sample or per sample and coil; `nothing` means unit weights.

In contrast, `normalOperator(op, D)` accepts an already squared, real
nonnegative diagonal. `normalOperator(W ∘ op)` also supplies that squared
diagonal automatically. Regularization is applied separately by the solver.
The result shares the operator's mutable workspaces and lifetime.
"""
function explicit_gpu_normal(op::ExplicitGPUOp; weights=nothing)
    return LinearOperatorCollection.normalOperator(op, weights===nothing ? nothing : abs2.(weights))
end

function Base.close(op::ExplicitGPUOp)
    state = op.state
    state.released && return nothing
    CUDA.device!(CUDA.device(state.x))
    CUDA.synchronize()
    for name in fieldnames(typeof(state))
        value = getfield(state, name)
        value isa CuArray && CUDA.unsafe_free!(value)
    end
    state.released = true
    return nothing
end

function explicit_gpu_normal(op::HighOrderKernelOp{T,F1,F2,S,O}; weights=nothing) where {T,F1,F2,S,O<:HOOp}
    return LinearOperatorCollection.normalOperator(op, weights === nothing ? nothing : abs2.(weights))
end
