"""
    recon_HOOp(HOOp::HOOp{Complex{T}},
        Data::AbstractArray{Complex{T}, 2},
        weight::AbstractVector{Complex{T}},
        recParams::Dict;
        release_backend::Bool = true,
        encoding_weights::Union{Nothing,AbstractVector} = nothing) where T <: AbstractFloat

Reconstruct an image with a high-order encoding operator.

`Data` has one row per sample/dynamic point and one column per receive coil.
For dynamic low-rank encoding, reshape `(nSam, nDyn, nCha)` data to
`(nSam * nDyn, nCha)` without permuting axes. `weight` contains one raw
complex sample weight per row, repeated identically for all coils. Match the
floating-point precision of the operator, data, and weights.

The wrapper weights both the data and encoding operator: with `A = HOOp`
and diagonal `W` formed from the repeated raw weights, it passes `W * A` and
`W * vec(Data)` to the solver. The data normal is `Aᴴ Wᴴ W A`, so a raw
weight contributes `abs2(weight)` to the normal. Do not square the weights
before passing them here. [`samplingDensity`](@ref) already returns
square-root density weights for this convention.

`encoding_weights` optionally supplies one additional real or complex row
factor of precision `T`, on the same backend as `weight`. It modifies the
encoding, not the measured data: the solver receives
`diag(weight .* encoding_weights) * HOOp` and `diag(weight) * vec(Data)`.
The default `nothing` preserves the point-voxel model. For a finite box
voxel, use `HighOrderMRI.voxel_aperture` with the same prepared first-order
trajectory as the operator. Do not multiply the data by that aperture.

`recParams` must explicitly contain `:solver`, `:reg`, and `:reconSize`.
Remaining solver settings are merged with MRIReco's `defaultRecoParams()`;
select a solver and regularizer that are compatible. The returned image is
a CPU `Array` reshaped to `recParams[:reconSize]`.

For CUDA low-rank reconstruction and resident explicit backends, pass data
and weights as CUDA arrays on the primary GPU to match the solver's CUDA
storage. This wrapper does not move those inputs to the GPU automatically.

For `HighOrderLowRankOp`, channel/readout backends initialize on a normal
request; rank execution can initialize on a forward, adjoint, or normal
request. By default, the wrapper releases the owned distributed backend in a
`finally` block, including when reconstruction throws an exception.

Set `release_backend=false` to retain that backend for another reconstruction
with the same operator and weights. Call `close(HOOp)` after its final use.
Backend release retains the parent's factor arrays and per-readout source
queue; `close(HOOp)` also releases that queue. Later use of the parent can
recreate these resources lazily. To change normal weights while retaining
a distributed parent (`:channel`, `:readout`, or `:rank`), release the old
backend first and obtain a fresh normal view. The `:single` path has no such
owned distributed backend.

Resident explicit operators require a separate `close(HOOp)` after their
final use; this keyword handles only low-rank distributed backends.

See [Reconstruction workflow](/guide/reconstruction) for solver configuration
and [Multi-GPU execution](/guide/multi-gpu) for execution strategies and cleanup.
"""
function recon_HOOp(
    HOOp      :: HOOp{Complex{T}},
    Data      :: AbstractArray{Complex{T},2},
    weight    :: AbstractVector{Complex{T}},
    recParams :: Dict;
    release_backend :: Bool = true,
    encoding_weights :: Union{Nothing,AbstractVector} = nothing,
) where T<:AbstractFloat
    try
        if !isnothing(encoding_weights)
            length(encoding_weights) == size(Data, 1) == length(weight) ||
                throw(DimensionMismatch("Encoding row factors must match data rows and weights"))
            eltype(encoding_weights) in (T, Complex{T}) ||
                throw(ArgumentError("Encoding row factors must match operator precision"))
            all(isfinite, encoding_weights) || throw(ArgumentError("Encoding row factors must be finite"))
        end
        if HOOp isa HighOrderLowRankOp && HOOp.q isa CuArray
            primary_gpu = first(HOOp.normal_backend.gpus)
            CUDA.device!(primary_gpu)
            CUDA.device_synchronize(; blocking=true)

            # A previous large operator can become unreachable only after a
            # top-level `HOOp = HighOrderLowRankOp(...)` assignment completes.
            # Collect it before the channel-distributed normal backend
            # replicates q, basis, csm, and NFFT workspaces across GPUs.
            GC.gc(true)
            CUDA.device!(primary_gpu)
        end

        recoParams = merge(defaultRecoParams(), recParams)

        _, nCha = size(Data)
        Data = vec(Data) .* repeat(weight, nCha)
        operator_weight = isnothing(encoding_weights) ? weight : weight .* encoding_weights
        W = WeightingOp(Complex{T}; weights=operator_weight, rep=nCha)
        E = ∘(W, HOOp)
        EᴴE = normalOperator(E)
        solver = createLinearSolver(recParams[:solver], E; AHA=EᴴE, reg=recParams[:reg], recoParams...,)
        x = solve!(solver, Data)
        return Array(reshape(x, recParams[:reconSize]))
    finally
        if release_backend && HOOp isa HighOrderLowRankOp
            release_highorder_normal_backend!(HOOp)
        end
    end
end
