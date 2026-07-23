"""
    recon_HOOp(
        HOOp::HOOp{Complex{T}},
        Data::AbstractArray{Complex{T},2},
        weight::AbstractVector{Complex{T}},
        recParams::Dict;
        release_backend=true,
    ) where T<:AbstractFloat

Reconstruct an image with a high-order encoding operator.

For a `HighOrderLowRankOp` using `normal_distribution=:channel`,
`normalOperator` lazily creates persistent multi-GPU workers and one NFFT plan
per participating GPU. By default, those reconstruction-only resources are
released in a `finally` block after `solve!`, including when reconstruction
throws an exception.

Before the normal backend is constructed, GPU execution is synchronized and a
single full Julia GC is performed. This collects an older `HighOrderLowRankOp`
that may have become unreachable after replacing a global `HOOp` variable,
without trimming CUDA.jl's reusable memory pools.

Set `release_backend=false` only when the same `HighOrderLowRankOp`, weights,
and multi-GPU normal backend will be reused for another reconstruction. The
caller must then invoke `close(HOOp)` when the final reconstruction finishes.
"""
function recon_HOOp(
    HOOp      :: HOOp{Complex{T}},
    Data      :: AbstractArray{Complex{T},2},
    weight    :: AbstractVector{Complex{T}},
    recParams :: Dict;
    release_backend :: Bool = true,
) where T<:AbstractFloat
    try
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
        W = WeightingOp(Complex{T}; weights=weight, rep=nCha)
        E = ∘(W, HOOp)
        EᴴE = normalOperator(E)
        solver = createLinearSolver(
            recParams[:solver],
            E;
            AHA=EᴴE,
            reg=recParams[:reg],
            recoParams...,
        )
        x = solve!(solver, Data)
        return reshape(x, recParams[:reconSize])
    finally
        if release_backend && HOOp isa HighOrderLowRankOp
            release_highorder_normal_backend!(HOOp)
        end
    end
end
