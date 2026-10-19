"""
Independent spatial factors represented by the original rSVD Q/Z matrices.

Only a fixed, evenly spaced subset of readouts is materialized on the host.
`size` describes the logical factor bank; indexing/materializing that entire
bank is intentionally unsupported. Consumers obtain one block through the
bounded `PerReadoutCache`. Q is saved before temporal phase/scale corrections.
"""
struct RegeneratedSpatialBasis{T<:AbstractFloat} <: AbstractMatrix{Complex{T}}
    Q          :: Array{Complex{T},3}
    Z          :: Array{Complex{T},3}
    times      :: Matrix{T}
    fieldmap   :: Vector{T}
    bf         :: Matrix{T}
    kspha_t    :: Array{T,3}
    host_basis :: Matrix{Complex{T}}
    host_slot  :: Vector{Int}
    host_budget_bytes :: Int
end

function RegeneratedSpatialBasis(times::Matrix{T}, fieldmap::Vector{T},
    bf::Matrix{T}, kspha::Array{T,3}, rank::Int, total_rank::Int,
    host_budget_bytes::Int) where {T<:AbstractFloat}
    nSam, nDyn = size(times)
    bytes_per_readout = Base.checked_mul(length(fieldmap), Base.checked_mul(rank, sizeof(Complex{T})))
    nCached = min(nDyn, host_budget_bytes ÷ bytes_per_readout)
    slots = zeros(Int, nDyn)
    # Fixed residency avoids cyclic LRU thrashing between CG iterations.
    for j in 1:nCached
        slots[1 + (j - 1) * nDyn ÷ nCached] = j
    end
    return RegeneratedSpatialBasis(
        Array{Complex{T}}(undef, nSam, total_rank, nDyn),
        Array{Complex{T}}(undef, total_rank, rank, nDyn),
        times, fieldmap, bf, permutedims(kspha, (2, 1, 3)),
        Matrix{Complex{T}}(undef, length(fieldmap), rank * nCached), slots,
        host_budget_bytes,
    )
end

Base.size(source::RegeneratedSpatialBasis) =
    (length(source.fieldmap), size(source.Z, 2) * size(source.Q, 3))
Base.size(source::RegeneratedSpatialBasis, d::Integer) = d <= 2 ? size(source)[d] : 1
Base.getindex(::RegeneratedSpatialBasis, ::Int, ::Int) =
    throw(ArgumentError("regenerated spatial factors must be accessed through a readout cache"))
Base.Array(::RegeneratedSpatialBasis) =
    throw(ArgumentError("materializing the full regenerated factor bank is unsupported"))
function Base.show(io::IO, ::MIME"text/plain", source::RegeneratedSpatialBasis)
    print(io, "RegeneratedSpatialBasis", size(source), " with ",
        count(!iszero, source.host_slot), " host-cached readouts")
end

function regenerated_basis_stats(source::RegeneratedSpatialBasis)
    return (; logical_bytes=prod(size(source)) * sizeof(eltype(source)),
        qz_bytes=sizeof(source.Q) + sizeof(source.Z),
        model_bytes=sizeof(source.times) + sizeof(source.fieldmap) + sizeof(source.bf) + sizeof(source.kspha_t),
        host_cache_bytes=sizeof(source.host_basis), host_budget_bytes=source.host_budget_bytes,
        host_cached_readouts=count(!iszero, source.host_slot))
end


"""
GPU producer for `(H'Q)Z`, with private metadata, intermediate, and library state.
Destination aliases do not own memory: the enclosing cache owns their storage.
Ready/consumed events, rather than managed-array stream switches, protect them.
"""
mutable struct SpatialFactorGenerator{T<:AbstractFloat}
    worker       :: DistributedGPUWorker
    Q            :: CuArray{Complex{T},3}
    Z            :: CuArray{Complex{T},3}
    times        :: CuMatrix{T}
    fieldmap     :: CuVector{T}
    bf           :: CuMatrix{T}
    kspha_t      :: CuArray{T,3}
    intermediate :: CuMatrix{Complex{T}}
    destinations :: Vector{CuMatrix{Complex{T}}}
    released     :: Bool
end

function SpatialFactorGenerator(source::RegeneratedSpatialBasis{T}, destinations,
    stream::CuStream) where {T}
    # Capture pointers on the consumer stream, before producer aliases exist.
    pointers = pointer.(destinations)
    worker = DistributedGPUWorker(CUDA.deviceid(CUDA.device()))
    result = Vector{Any}(undef, 1)
    try
        run_on_workers!(result, [worker]; phase=:factor_generator_setup) do _
            CUDA.stream!(stream) do
                aliases = [unsafe_wrap(CuArray, p, size(d); own=false)
                    for (p, d) in zip(pointers, destinations)]
                SpatialFactorGenerator{T}(worker, CuArray(source.Q), CuArray(source.Z),
                    CuArray(source.times), CuArray(source.fieldmap), CuArray(source.bf),
                    CuArray(source.kspha_t),
                    CuMatrix{Complex{T}}(undef, size(source, 1), size(source.Q, 2)),
                    aliases, false)
            end
        end
        # All consumers/aliases may use the initialization after this point.
        CUDA.synchronize(stream)
        return result[1]::SpatialFactorGenerator{T}
    catch
        shutdown_distributed_workers!([worker])
        rethrow()
    end
end

function submit_generated_factor!(generator::SpatialFactorGenerator, slot::Int, dyn::Int,
    stream::CuStream, consumed, used::Bool, started::CuEvent, done::CuEvent, ready::CuEvent)
    generator.released && error("spatial factor generator was released")
    acknowledgements = Vector{Nothing}(undef, 1)
    # Wait only for CPU submission/recording of the ready event, not GPU completion.
    run_on_workers!(acknowledgements, [generator.worker]; phase=:factor_generation_submit) do _
        CUDA.stream!(stream) do
            used && CUDA.wait(consumed, stream)
            CUDA.record(started, stream)
            @views run_kernel_rsvd_adjoint!(generator.intermediate, generator.Q[:, :, dyn],
                generator.times[:, dyn], generator.fieldmap, generator.bf,
                generator.kspha_t[:, :, dyn]; threads=256, kspha_transposed=true)
            @views mul!(generator.destinations[slot], generator.intermediate, generator.Z[:, :, dyn])
            CUDA.record(done, stream)
            CUDA.record(ready, stream)
        end
        nothing
    end
    return nothing
end

function Base.close(generator::SpatialFactorGenerator)
    generator.released && return nothing
    try
        acknowledgements = Vector{Nothing}(undef, 1)
        run_on_workers!(acknowledgements, [generator.worker]; phase=:factor_generator_release) do _
            for array in (generator.Q, generator.Z, generator.times, generator.fieldmap,
                generator.bf, generator.kspha_t, generator.intermediate)
                CUDA.unsafe_free!(array)
            end
            foreach(CUDA.unsafe_free!, generator.destinations)
            nothing
        end
    finally
        generator.released = true
        shutdown_distributed_workers!([generator.worker])
    end
    return nothing
end
