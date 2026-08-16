include(joinpath(@__DIR__, "DemoCommon.jl"))

using LinearAlgebra
using Printf
using Statistics


# This script benchmarks complete reconstructions after one warm-up run. The
# operator constructor is timed separately because HighOrderLowRankOp has a
# non-negligible one-time rSVD setup cost. The first reconstruction is also
# reported separately because it includes compilation and, for the distributed
# low-rank operator, construction of the lazy normal-operator backend.
benchmark_samples = 3
benchmark_single_gpu = true
benchmark_multi_gpu = true
nVirtualCoils = 16
maximum_benchmark_gpus = 8

benchmark_samples > 0 || throw(ArgumentError("benchmark_samples must be positive"))
CUDA.functional() || error("CUDA is not functional")

available_gpu_ids = Int[CUDA.deviceid(device) for device in CUDA.devices()]
isempty(available_gpu_ids) && error("No CUDA devices are visible")

primary_gpu = first(available_gpu_ids)
multi_gpu_ids = available_gpu_ids[1:min(maximum_benchmark_gpus, length(available_gpu_ids))]
CUDA.device!(primary_gpu)

if benchmark_multi_gpu && length(multi_gpu_ids) < 2
    @warn "Multi-GPU benchmark cases are disabled because fewer than two GPUs are visible" available_gpu_ids
    benchmark_multi_gpu = false
end


# --------------------------------------------------------------------------
# Shared measured-data preprocessing
# --------------------------------------------------------------------------
recon = "0111"
kspha_measured = T.(ksphaMeasured')
times_measured = T.(datatime)
fieldmap = T.(b0)
mask_recon = Bool.(mask)
csm_physical = Complex{T}.(csm)
weight_host = Complex{T}.(weightMeasured)

kdata_physical = Complex{T}.(data["kdata"])
kdata_physical ./= exp.(Complex{T}(0, T(2π)) .* T.(k0_ecc))'
kdata_physical .*= exp.(Complex{T}(0, T(-2π)) .* T.(ksphaMeasured[:, 1]))

kdata_compressed, csm_compressed, coil_transform = compress_coils(
    kdata_physical,
    csm_physical;
    data_coil_dim=2,
    csm_coil_dim=3,
    n_virtual_coils=nVirtualCoils,
)

@info(
    "Coil compression",
    physical_coils=size(coil_transform, 1),
    virtual_coils=size(coil_transform, 2),
    retained_energy=coil_transform.retained_energy,
)

# HighOrderOp and HighOrderLowRankOp advertise CuArray solver storage, whereas
# HighOrderKernelOp intentionally uses CPU solver vectors around its GPU
# workspaces. Convert inputs once, outside every timed reconstruction.
kdata_physical_gpu = CuArray(kdata_physical)
kdata_compressed_gpu = CuArray(kdata_compressed)
weight_gpu = CuArray(weight_host)


function synchronize_devices!(gpu_ids::AbstractVector{Int})
    for gpu_id in gpu_ids
        CUDA.device!(gpu_id)
        CUDA.device_synchronize(; blocking=true)
    end
    CUDA.device!(first(gpu_ids))
    return nothing
end


function benchmark_reconstruction_case(
    label::String,
    variant::Symbol,
    build_operator,
    reconstruction_data,
    reconstruction_weight,
    gpu_ids::Vector{Int},
    rec_params::Dict;
    samples::Int,
)
    @info "Benchmark case" label gpu_ids samples

    # Collect objects from the preceding case, but retain CUDA.jl's reusable
    # memory pools. CUDA.reclaim() is deliberately not part of timed samples.
    GC.gc(true)
    CUDA.device!(first(gpu_ids))

    operator = nothing
    image = nothing
    result = nothing

    try
        setup_stats = @timed begin
            operator = build_operator()
            synchronize_devices!(gpu_ids)
        end

        first_reconstruction_stats = @timed begin
            image = recon_HOOp(
                operator,
                reconstruction_data,
                reconstruction_weight,
                rec_params;
                release_backend=false,
            )
            synchronize_devices!(gpu_ids)
        end

        elapsed = Vector{Float64}(undef, samples)
        allocated = Vector{Int}(undef, samples)
        gc_time = Vector{Float64}(undef, samples)

        for sample = 1:samples
            GC.gc(true)
            stats = @timed begin
                image = recon_HOOp(
                    operator,
                    reconstruction_data,
                    reconstruction_weight,
                    rec_params;
                    release_backend=false,
                )
                synchronize_devices!(gpu_ids)
            end
            elapsed[sample] = stats.time
            allocated[sample] = stats.bytes
            gc_time[sample] = stats.gctime
        end

        result = (
            label=label,
            variant=variant,
            gpu_ids=copy(gpu_ids),
            setup_seconds=setup_stats.time,
            first_reconstruction_seconds=first_reconstruction_stats.time,
            median_seconds=median(elapsed),
            minimum_seconds=minimum(elapsed),
            maximum_seconds=maximum(elapsed),
            median_cpu_allocated_bytes=median(allocated),
            median_gc_seconds=median(gc_time),
            samples=copy(elapsed),
            image=copy(image),
        )
    finally
        if operator isa HighOrderLowRankOp
            close(operator)
        end
        operator = nothing
        GC.gc(false)
        CUDA.device!(primary_gpu)
    end

    return result
end


# --------------------------------------------------------------------------
# Benchmark cases
# --------------------------------------------------------------------------
function build_highorder_operator(csm_current)
    return HighOrderOp(
        gridding,
        kspha_measured,
        times_measured;
        recon_terms=recon,
        nBlock=nBlock,
        csm=csm_current,
        fieldmap=fieldmap,
        mask=mask_recon,
        arrayType=CuArray,
        verbose=verbose,
    )
end


function build_kernel_operator(csm_current, gpu_ids)
    return HighOrderKernelOp(
        gridding,
        kspha_measured,
        times_measured;
        recon_terms=recon,
        csm=csm_current,
        fieldmap=fieldmap,
        mask=mask_recon,
        arrayType=CuArray,
        gpus=gpu_ids,
        verbose=verbose,
    )
end


function build_lowrank_operator(csm_current, gpu_ids; distributed::Bool)
    return HighOrderLowRankOp(
        gridding,
        kspha_measured,
        times_measured;
        recon_terms=recon,
        csm=csm_current,
        fieldmap=fieldmap,
        mask=mask_recon,
        arrayType=CuArray,
        gpus=gpu_ids,
        L_rank=25,
        rsvd_seed=1234,
        rsvd_chunk=4096,
        rsvd_oversample=5,
        rsvd_finalize=:gram,
        rsvd_backend=:kernel,
        rsvd_distribution=distributed ? :voxel : :single,
        shared_rank_max=32,
        shared_basis_tol=T(1e-2),
        normal_distribution=distributed ? :channel : :single,
        verbose=verbose,
    )
end


datasets = (
    (
        label="physical coils",
        variant=:physical,
        csm=csm_physical,
        data_host=kdata_physical,
        data_gpu=kdata_physical_gpu,
    ),
    (
        label="compressed coils",
        variant=:compressed,
        csm=csm_compressed,
        data_host=kdata_compressed,
        data_gpu=kdata_compressed_gpu,
    ),
)

cases = Any[]
for dataset in datasets
    if benchmark_single_gpu
        single_gpu_ids = [primary_gpu]
        push!(cases, (
            label="HighOrderOp / $(dataset.label) / 1 GPU",
            variant=dataset.variant,
            gpu_ids=single_gpu_ids,
            data=dataset.data_gpu,
            weight=weight_gpu,
            build=let csm_current=dataset.csm
                () -> build_highorder_operator(csm_current)
            end,
        ))
        push!(cases, (
            label="HighOrderKernelOp / $(dataset.label) / 1 GPU",
            variant=dataset.variant,
            gpu_ids=single_gpu_ids,
            data=dataset.data_host,
            weight=weight_host,
            build=let csm_current=dataset.csm, gpu_ids=single_gpu_ids
                () -> build_kernel_operator(csm_current, gpu_ids)
            end,
        ))
        push!(cases, (
            label="HighOrderLowRankOp / $(dataset.label) / 1 GPU",
            variant=dataset.variant,
            gpu_ids=single_gpu_ids,
            data=dataset.data_gpu,
            weight=weight_gpu,
            build=let csm_current=dataset.csm, gpu_ids=single_gpu_ids
                () -> build_lowrank_operator(csm_current, gpu_ids; distributed=false)
            end,
        ))
    end

    if benchmark_multi_gpu
        gpu_ids = copy(multi_gpu_ids)
        push!(cases, (
            label="HighOrderKernelOp / $(dataset.label) / $(length(gpu_ids)) GPUs",
            variant=dataset.variant,
            gpu_ids=gpu_ids,
            data=dataset.data_host,
            weight=weight_host,
            build=let csm_current=dataset.csm, gpu_ids=gpu_ids
                () -> build_kernel_operator(csm_current, gpu_ids)
            end,
        ))
        push!(cases, (
            label="HighOrderLowRankOp / $(dataset.label) / $(length(gpu_ids)) GPUs",
            variant=dataset.variant,
            gpu_ids=gpu_ids,
            data=dataset.data_gpu,
            weight=weight_gpu,
            build=let csm_current=dataset.csm, gpu_ids=gpu_ids
                () -> build_lowrank_operator(csm_current, gpu_ids; distributed=true)
            end,
        ))
    end
end


benchmark_results = Any[]
for case in cases
    push!(
        benchmark_results,
        benchmark_reconstruction_case(
            case.label,
            case.variant,
            case.build,
            case.data,
            case.weight,
            case.gpu_ids,
            recParams;
            samples=benchmark_samples,
        ),
    )
end


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
reference_images = Dict{Symbol,Array}()

println()
println("Reconstruction benchmark summary")
@printf(
    "%-58s %9s %9s %9s %9s %11s %10s\n",
    "case",
    "setup/s",
    "first/s",
    "median/s",
    "min/s",
    "CPU MiB",
    "rel.error",
)

for result in benchmark_results
    reference = get!(reference_images, result.variant, result.image)
    relative_error = norm(result.image - reference) / max(norm(reference), eps(T))
    @printf(
        "%-58s %9.3f %9.3f %9.3f %9.3f %11.1f %10.3e\n",
        result.label,
        result.setup_seconds,
        result.first_reconstruction_seconds,
        result.median_seconds,
        result.minimum_seconds,
        result.median_cpu_allocated_bytes / 2.0^20,
        relative_error,
    )
end

CUDA.device!(primary_gpu)
