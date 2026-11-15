# Experimental execution for the pinned NonuniformFFTs 0.9.6 plan layout.
# Native resources are supplied explicitly; normal_options=(;) never loads them.
const CD = parentmodule(CUDA.CuStream)

mutable struct PrunedFFTPlan
    library::Ptr{Cvoid}
    handle::Ptr{Cvoid}
    execute::Ptr{Cvoid}
    destroy::Ptr{Cvoid}
    context::CUDA.CuContext
    shape::NTuple{4,Int}
end

function PrunedFFTPlan(buffer, path, left, right)
    library = Libdl.dlopen(path, Libdl.RTLD_NOW | Libdl.RTLD_LOCAL)
    handle = C_NULL
    destroy = C_NULL
    try
        create = Libdl.dlsym(library, :joint_vkfft_create)
        execute = Libdl.dlsym(library, :joint_vkfft_execute)
        destroy = Libdl.dlsym(library, :joint_vkfft_destroy)
        result = Ref{Cint}()
        nx, ny, nz, batch = size(buffer)
        handle = ccall(create, Ptr{Cvoid},
            (UInt64,UInt64,UInt64,UInt64,CUDA.CuPtr{ComplexF32},CUDA.cuFFT.cudaStream_t,
             Cint,Cint,Cint,Ptr{UInt64},Ptr{UInt64},Ref{Cint}),
            nx,ny,nz,batch,buffer,CUDA.stream(),256,0,1,UInt64[left...],UInt64[right...],result)
        result[] == 0 && handle != C_NULL || error("Experimental VkFFT plan failed: $(result[])")
        plan = PrunedFFTPlan(library,handle,execute,destroy,CUDA.context(),size(buffer))
        finalizer(close,plan)
        return plan
    catch
        handle == C_NULL || destroy == C_NULL || ccall(destroy,Cvoid,(Ptr{Cvoid},),handle)
        Libdl.dlclose(library)
        rethrow()
    end
end

function Base.close(plan::PrunedFFTPlan)
    plan.handle == C_NULL && return nothing
    CUDA.context!(plan.context) do
        CUDA.device_synchronize(; blocking=true)
        ccall(plan.destroy,Cvoid,(Ptr{Cvoid},),plan.handle)
    end
    plan.handle = C_NULL
    Libdl.dlclose(plan.library)
    plan.library = C_NULL
    return nothing
end

struct PrunedFFTTransform
    plan::PrunedFFTPlan
    direction::Cint
end

function Base.:*(transform::PrunedFFTTransform,buffer::CuArray{ComplexF32,4})
    plan = transform.plan
    plan.handle != C_NULL || error("Experimental VkFFT plan was closed")
    size(buffer) == plan.shape || throw(DimensionMismatch("experimental FFT buffer shape changed"))
    CUDA.context() == plan.context || throw(ArgumentError("experimental FFT used on the wrong CUDA context"))
    result = GC.@preserve plan ccall(plan.execute,Cint,
        (Ptr{Cvoid},CUDA.CuPtr{ComplexF32},CUDA.cuFFT.cudaStream_t,Cint),
        plan.handle,buffer,CUDA.stream(),transform.direction)
    result == 0 || error("Experimental VkFFT execution failed: $result")
    return buffer
end

mutable struct PrunedHighOrderExecution{M,F,A}
    fft::PrunedFFTPlan
    native_module::M
    gather::F
    spread::F
    metadata::A
    zero_tiles::CuVector{Int32}
    image_to_fft::CuVector{Int32}
    image_to_mask::CuVector{Int32}
    coefficient_cache_bytes::Int
    released::Bool
end

function Base.close(execution::PrunedHighOrderExecution)
    execution.released && return nothing
    execution.released = true
    # The owning worker synchronizes before release. finalize removes the
    # CuModule finalizer too, so its module is not unloaded a second time.
    try
        close(execution.fft)
    finally
        try
            finalize(execution.native_module)
        finally
            CUDA.unsafe_free!(execution.metadata)
            CUDA.unsafe_free!(execution.zero_tiles)
            CUDA.unsafe_free!(execution.image_to_fft)
            CUDA.unsafe_free!(execution.image_to_mask)
        end
    end
    return nothing
end

function pruned_zero_intervals(plan)
    dims = size(first(plan.data.us))
    bounds = map(1:3) do d
        present = falses(dims[d])
        present[Array(plan.index_map[d])] .= true
        gap = findall(!,present)
        isempty(gap) && return (0,0)
        all(diff(gap) .== 1) || throw(ArgumentError("pruned FFT requires one padding interval per axis"))
        return (first(gap)-1,last(gap))
    end
    return Tuple(first.(bounds)),Tuple(last.(bounds))
end

function kernel_pruned_image_maps!(image_to_fft,image_to_mask,index_map,mask_idx,dims,fft_dims)
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    if i <= length(image_to_fft)
        a = mod(i-1,dims[1])+1
        b = mod(div(i-1,dims[1]),dims[2])+1
        c = div(i-1,dims[1]*dims[2])+1
        @inbounds image_to_fft[i] = index_map[1][a]+fft_dims[1]*((index_map[2][b]-1)+fft_dims[2]*(index_map[3][c]-1))
    end
    if i <= length(mask_idx)
        @inbounds image_to_mask[mask_idx[i]] = i
    end
    return nothing
end

function kernel_pruned_coefficients!(out,ids,origins,windows,nt,dims)
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x-1
    if i < length(out)
        entry = div(i,12)+1
        axis,cell = div(mod(i,12),4),mod(i,4)
        @inbounds bits,p = UInt32(origins[entry]),ids[entry]
        a = mod(Int32((bits>>(4axis))&15)+cell-3,dims[axis+1])+1
        @inbounds out[i+1] = 1 <= a <= 6 ? windows[p+nt*(a-1+6axis)] : 0f0
    end
    return nothing
end

function pruned_batched_plan(plan,starts,windows,ids,chunks,mask_idx,grid_size,options,::Val{B}) where {B}
    dims = size(first(plan.data.us))
    nt,nf = size(starts,1),prod(dims)
    nf*B < typemax(Int32) && nt*B < typemax(Int32) || throw(ArgumentError("native pruned indices exceed Int32"))
    18nt <= typemax(Int32) || throw(ArgumentError("native pruned window indices exceed Int32"))
    cache_bytes = 12length(ids)*sizeof(Float32)
    cached = cache_bytes <= get(options,:coefficient_cache_bytes,0)
    !cached || 12length(ids) <= typemax(Int32) || throw(ArgumentError("native pruned coefficient indices exceed Int32"))
    cpu_starts = Array(starts)
    tiles = cld.(dims,4)
    origins = Vector{UInt16}(undef,length(ids))
    for chunk in axes(chunks,1)
        tile = chunks[chunk,3]-1
        origin = (mod(tile,tiles[1])*4+1,mod(div(tile,tiles[1]),tiles[2])*4+1,div(tile,tiles[1]*tiles[2])*4+1)
        for entry in chunks[chunk,1]:chunks[chunk,2]
            p = ids[entry]
            delta = ntuple(3) do d
                v = origin[d]+3-cpu_starts[p,d]
                v = v < -2 ? v+dims[d] : (v > 6 ? v-dims[d] : v)
                -2 <= v <= 6 || throw(ArgumentError("native output origin exceeds its 4-bit encoding"))
                v+2
            end
            origins[entry] = UInt16(delta[1] | (delta[2]<<4) | (delta[3]<<8))
        end
    end
    gpu_origins = CuArray(origins)
    needed = trues(prod(tiles))
    for i in axes(chunks,1)
        chunks[i,4] == 1 && (needed[chunks[i,3]] = false)
    end
    metadata = if cached
        coefficients = CuArray{Float32}(undef,12length(ids))
        @cuda threads=256 blocks=cld(length(coefficients),256) kernel_pruned_coefficients!(
            coefficients,CuArray(ids),gpu_origins,windows,nt,dims)
        coefficients
    else
        gpu_origins
    end
    image_to_fft = CUDA.zeros(Int32,prod(grid_size))
    image_to_mask = CUDA.zeros(Int32,prod(grid_size))
    @cuda threads=256 blocks=cld(prod(grid_size),256) kernel_pruned_image_maps!(
        image_to_fft,image_to_mask,plan.index_map,mask_idx,grid_size,dims)
    left,right = pruned_zero_intervals(plan)
    fft = PrunedFFTPlan(plan.data.buffer,options.vkfft_library,left,right)
    native_module = nothing
    try
        native_module = CD.CuModule(read(options.native_ptx,String))
        gather = CD.CuFunction(native_module,"native_gather_u9_$B")
        spread = CD.CuFunction(native_module,cached ? "native_spread_coeffs_$B" : "native_spread_$B")
        execution = PrunedHighOrderExecution(fft,native_module,gather,spread,metadata,
            CuArray(Int32.(findall(needed))),image_to_fft,image_to_mask,cached ? cache_bytes : 0,false)
        data = BatchedFFTData{3,B,typeof(plan.data.ks),typeof(plan.data.us),typeof(plan.data.buffer),PrunedFFTTransform,PrunedFFTTransform}(
            plan.data.ks,plan.data.us,plan.data.buffer,PrunedFFTTransform(fft,-1),PrunedFFTTransform(fft,1))
        transformed = NonuniformFFTs.PlanNUFFT(plan.kernels,plan.backend,plan.kernel_evalmode,plan.σ,
            plan.points_ref,data,plan.blocks,plan.fftshift,plan.index_map,plan.timer,plan.synchronise,
            plan.point_transform_fold,plan.cpu_use_atomics)
        return transformed,execution
    catch
        close(fft)
        native_module === nothing || finalize(native_module)
        rethrow()
    end
end

function kernel_pruned_pad!(buffer,x,basis,r,csm,c0,image_to_fft,image_to_mask,beta,nf,C,::Val{B}) where {B}
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    if i <= length(image_to_fft)
        @inbounds j,v = image_to_fft[i],image_to_mask[i]
        for coil in 1:B
            @inbounds buffer[j+nf*(coil-1)] = v > 0 && coil+c0 <= C ?
                x[v]*conj(basis[v,r])*csm[v,coil+c0]*beta[v] : zero(ComplexF32)
        end
    end
    return nothing
end

function pad_batch!(workspace::BatchedHighOrderNormalWorkspace{B},x,r,c0,execution::PrunedHighOrderExecution) where {B}
    execution.released && error("Experimental pruned execution was released")
    nf = length(workspace.plan.data.buffer) ÷ B
    @cuda threads=256 blocks=cld(length(execution.image_to_fft),256) kernel_pruned_pad!(
        workspace.plan.data.buffer,x,workspace.basis,r,workspace.csm,c0,
        execution.image_to_fft,execution.image_to_mask,workspace.deconvolution,nf,size(workspace.csm,2),Val(B))
    return nothing
end

function kernel_native_accumulate!(channel,samples,q,r,nt)
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    if i <= length(channel)
        t = mod(i-1,nt)+1
        @inbounds channel[i] += samples[i]*q[t,r]
    end
    return nothing
end

function gather_batch!(workspace::BatchedHighOrderNormalWorkspace{B},r,execution::PrunedHighOrderExecution) where {B}
    p = workspace.plan
    nx,ny,nz = size(first(p.data.us))
    nt,nf = size(workspace.q,1),nx*ny*nz
    types = Tuple{CD.CuPtr{ComplexF32},CD.CuPtr{ComplexF32},CD.CuPtr{Int32},CD.CuPtr{Float32},CD.CuPtr{Int32},
        Int32,Int32,Int32,Int32,Int32,Float32}
    prefactor = Float32(prod(map(NonuniformFFTs.Kernels.gridstep,p.kernels)))
    CD.cudacall(execution.gather,types,workspace.samples,p.data.buffer,workspace.starts,
        workspace.windows,workspace.permutation,Int32(nx),Int32(ny),Int32(nz),Int32(nt),Int32(nf),prefactor;
        threads=128,blocks=cld(nt*8,128),stream=CUDA.stream())
    @cuda threads=256 blocks=cld(nt*B,256) kernel_native_accumulate!(workspace.channel,workspace.samples,workspace.q,r,nt)
    return nothing
end

function kernel_pruned_clear!(buffer,tiles,dims,nf,::Val{B}) where {B}
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x-1
    if i < length(tiles)*64*B
        cell,coil,entry = mod(i,64),mod(div(i,64),B),div(i,64B)+1
        @inbounds tile = tiles[entry]-1
        tx,ty = cld(dims[1],4),cld(dims[2],4)
        x = mod(tile,tx)*4+mod(cell,4)
        y = mod(div(tile,tx),ty)*4+mod(div(cell,4),4)
        z = div(tile,tx*ty)*4+div(cell,16)
        if x < dims[1] && y < dims[2] && z < dims[3]
            @inbounds buffer[x+dims[1]*(y+dims[2]*z)+1+nf*coil] = zero(ComplexF32)
        end
    end
    return nothing
end

function spread_batch!(workspace::BatchedHighOrderNormalWorkspace{B},execution::PrunedHighOrderExecution) where {B}
    p = workspace.plan
    nx,ny,nz = size(first(p.data.us))
    nt,nf,nc = size(workspace.q,1),nx*ny*nz,size(workspace.chunks,1)
    if !isempty(execution.zero_tiles)
        @cuda threads=256 blocks=cld(length(execution.zero_tiles)*64B,256) kernel_pruned_clear!(
            p.data.buffer,execution.zero_tiles,(nx,ny,nz),nf,Val(B))
    end
    types = Tuple{CD.CuPtr{ComplexF32},CD.CuPtr{ComplexF32},CD.CuPtr{Int32},CD.CuPtr{Int32},
        CD.CuPtr{eltype(execution.metadata)},CD.CuPtr{Float32},CD.CuPtr{Int32},Int32,Int32,Int32,Int32,Int32,Int32}
    fields = execution.coefficient_cache_bytes > 0 ? 12+2B : 20+2B
    CD.cudacall(execution.spread,types,p.data.buffer,workspace.samples,workspace.point_ids,workspace.chunks,
        execution.metadata,workspace.windows,workspace.permutation,Int32(nc),Int32(nx),Int32(ny),Int32(nz),Int32(nt),Int32(nf);
        threads=64,blocks=nc,shmem=fields*32*sizeof(Float32),stream=CUDA.stream())
    return nothing
end
