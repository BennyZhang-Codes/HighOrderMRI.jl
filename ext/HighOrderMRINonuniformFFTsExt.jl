module HighOrderMRINonuniformFFTsExt

using HighOrderMRI, CUDA, NonuniformFFTs, LinearAlgebra, Libdl
const H = HighOrderMRI

# NonuniformFFTs 0.9.6 plan internals are isolated in this optional extension.
# Only the coil transform dimension changes; nodes, windows and FFT sizes do not.
struct BatchedFFTData{N,B,K,U,A,F,I} <: NonuniformFFTs.AbstractNUFFTData{ComplexF32,N,B}
    ks::K
    us::U
    buffer::A
    forward::F
    backward::I
end
NonuniformFFTs.output_field(data::BatchedFFTData) = data.us
function NonuniformFFTs._type1_fft!(data::BatchedFFTData)
    data.forward * data.buffer
    return data.us
end
function NonuniformFFTs._type2_fft!(data::BatchedFFTData)
    data.backward * data.buffer
    return data.us
end

struct BatchedHighOrderNormalWorkspace{B,M,P,E}
    plan::P
    q::CuMatrix{ComplexF32}
    basis::CuMatrix{ComplexF32}
    csm::CuMatrix{ComplexF32}
    weights2::CuVector{Float32}
    samples::CuMatrix{ComplexF32}
    channel::CuMatrix{ComplexF32}
    mask_to_fft::CuVector{Int32}
    fft_to_mask::CuVector{Int32}
    deconvolution::CuVector{Float32}
    starts::CuMatrix{Int32}
    windows::CuArray{Float32,3}
    permutation::CuVector{Int32}
    point_ids::CuVector{Int32}
    chunks::CuMatrix{Int32}
    execution::E
end

function batched_fft_plan(p, ::Val{B}; plan_transforms=true) where {B}
    dims = size(first(p.data.us))
    buffer = CUDA.zeros(ComplexF32, dims..., B)
    us = ntuple(c -> view(buffer, :, :, :, c), Val(B))
    axes = (1, 2, 3)
    forward = plan_transforms ? NonuniformFFTs.AbstractFFTs.plan_fft!(buffer, axes) : nothing
    backward = plan_transforms ? NonuniformFFTs.AbstractFFTs.plan_bfft!(buffer, axes) : nothing
    data = BatchedFFTData{3,B,typeof(p.data.ks),typeof(us),typeof(buffer),typeof(forward),typeof(backward)}(
        p.data.ks, us, buffer, forward, backward)
    return NonuniformFFTs.PlanNUFFT(p.kernels, p.backend, p.kernel_evalmode, p.σ,
        p.points_ref, data, p.blocks, p.fftshift, p.index_map, p.timer, p.synchronise,
        p.point_transform_fold, p.cpu_use_atomics)
end

function kernel_cache_windows!(starts, windows, permutation, points, kernels, mode, fold, nt, ::Val{M}) where {M}
    p = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if p <= nt
        @inbounds j = permutation[p]
        for d in 1:3
            @inbounds x = fold(points[d][j])
            g = NonuniformFFTs.Kernels.evaluate_kernel(mode, kernels[d], x)
            @inbounds starts[p,d] = g.i
            for k in 1:2M
                @inbounds windows[p,k,d] = g.values[k]
            end
        end
    end
    return nothing
end

function kernel_spatial_maps!(mask_to_fft, fft_to_mask, deconvolution, indices, index_map, phi, dims, fft_dims, nv)
    v = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if v <= nv
        @inbounds i = indices[v]
        a = mod(i-1,dims[1])+1
        b = mod(div(i-1,dims[1]),dims[2])+1
        c = div(i-1,dims[1]*dims[2])+1
        @inbounds begin
            j = index_map[1][a]+fft_dims[1]*((index_map[2][b]-1)+fft_dims[2]*(index_map[3][c]-1))
            mask_to_fft[v] = j
            isempty(fft_to_mask) || (fft_to_mask[j] = v)
            deconvolution[v] = 1f0/(phi[1][a]*phi[2][b]*phi[3][c])
        end
    end
    return nothing
end

function output_axis_tiles(start, size, half_support)
    first = second = third = count = 0
    for k in 1:2half_support
        cell = mod(start-half_support+k-1,size)+1
        tile = (cell-1)÷4+1
        if tile != first && tile != second && tile != third
            count += 1
            if count == 1
                first = tile
            elseif count == 2
                second = tile
            elseif count == 3
                third = tile
            else
                throw(ArgumentError("interpolation stencil spans more than three output tiles"))
            end
        end
    end
    return (first,second,third),count
end

function output_point_lists(starts, dims, half_support)
    ntiles = cld.(dims,4)
    counts = zeros(Int32,prod(ntiles))
    function each_tile(f,p)
        a,na = output_axis_tiles(starts[p,1],dims[1],half_support)
        b,nb = output_axis_tiles(starts[p,2],dims[2],half_support)
        c,nc = output_axis_tiles(starts[p,3],dims[3],half_support)
        for k in 1:nc, j in 1:nb, i in 1:na
            f(a[i]+ntiles[1]*((b[j]-1)+ntiles[2]*(c[k]-1)))
        end
        return nothing
    end
    for p in axes(starts,1)
        each_tile(t -> counts[t]+=1,p)
    end
    offsets = cumsum(vcat(Int64(0),Int64.(counts)))
    offsets[end] < typemax(Int32) || throw(ArgumentError("batched point-list indices exceed Int32"))
    ids = Vector{Int32}(undef,offsets[end])
    cursor = copy(offsets)
    for p in axes(starts,1)
        each_tile(p) do t
            cursor[t] += 1
            ids[cursor[t]] = p
        end
    end
    chunks = NTuple{4,Int32}[]
    for tile in eachindex(counts), first in offsets[tile]+1:256:offsets[tile+1]
        push!(chunks,(Int32(first),Int32(min(first+255,offsets[tile+1])),
            Int32(tile),Int32(counts[tile]<=256)))
    end
    return ids,permutedims(reinterpret(reshape,Int32,chunks))
end

@inline periodic_window_index(value,size) =
    ifelse(value<1,value+size,ifelse(value>size,value-size,value))

function H.highorder_batched_workspace(plan::NonuniformFFTs.NFFTPlan{Float32,3},
    q,basis,csm,weights2,mask_idx,grid_size,batch; options=(;),
)
    batch in (2,4,8) || throw(ArgumentError("batched coil size must be 2, 4 or 8"))
    p = plan.p
    p.blocks.method === :global_memory || throw(ArgumentError("batched normal requires global-memory point blocks"))
    NonuniformFFTs.get_sort_points(p.blocks) === NonuniformFFTs.False() ||
        throw(ArgumentError("batched normal requires unpermuted point storage"))
    M = NonuniformFFTs.Kernels.half_support(first(p.kernels))
    dims = size(first(p.data.us))
    minimum(dims) >= 2M || throw(ArgumentError("batched FFT axes must cover the interpolation stencil"))
    maximum(dims) <= 2^24 || throw(ArgumentError("batched point offsets exceed exact Float32 integer range"))
    nt,nv = size(q,1),length(mask_idx)
    nt < typemax(Int32) || throw(ArgumentError("batched sample indices exceed Int32"))
    starts = CUDA.zeros(Int32,nt,3)
    windows = CUDA.zeros(Float32,nt,2M,3)
    permutation = CuArray(Int32.(Array(p.blocks.pointperm)))
    @cuda threads=256 blocks=cld(nt,256) kernel_cache_windows!(starts,windows,permutation,
        p.points,p.kernels,p.kernel_evalmode,p.point_transform_fold,nt,Val(M))
    ids,chunks = output_point_lists(Array(starts),dims,M)
    mask_to_fft = CUDA.zeros(Int32,nv)
    experimental = get(options, :fft, :cufft) === :vkfft_pruned
    fft_to_mask = CUDA.zeros(Int32,experimental ? 0 : prod(dims))
    deconvolution = CUDA.zeros(Float32,nv)
    phi = map(NonuniformFFTs.fourier_coefficients,p.kernels)
    @cuda threads=256 blocks=cld(nv,256) kernel_spatial_maps!(mask_to_fft,fft_to_mask,
        deconvolution,mask_idx,p.index_map,phi,grid_size,dims,nv)
    batched = batched_fft_plan(p,Val(batch); plan_transforms=!experimental)
    execution = nothing
    try
        if experimental
            M == 3 || throw(ArgumentError("experimental pruned execution requires NUFFT half-support 3"))
            batched, execution = pruned_batched_plan(batched, starts, windows, ids, chunks, mask_idx, grid_size, options, Val(batch))
        end
        return BatchedHighOrderNormalWorkspace{batch,M,typeof(batched),typeof(execution)}(batched,q,basis,csm,weights2,
            CUDA.zeros(ComplexF32,nt,batch),CUDA.zeros(ComplexF32,nt,batch),
            mask_to_fft,fft_to_mask,deconvolution,starts,windows,permutation,CuArray(ids),CuArray(chunks),execution)
    catch
        execution === nothing || close(execution)
        rethrow()
    end
end

function kernel_batched_pad!(buffer,x,basis,r,csm,c0,fft_to_mask,beta,nf,C,::Val{B}) where {B}
    j = (blockIdx().x-1)*blockDim().x+threadIdx().x
    if j <= nf
        @inbounds v = fft_to_mask[j]
        if v == 0
            for coil in 1:B
                @inbounds buffer[j+nf*(coil-1)] = 0f0
            end
        else
            @inbounds value = x[v]*conj(basis[v,r])
            @inbounds scale = beta[v]
            for coil in 1:B
                @inbounds buffer[j+nf*(coil-1)] = coil+c0<=C ? value*csm[v,coil+c0]*scale : 0f0
            end
        end
    end
    return nothing
end

function kernel_batched_gather!(channel,grid,starts,windows,permutation,dims,nt,nf,prefactor,q,r,::Val{M},::Val{B}) where {M,B}
    thread = (blockIdx().x-1)*blockDim().x+threadIdx().x-1
    lane = mod(thread,8)
    p = div(thread,8)+1
    if p <= nt
        # Each sample owns eight lanes, including in the final partial warp.
        # Name only that group in the shuffle's synchronisation mask.
        members = UInt32(0xff) << (8 * mod(div(thread, 8), 4))
        @inbounds origin = (starts[p,1]-M,starts[p,2]-M,starts[p,3]-M)
        values = ntuple(_ -> zero(ComplexF32),Val(B))
        for k in lane:8:(2M)^3-1
            a,b,c = mod(k,2M)+1,mod(div(k,2M),2M)+1,div(k,(2M)^2)+1
            x,y,z = periodic_window_index(origin[1]+a,dims[1]),
                periodic_window_index(origin[2]+b,dims[2]),periodic_window_index(origin[3]+c,dims[3])
            index = x+dims[1]*((y-1)+dims[2]*(z-1))
            @inbounds factor = windows[p,a,1]*windows[p,b,2]*windows[p,c,3]
            previous = values
            values = ntuple(coil -> (@inbounds previous[coil]+grid[index+nf*(coil-1)]*factor),Val(B))
        end
        for offset in (4,2,1)
            previous = values
            values = ntuple(Val(B)) do coil
                value = previous[coil]
                value+ComplexF32(shfl_xor_sync(members,real(value),offset,8),
                    shfl_xor_sync(members,imag(value),offset,8))
            end
        end
        if lane == 0
            @inbounds j = permutation[p]
            for coil in 1:B
                @inbounds channel[j+nt*(coil-1)] += values[coil]*prefactor*q[j,r]
            end
        end
    end
    return nothing
end

function kernel_weighted_samples!(samples,channel,q,weights,r,nt,::Val{W}) where {W}
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    if i <= length(samples)
        t = mod(i-1,nt)+1
        @inbounds samples[i] = channel[i]*conj(q[t,r])*(W ? weights[t] : 1f0)
    end
    return nothing
end

@inline function packed_sample(shared::CuDeviceVector{Float32,A},offset,coil,::Val{M}) where {A,M}
    first = offset+4cld(3+6M,4)+4div(coil-1,2)+1
    pointer4 = reinterpret(Core.LLVMPtr{NTuple{4,VecElement{Float32}},A},pointer(shared,first))
    values = unsafe_load(pointer4,1,Val(16))
    component = 2mod(coil-1,2)+1
    return ComplexF32(values[component].value,values[component+1].value)
end

function kernel_batched_spread!(grid,samples,starts,windows,permutation,ids,chunks,dims,nt,nf,::Val{M},::Val{B}) where {M,B}
    chunk,tid = blockIdx().x,threadIdx().x
    sample_base = 4cld(3+6M,4)
    fields = sample_base+2B
    shared = CuDynamicSharedArray(Float32,fields*32)
    @inbounds first,last,tile = chunks[chunk,1],chunks[chunk,2],chunks[chunk,3]-1
    @inbounds single = chunks[chunk,4]==1
    tiles = cld.(dims,4)
    origin = (mod(tile,tiles[1])*4,mod(div(tile,tiles[1]),tiles[2])*4,div(tile,tiles[1]*tiles[2])*4)
    coords = (mod(tid-1,4)+1+origin[1],mod(div(tid-1,4),4)+1+origin[2],div(tid-1,16)+1+origin[3])
    total = ntuple(_ -> zero(ComplexF32),Val(B))
    for batch_first in first:32:last
        count = min(32,last-batch_first+1)
        for localp in tid:blockDim().x:count
            @inbounds p = ids[batch_first+localp-1]
            offset = (localp-1)*fields
            for d in 1:3
                @inbounds shared[offset+d] = Float32(starts[p,d])
                for k in 1:2M
                    @inbounds shared[offset+3+(d-1)*2M+k] = windows[p,k,d]
                end
            end
            @inbounds j = permutation[p]
            for coil in 1:B
                @inbounds value = samples[j+nt*(coil-1)]
                @inbounds shared[offset+sample_base+2coil-1] = real(value)
                @inbounds shared[offset+sample_base+2coil] = imag(value)
            end
        end
        sync_threads()
        for localp in 1:count
            offset = (localp-1)*fields
            @inbounds delta = ntuple(d -> periodic_window_index(Float32(coords[d]+M)-shared[offset+d],dims[d]),3)
            if all(d -> 1<=delta[d]<=2M,1:3)
                a,b,c = unsafe_trunc.(Int32,delta)
                @inbounds factor = shared[offset+3+a]*shared[offset+3+2M+b]*shared[offset+3+4M+c]
                previous = total
                total = ntuple(coil -> previous[coil]+packed_sample(shared,offset,coil,Val(M))*factor,Val(B))
            end
        end
        sync_threads()
    end
    if coords[1]<=dims[1] && coords[2]<=dims[2] && coords[3]<=dims[3]
        index = coords[1]+dims[1]*((coords[2]-1)+dims[2]*(coords[3]-1))
        for coil in 1:B
            i = index+nf*(coil-1)
            value = total[coil]
            if single
                @inbounds grid[2i-1] = real(value)
                @inbounds grid[2i] = imag(value)
            else
                @inbounds NonuniformFFTs._atomic_add!(grid,real(value),(2i-1,))
                @inbounds NonuniformFFTs._atomic_add!(grid,imag(value),(2i,))
            end
        end
    end
    return nothing
end

function kernel_batched_reduce!(out,buffer,basis,r,csm,c0,mask_to_fft,beta,nf,nv,C,scale,::Val{B}) where {B}
    v = (blockIdx().x-1)*blockDim().x+threadIdx().x
    if v <= nv
        @inbounds j = mask_to_fft[v]
        value = zero(ComplexF32)
        for coil in 1:B
            if coil+c0 <= C
                @inbounds value += buffer[j+nf*(coil-1)]*conj(csm[v,coil+c0])
            end
        end
        @inbounds out[v] += value*(beta[v]*scale)*basis[v,r]
    end
    return nothing
end

function forward_batch!(workspace::BatchedHighOrderNormalWorkspace{B,M},x,c0) where {B,M}
    p = workspace.plan
    dims = size(first(p.data.us))
    nf,nt,C = prod(dims),size(workspace.q,1),size(workspace.csm,2)
    prefactor = prod(map(NonuniformFFTs.Kernels.gridstep,p.kernels))
    fill!(workspace.channel,0)
    for r in axes(workspace.q,2)
        pad_batch!(workspace,x,r,c0,workspace.execution)
        NonuniformFFTs._type2_fft!(p.data)
        gather_batch!(workspace,r,workspace.execution)
    end
    return nothing
end

function adjoint_batch!(out,workspace::BatchedHighOrderNormalWorkspace{B,M},c0,weighted) where {B,M}
    p = workspace.plan
    dims = size(first(p.data.us))
    nf,nt,nv,C = prod(dims),size(workspace.q,1),length(out),size(workspace.csm,2)
    scale = Float32(prod(n -> 2π/n,dims))
    for r in axes(workspace.q,2)
        @cuda threads=256 blocks=cld(nt*B,256) kernel_weighted_samples!(workspace.samples,
            workspace.channel,workspace.q,workspace.weights2,r,nt,weighted)
        spread_batch!(workspace,workspace.execution)
        NonuniformFFTs._type1_fft!(p.data)
        @cuda threads=256 blocks=cld(nv,256) kernel_batched_reduce!(out,p.data.buffer,
            workspace.basis,r,workspace.csm,c0,workspace.mask_to_fft,workspace.deconvolution,nf,nv,C,scale,Val(B))
    end
    return nothing
end

function H.highorder_batched_normal!(out,workspace::BatchedHighOrderNormalWorkspace{B},x) where {B}
    fill!(out,0)
    for c0 in 0:B:size(workspace.csm,2)-1
        forward_batch!(workspace,x,c0)
        adjoint_batch!(out,workspace,c0,Val(true))
    end
    return out
end

function H.highorder_batched_adjoint!(out,workspace::BatchedHighOrderNormalWorkspace{B},signal) where {B}
    fill!(out,0)
    for c0 in 0:B:size(workspace.csm,2)-1
        count = min(B,size(workspace.csm,2)-c0)
        fill!(workspace.channel,0)
        @views workspace.channel[:,1:count] .= signal[:,c0+1:c0+count]
        # A bare adjoint receives any weighting from the caller, once.
        adjoint_batch!(out,workspace,c0,Val(false))
    end
    return out
end

function H.highorder_batched_forward!(signal,workspace::BatchedHighOrderNormalWorkspace{B},x) where {B}
    for c0 in 0:B:size(workspace.csm,2)-1
        forward_batch!(workspace,x,c0)
        count = min(B,size(workspace.csm,2)-c0)
        @views signal[:,c0+1:c0+count] .= workspace.channel[:,1:count]
    end
    return signal
end


function pad_batch!(workspace::BatchedHighOrderNormalWorkspace{B},x,r,c0,::Nothing) where {B}
    p = workspace.plan
    nf, C = length(p.data.buffer) ÷ B, size(workspace.csm,2)
    @cuda threads=256 blocks=cld(nf,256) kernel_batched_pad!(p.data.buffer,x,
        workspace.basis,r,workspace.csm,c0,workspace.fft_to_mask,workspace.deconvolution,nf,C,Val(B))
    return nothing
end

function gather_batch!(workspace::BatchedHighOrderNormalWorkspace{B,M},r,::Nothing) where {B,M}
    p = workspace.plan
    dims = size(first(p.data.us))
    nf, nt = prod(dims), size(workspace.q,1)
    prefactor = prod(map(NonuniformFFTs.Kernels.gridstep,p.kernels))
    @cuda threads=128 blocks=cld(nt*8,128) kernel_batched_gather!(workspace.channel,p.data.buffer,
        workspace.starts,workspace.windows,workspace.permutation,dims,nt,nf,prefactor,workspace.q,r,Val(M),Val(B))
    return nothing
end

function spread_batch!(workspace::BatchedHighOrderNormalWorkspace{B,M},::Nothing) where {B,M}
    p = workspace.plan
    dims, nt = size(first(p.data.us)), size(workspace.q,1)
    fill!(p.data.buffer,0)
    shared_bytes = (4cld(3+6M,4)+2B)*32*sizeof(Float32)
    @cuda threads=64 blocks=size(workspace.chunks,1) shmem=shared_bytes kernel_batched_spread!(
        reinterpret(Float32,p.data.buffer),workspace.samples,workspace.starts,workspace.windows,
        workspace.permutation,workspace.point_ids,workspace.chunks,dims,nt,prod(dims),Val(M),Val(B))
    return nothing
end

function H.release_highorder_batched_workspace!(workspace::BatchedHighOrderNormalWorkspace)
    workspace.execution === nothing || close(workspace.execution)
    return nothing
end

include("NonuniformFFTs/experimental_pruned.jl")

end
