# This file contains functions which help to compute approximations to the augmented encoding operator ̃E
# using various methods such as streaming SVD, interpolation-based methods, and turnstile sketching methods.

# Gets the spatial and temporal functions (separable) that approximate the encoding matrix perturbations which arise in MRI imaging measured by external NMR probes
function get_bl_cl_coeffs(grid, kspha; method=:streaming, num_L=10, reduc_fac_space=2, reduc_fac_time=90, bf_choice=collect(4:16), use_gpu=true)

    if method == :streaming
        return get_bl_cl_coeffs_isvd(grid, kspha; num_L=num_L, reduc_fac_space=reduc_fac_space, reduc_fac_time=reduc_fac_time, bf_choice=bf_choice, use_gpu=use_gpu)
    elseif method == :interp
        return get_bl_cl_coeffs_interp(grid, kspha; num_L=num_L, reduc_fac_space=reduc_fac_space, reduc_fac_time=reduc_fac_time, bf_choice=bf_choice, use_gpu=use_gpu)
    elseif method == :turnstile
        return get_bl_cl_coeffs_turnstile(grid, kspha; num_L=num_L, reduc_fac_space=reduc_fac_space, reduc_fac_time=reduc_fac_time, bf_choice=bf_choice, use_gpu=use_gpu)
    else
        error("Unknown method: $method")
    end
end

function get_bl_cl_coeffs_interp(grid, kspha; num_L=10, reduc_fac_space=2, reduc_fac_time=90, bf_choice=collect(4:16), use_gpu=false)

    num_basis = length(bf_choice)

    # compute basis functions (solid harmonics)
    bf = basisfunc_spha(grid.x, grid.y, grid.z, bf_choice)

    # for ii = 1:size(bf,2)
    #     bf[:,ii] = rotl90(reshape(bf[:,ii],nX,nY))[:]
    # end

    s_bf_orig = size(bf)

    # put the basis functions on GPU
    if use_gpu
        bf = bf |> gpu
    end

    s_bf_orig = size(bf)

    # put the basis functions on GPU
    if use_gpu
        bf = bf |> gpu
    end

    # subsample the basis functions by meanpooling
    pooled_bf = meanpool(reshape(bf, s_bf_orig[1], 1, num_basis), (reduc_fac_space^3,))

    if use_gpu
        CUDA.unsafe_free!(bf)
    end

    s_kspha_orig = size(kspha)

    # put kspha on GPU
    if use_gpu
        kspha = reshape(kspha[:, bf_choice], s_kspha_orig[1], 1, num_basis) |> gpu
    else
        kspha = reshape(kspha[:, bf_choice], s_kspha_orig[1], 1, num_basis)
    end

    pooled_kspha = meanpool(kspha, (reduc_fac_time,))

    if use_gpu
        CUDA.unsafe_free!(kspha)
    end

    # Compute the SVD of the higher order perturbations to the encoding operator


    # make the small version of E
    E_lite = cispi.(-2 * dropdims(pooled_bf, dims=2) * dropdims(pooled_kspha, dims=2)')
    @info size(E_lite)
    @info size(dropdims(pooled_bf, dims=2))
    @info size(dropdims(pooled_kspha, dims=2))
    U_e, S_e, V_e = svd(E_lite) # There may be better ways to compute such a decomposition! Remember, we don't need the singular values alone, we just need an orthogonal decomposition that approximates the encoding operator.

    if use_gpu
        CUDA.unsafe_free!(E_lite)
    end

    bls = reshape(U_e[:, 1:num_L], size(U_e, 1), 1, num_L)
    bls_linear_fullsize = upsample_linear(bls; size=s_bf_orig[1])
    bls_fullsize = reshape(bls_linear_fullsize, grid.nX, grid.nY, grid.nZ, 1, num_L)

    cls = reshape((V_e.*conj(S_e)')[:, 1:num_L], size(V_e, 1), 1, num_L)
    cls_fullsize = upsample_linear(cls; size=s_kspha_orig[1])

    if use_gpu
        bls_fullsize = bls_fullsize |> cpu
        cls_fullsize = cls_fullsize |> cpu
        U_e = U_e |> cpu
        S_e = S_e |> cpu
        V_e = V_e |> cpu
    end

    bls = dropdims(bls_fullsize, dims=4)
    cls = dropdims(cls_fullsize, dims=2)
    @info size(bls)
    @info size(cls)

    return bls, cls

end

function get_bl_cl_coeffs_isvd(grid, kspha; num_L=10, reduc_fac_space=2, reduc_fac_time=90, bf_choice=collect(4:16), use_gpu=false)

    num_basis = length(bf_choice)

    # compute basis functions (solid harmonics)
    bf = basisfunc_spha(grid.x, grid.y, grid.z, bf_choice)

    s_bf_orig = size(bf)

    # put the basis functions on GPU
    if use_gpu
        bf = bf |> gpu
    end

    s_bf_orig = size(bf)

    # put the basis functions on GPU
    if use_gpu
        bf = bf |> gpu
    end

    s_kspha_orig = size(kspha)

    # put kspha on GPU
    if use_gpu
        kspha |> gpu
    end

    # Compute the SVD of the higher order perturbations to the encoding operator
    X = cispi.(-2 * bf * kspha[:, bf_choice]')

    # Use an incremental streaming based-SVD to compute the basis functions and coefficients
    # This is more memory efficient than the full SVD, especially for large datasets
    U, s = IncrementalSVD.update!(nothing, nothing, cispi.(-2 * bf * kspha[1:80:end, bf_choice]'))

    for ii = 2:10:80 # TODO update these parameters 
        IncrementalSVD.update!(U, s, cispi.(-2 * bf * kspha[ii:80:end, bf_choice]'))
    end

    if use_gpu
        U = U |> gpu
        s = s |> gpu
    end
    
    Vt = Diagonal(s) \ (U' * X) # NEED to make sure we can always compute this, it may be extremely large!

    bls = reshape(U[:, 1:num_L], grid.nX, grid.nY, 1, num_L) # TODO add Z dimension
    cls = (Vt.*(s))[1:num_L, :]'

    return bls, cls

end

function get_bl_cl_coeffs_turnstile(grid, kspha; num_L=10, reduc_fac_space=2, reduc_fac_time=90, bf_choice=collect(4:16), use_gpu=false)


    # TODO: Run things on the GPU, it might make stuff a LOT faster
    # TODO: Now, for 3D etc... and segmented data, we know the k-space modulation will not necessarily be smooth, but! we know that the spatial support will be smooth. So we can probably hybridize our approach with Berti's original interpolation approach

    # compute basis functions (solid harmonics)
    bf = basisfunc_spha(grid.x, grid.y, grid.z, bf_choice)

    # # put the argument args onto the gpu
    # if use_gpu
    #     bf = bf |> gpu
    #     kspha |> gpu
    # end

    # Use a turnstile model streaming based-SVD to compute the basis functions and coefficients 
    # (See Tropp et al. 2017 "Practical Sketching Algorithms for Low-Rank Matrix Approximation")

    # Notes on dimensions of the matrices: 
    # A is (num_kspace_points x num_basis)
    # bf is (num_voxels x num_basis)
    # kspha is (num_kspace_points x num_basis)  

    # The example is set up to stream over the rows of A (i.e., over k-space points) 
    # This is extremely helpful because this is how we get k-space coefficients
    # As far as I'm aware, this method should be a really good way to do this if we don't assume temporal smoothness

    # set up blocksize
    stream_block_size = min(num_L + 100, size(kspha,1))

    # Make the turnstile model (streaming rows off my matrix)
    row_iterator = HighOrderMRI.row_blocks(kspha[:,bf_choice], bf; blocksize=stream_block_size)

    # Set up the sketching matrices
    Ω, γ = HighOrderMRI.make_standard_normal_sketching_matrices(size(kspha,1), size(bf,1), stream_block_size, stream_block_size)
    Ψ, Φ = HighOrderMRI.make_standard_normal_sketching_matrices(size(kspha,1), size(bf,1), 2*stream_block_size, 2*stream_block_size)

    X,Y,Z = HighOrderMRI.streaming_sketch(row_iterator, γ, Ω, Ψ, Φ)
    U,s,V = HighOrderMRI.svd_from_sketch(X,Y,Z,Φ,Ψ)

    # # Vt = transpose(V)
    # if use_gpu
    #     U = U |> gpu
    #     s = s |> gpu
    #     V = V |> gpu
    # end
    
    # bls = reshape((V[:,1:num_L].* s[1:num_L]')[:,1:num_L], 200, 200, 1, num_L)
    # bls = permutedims(reshape( Diagonal(s[1:num_L])*Vt[1:num_L,:], num_L,1,grid.nX, grid.nY),[3,4,2,1])
    # cls = conj.(U[:, 1:num_L])

    # Remember! Since we streamed the rows, we have approximated ̃Eᴴ as U * S * Vᴴ, not ̃E. To get ̃E, we use simple rearrangement: ̃E = ̄V * S * Uᵀ where the left singular vectors are conj.(V)*Diag(s) and right singular vectors are transpose of U. 
    # Remember again, though that the cₗ values are the conjugate of the right singular vectors, so we conjugate U elementwise prior to returning

    bls = reshape(conj.(V[:,1:num_L])*Diagonal(s[1:num_L]), grid.nX,grid.nY,1,num_L)
    cls = conj.(U[:,1:num_L])

    return bls, cls

end

# Create Gaussian sketch matrices (you can replace with SRFT or CountSketch)
# n = # columns of A, m = # rows of A
function make_standard_normal_sketching_matrices(m::Int, n::Int, ℓ::Int, s::Int; rng=Random.GLOBAL_RNG)
    Ω = randn(rng, ComplexF64, (n, ℓ))   # for Y = A * Ω
    Ψ = randn(rng, ComplexF64, (s, m))   # for Z = Ψ * A
    return Ω, Ψ
end

"""
    row_blocks(A; blocksize=10000)

Iterate over row blocks of a dense matrix A, each of size at most blocksize × size(A,2).
"""
function row_blocks(kspha, bf; blocksize=100)
    m = size(kspha,1)
    @views return ( cispi.(-2 * kspha[i:min(i+blocksize-1, m), :] * bf') for i in 1:blocksize:m)
end

# Streaming updates for the sketch matrices of SketchySVD
# stream_blocks is an iterator that yields row-blocks of A as matrices (rows x n)
function streaming_sketch(stream_blocks,γ,Ω,Ψ,Φ)
    m, s = size(γ')    # number of rows of A
    n, ℓ = size(Ω)

    # Initialize the sketch matrices
    Y = zeros(ComplexF64, m, ℓ)
    X = zeros(ComplexF64, s, n)  

    # Sketch core
    Z = zeros(ComplexF64, size(Φ,1), size(Ψ,2))  

    # Probably there's a more efficient way to iterate through this
    row_offset = 0
    for block in stream_blocks

        @info row_offset
        # block: r x n (r rows of A)
        r = size(block, 1)
        rows_range = (row_offset+1):(row_offset+r)

        # Follow the update equations from SketchySVD slides
        Y[rows_range, :] .+= block * Ω
        X .+= γ[:, rows_range] * block
        Z .+= Φ[:, rows_range] * block * Ψ
        
        row_offset += r
    end

    return X,Y,Z
end

# Turnstile Model SketchySVD from Tropp et al.
function svd_from_sketch(X,Y,Z,Φ,Ψ; r=50)

    # thin QR on Y and conj_transpose(X)
    qr_y = qr(Y)
    qr_p = qr(X')

    Q = Matrix(qr_y.Q)
    P = Matrix(qr_p.Q)

    C = ((Φ*Q)\Z)/(P'*Ψ)

    svd_core = svd(C)

    return Q*svd_core.U, svd_core.S, P*svd_core.V

end
