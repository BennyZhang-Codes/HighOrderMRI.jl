using CUDA
using AbstractNFFTs

const T = Float32

function highorder_lowrank_test_data(; nDyn::Int=2)
    nX, nY, nZ = 4, 4, 1
    nSam, nCha, nTerm = 12, 2, 9
    recon_terms = "111"

    grid = Grid(nX, nY, nZ, one(T), one(T), one(T))
    kspha = zeros(T, nTerm, nSam, nDyn)
    times = zeros(T, nSam, nDyn)

    for dyn = 1:nDyn
        times[:, dyn] .= range(zero(T), T(1e-3); length=nSam)
        kspha[2, :, dyn] .= range(T(-0.2), T(0.2); length=nSam)
        kspha[3, :, dyn] .= T(0.05 * (dyn - 1))
        kspha[5, :, dyn] .= T(1e-3 * dyn)
    end

    fieldmap = reshape(T.(0:(nX * nY - 1)) ./ T(nX * nY), nX, nY)
    csm = ones(Complex{T}, nX, nY, nCha)
    mask = trues(nX, nY)
    mask[1, 1] = false

    return grid, kspha, times, fieldmap, csm, mask, recon_terms
end

@testset "perform_rsvd_chunked" begin
    
    grid, kspha, times, fieldmap, _, mask, _ = highorder_lowrank_test_data()
    nVox = sum(mask)
    nSam = size(kspha, 2)
    L_rank = 3
    p_oversample = 3

    L_total = L_rank + p_oversample
    
    
    fieldmap_masked = vec(fieldmap)[vec(mask)]
    bf = HighOrderMRI.basisfunc_spha(
        grid.x[vec(mask)],
        grid.y[vec(mask)],
        grid.z[vec(mask)],
        collect(1:size(kspha, 1)),
    )
    bf_err = bf[:, 5:end]
    kspha_err = kspha[5:end, :, 1]

    if CUDA.functional()
        times_d = CuArray(times[:, 1])
        fieldmap_d = CuArray(fieldmap_masked)
        bf_err_d = CuArray(bf_err)
        kspha_err_d = CuArray(kspha_err)

        rsvd_workspace = HighOrderMRI.RSVDWorkspace(times_d, T, nSam, nVox, L_total, nVox)
        @assert size(rsvd_workspace.omega)         == (nVox, L_total)
        @assert size(rsvd_workspace.W)             == (nSam, L_total)
        @assert size(rsvd_workspace.B_adj)         == (nVox, L_total)
        @assert size(rsvd_workspace.gram)          == (L_total, L_total)
        @assert size(rsvd_workspace.right_vectors) == (L_total, L_total)
        
        u_full, s_full, v_full = HighOrderMRI.perform_rsvd(
            times_d, fieldmap_d, bf_err_d, kspha_err_d,
            nVox, nSam, L_rank, nVox, rsvd_workspace;
            seed=17, p_oversample=p_oversample,
        )


        rsvd_workspace = HighOrderMRI.RSVDWorkspace(times_d, T, nSam, nVox, L_total, 3)
        u_chunk, s_chunk, v_chunk = HighOrderMRI.perform_rsvd(
            times_d, fieldmap_d, bf_err_d, kspha_err_d,
            nVox, nSam, L_rank, 3, rsvd_workspace;
            seed=17, p_oversample=p_oversample,
        )

        E_full = Array(u_full * Diagonal(s_full) * adjoint(v_full))
        E_chunk = Array(u_chunk * Diagonal(s_chunk) * adjoint(v_chunk))
        relerr_chunk = norm(E_full - E_chunk) / norm(E_full)

        @test size(u_chunk) == (nSam, L_rank)
        @test size(v_chunk) == (nVox, L_rank)
        @test length(s_chunk) == L_rank
        @test all(isfinite, Array(s_chunk))
        @test relerr_chunk < T(1e-4)


        L_rank_gram = 1
        p_oversample_gram = 3
        L_total_gram = L_rank_gram + p_oversample_gram

        rsvd_workspace_gram = HighOrderMRI.RSVDWorkspace(times_d, T, nSam, nVox, L_total_gram, 3)
        
        v_scaled_gram = similar(times_d, Complex{T}, nVox, L_rank_gram)
        
        u_gram, energy_gram = HighOrderMRI.perform_rsvd(times_d, fieldmap_d, bf_err_d,
            kspha_err_d, nVox, nSam, L_rank_gram, 3, rsvd_workspace_gram;
            seed=17, p_oversample=p_oversample, 
            rsvd_finalize=:gram, v_scaled=v_scaled_gram, gram_allow_fallback=false)
        
        u_chunk, s_chunk, v_chunk = HighOrderMRI.perform_rsvd(times_d, fieldmap_d, bf_err_d, kspha_err_d,
            nVox, nSam, L_rank_gram, 3, rsvd_workspace_gram;
            seed=17, p_oversample=p_oversample,
            rsvd_finalize=:svd,
        )

        E_svd = Array(u_chunk * Diagonal(s_chunk) * adjoint(v_chunk))
        E_gram = Array(u_gram * adjoint(v_scaled_gram))
        gram_error = norm(E_svd - E_gram) / max(norm(E_svd), eps(T))
        energy_svd = sum(abs2, Array(s_chunk))
        
        @test gram_error < T(1e-3)
        @test energy_gram ≈ energy_svd rtol=T(1e-3)
        @test all(isfinite, Array(u_gram))
        @test all(isfinite, Array(v_scaled_gram))

        phase_ref = reshape(times[:, 1], :, 1) * reshape(fieldmap_masked, 1, :) + transpose(kspha_err) * transpose(bf_err)
        E_ref = @. cis(T(2π) * phase_ref)

        omega_ref = randn(Complex{T}, nVox, L_total)
        W_ref = E_ref * omega_ref
        
        omega_d = CuArray(omega_ref)
        W_d = CUDA.zeros(Complex{T}, nSam, L_total)
        
        HighOrderMRI.run_kernel_rsvd_forward!(W_d, omega_d, times_d, fieldmap_d, bf_err_d, kspha_err_d)

        W_kernel = Array(W_d)
        forward_kernel_error = norm(W_kernel - W_ref) / max(norm(W_ref), eps(T))
        @show forward_kernel_error
        @test forward_kernel_error < T(1e-4)
        

        Q_ref = randn(Complex{T}, nSam, L_total)
        B_adj_ref = adjoint(E_ref) * Q_ref
        Q_d = CuArray(Q_ref)
        B_adj_d = CUDA.zeros(Complex{T}, nVox, L_total)
        HighOrderMRI.run_kernel_rsvd_adjoint!(B_adj_d, Q_d, times_d, fieldmap_d, bf_err_d, kspha_err_d)
        B_adj_kernel = Array(B_adj_d)
        adjoint_kernel_error = norm(B_adj_kernel - B_adj_ref) / max(norm(B_adj_ref), eps(T))
        @show adjoint_kernel_error 
        @test adjoint_kernel_error < T(1e-4)
        


        B_adj_warp_d = CUDA.zeros(Complex{T}, nVox, L_total)

        HighOrderMRI.run_kernel_rsvd_adjoint_warp!(B_adj_warp_d, Q_d, times_d, fieldmap_d, bf_err_d, kspha_err_d; threads=128)
        B_adj_warp = Array(B_adj_warp_d)
        adjoint_warp_error = norm(B_adj_warp - B_adj_ref) / max(norm(B_adj_ref), eps(T))
        adjoint_layout_error = norm(B_adj_warp - B_adj_kernel) / max(norm(B_adj_kernel), eps(T))
        @show adjoint_warp_error adjoint_layout_error
        @test adjoint_warp_error < T(1e-4)
        @test adjoint_layout_error < T(1e-4)
        

        u_kernel, s_kernel, v_kernel = HighOrderMRI.perform_rsvd(times_d, fieldmap_d, bf_err_d, kspha_err_d,
            nVox, nSam, L_rank_gram, 3, rsvd_workspace_gram;
            seed=17, p_oversample=p_oversample,
            rsvd_finalize=:svd, rsvd_backend=:kernel,
        )
        E_kernel = Array(u_kernel * Diagonal(s_kernel) * adjoint(v_kernel))
        integrated_kernel_error = norm(E_kernel - E_svd) / max(norm(E_svd), eps(T))
        @show integrated_kernel_error
        @test integrated_kernel_error < T(1e-3)
        
    end

    rsvd_workspace = HighOrderMRI.RSVDWorkspace(times[:, 1], T, nSam, nVox, L_total, nVox)
    @assert size(rsvd_workspace.omega)         == (nVox, L_total)
    @assert size(rsvd_workspace.W)             == (nSam, L_total)
    @assert size(rsvd_workspace.B_adj)         == (nVox, L_total)
    @assert size(rsvd_workspace.gram)          == (L_total, L_total)
    @assert size(rsvd_workspace.right_vectors) == (L_total, L_total)

    u_full, s_full, v_full = HighOrderMRI.perform_rsvd(
        times[:, 1], fieldmap_masked, bf_err, kspha_err,
        nVox, nSam, L_rank, nVox, rsvd_workspace;
        seed=17, p_oversample=p_oversample,
    )

    rsvd_workspace = HighOrderMRI.RSVDWorkspace(times[:, 1], T, nSam, nVox, L_total, 3)
    u_chunk, s_chunk, v_chunk = HighOrderMRI.perform_rsvd(
        times[:, 1], fieldmap_masked, bf_err, kspha_err,
        nVox, nSam, L_rank, 3, rsvd_workspace;
        seed=17, p_oversample=p_oversample,
    )

    E_full = u_full * Diagonal(s_full) * adjoint(v_full)
    E_chunk = u_chunk * Diagonal(s_chunk) * adjoint(v_chunk)

    @test norm(E_full - E_chunk) / norm(E_full) < 1f-4

    rsvd_workspace_gram = HighOrderMRI.RSVDWorkspace(times[:, 1], T, nSam, nVox, L_total, 3)
    v_scaled_gram = zeros(Complex{T}, nVox, L_rank)
    
    u_gram, energy_gram = HighOrderMRI.perform_rsvd(times[:, 1], fieldmap_masked, bf_err, kspha_err,
        nVox, nSam, L_rank, 3, rsvd_workspace_gram; 
        seed=17, p_oversample=p_oversample,
        rsvd_finalize=:gram, v_scaled=v_scaled_gram, gram_allow_fallback=false)
    
    E_gram = u_gram * adjoint(v_scaled_gram)
    
    @test norm(E_chunk - E_gram) / norm(E_chunk) < T(1e-3)
    @test energy_gram ≈ sum(abs2, s_chunk) rtol=T(1e-3)
end

@testset "HighOrderLowRankOp" begin
    grid, kspha, times, fieldmap, csm, mask, recon_terms = highorder_lowrank_test_data()
    nSam = size(kspha, 2)
    nDyn = size(kspha, 3)
    nCha = size(csm, 3)
    L_rank = 2

    op = HighOrderLowRankOp(
        grid,
        kspha,
        times;
        fieldmap=fieldmap,
        csm=csm,
        mask=mask,
        recon_terms=recon_terms,
        L_rank=L_rank,
        rsvd_seed=0,
        rsvd_finalize=:gram,
    )

    @test op.nDyn == nDyn

    nVox = sum(mask)
    nPoint = nSam * nDyn
    expected_shared_rank_max = min(128, nVox, L_rank * nDyn)
    shared_rank = size(op.basis, 2)

    @test op.nDyn == nDyn
    @test 0 < shared_rank <= expected_shared_rank_max

    @test size(op.basis) == (nVox, shared_rank)

    @test size(op.q) == (nPoint, shared_rank)

    @test size(op.csm) == (nVox, nCha)
    @test eltype(op.q) == Complex{T}
    @test eltype(op.basis) == Complex{T}

    @test all(isfinite, Array(op.q))
    @test all(isfinite, Array(op.basis))

    @test !hasproperty(op, :u)
    @test !hasproperty(op, :v_shared)

    @test size(op.nfft_traj, 3) == nDyn
    @test size(op.nfft_traj, 2) == nSam
    @test AbstractNFFTs.size_out(op.nfftplan) == (nPoint,)
    
    @test length(op.workspace.k_signal) == nPoint
    @test length(op.workspace.k_weighted) == nPoint
    
    @test size(op) == (nSam * nCha * nDyn, prod(grid.matrixSize))


    x = Complex{T}.(reshape(T.(1:prod(grid.matrixSize)), :))
    y = op * x
    x_adj = adjoint(op) * y

    y_repeat = op * x
    @test y_repeat ≈ y rtol=T(1e-5) atol=T(1e-6)

    @test length(y) == nSam * nCha * nDyn
    @test length(x_adj) == prod(grid.matrixSize)

    y_scaled = similar(y)
    mul!(y_scaled, op, x, one(T), zero(T))
    @test y_scaled ≈ y rtol=T(1e-5) atol=T(1e-6)

    y_test = Complex{T}.(collect(1:length(y))) .* Complex{T}(T(0.1), T(-0.05))
    lhs = dot(op * x, y_test)
    rhs = dot(x, adjoint(op) * y_test)
    relerr = abs(lhs - rhs) / max(abs(lhs), abs(rhs), eps(T))
    @test relerr < T(1e-4)

    op_2d = HighOrderLowRankOp(
        grid,
        kspha[:, :, 1],
        times[:, 1];
        fieldmap=fieldmap,
        csm=csm,
        mask=mask,
        recon_terms=recon_terms,
        L_rank=L_rank,
        rsvd_seed=0,
        rsvd_finalize=:gram,
    )
    shared_rank_2d = size(op_2d.basis, 2)
    @test op_2d.nDyn == 1
    @test size(op_2d.q) == (nSam, shared_rank_2d)
    @test size(op_2d.basis) == (sum(mask), shared_rank_2d)
    @test AbstractNFFTs.size_out(op_2d.nfftplan) == (nSam,)
    @test size(op_2d) == (nSam * nCha, prod(grid.matrixSize))
end
