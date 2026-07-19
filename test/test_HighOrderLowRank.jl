using CUDA

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
    end

    rsvd_workspace = HighOrderMRI.RSVDWorkspace(times[:, 1], T, nSam, nVox, L_total, nVox)
    u_full, s_full, v_full = HighOrderMRI.perform_rsvd(
        times[:, 1], fieldmap_masked, bf_err, kspha_err,
        nVox, nSam, L_rank, nVox, rsvd_workspace;
        seed=17,
    )

    rsvd_workspace = HighOrderMRI.RSVDWorkspace(times[:, 1], T, nSam, nVox, L_total, 3)
    u_chunk, s_chunk, v_chunk = HighOrderMRI.perform_rsvd(
        times[:, 1], fieldmap_masked, bf_err, kspha_err,
        nVox, nSam, L_rank, 3, rsvd_workspace;
        seed=17,
    )

    E_full = u_full * Diagonal(s_full) * adjoint(v_full)
    E_chunk = u_chunk * Diagonal(s_chunk) * adjoint(v_chunk)

    @test norm(E_full - E_chunk) / norm(E_full) < 1f-4
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
    )

    @test op.nDyn == nDyn
    @test size(op.u) == (nSam, L_rank, nDyn)
    @test size(op.v) == (sum(mask), L_rank, nDyn)
    @test size(op.v_star) == (sum(mask), L_rank, nDyn)
    @test length(op.nfftplan) == nDyn
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
    relerr = abs(lhs - rhs) / (abs(lhs) + abs(rhs))
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
    )
    @test op_2d.nDyn == 1
    @test size(op_2d) == (nSam * nCha, prod(grid.matrixSize))
end
