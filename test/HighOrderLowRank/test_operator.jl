@testset "HighOrderLowRankOp" begin
    grid, kspha, times, fieldmap, csm, mask, recon_terms =
        highorder_lowrank_test_data()
    nSam = size(kspha, 2)
    nDyn = size(kspha, 3)
    nCha = size(csm, 3)
    L_rank = 2

    op = HighOrderLowRankOp(
        grid,
        kspha,
        times;
        fieldmap,
        csm,
        mask,
        recon_terms,
        L_rank,
        rsvd_seed=0,
        rsvd_finalize=:gram,
    )

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
        fieldmap,
        csm,
        mask,
        recon_terms,
        L_rank,
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
