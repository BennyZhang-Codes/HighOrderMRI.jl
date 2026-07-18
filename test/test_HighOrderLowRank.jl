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
