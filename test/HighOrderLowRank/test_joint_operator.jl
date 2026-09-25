function joint_operator_test_problem(nZ; nCha=2)
    T = Float32
    nX, nY, nSam, nDyn = 8, 9, 19, 3
    grid = Grid(nX, nY, nZ, T(0.8), T(1.3), T(0.9))
    kspha = zeros(T, 9, nSam, nDyn)
    times = repeat(reshape(collect(range(0f0, 1f-3; length=nSam)), :, 1), 1, nDyn)
    for d in 1:nDyn
        kspha[1, :, d] .= T(0.01d)
        kspha[2, :, d] .= range(-0.04f0, 0.05f0; length=nSam)
        kspha[3, :, d] .= range(0.03f0, -0.02f0; length=nSam)
        kspha[4, :, d] .= T(0.02d)
        kspha[5, :, d] .= range(-5f-4, 5f-4; length=nSam)
        kspha[9, :, d] .= T(2e-4d)
    end
    rng = Random.Xoshiro(827)
    fieldmap = randn(rng, T, nX, nY, nZ) .* T(5)
    csm = randn(rng, Complex{T}, nX, nY, nZ, nCha)
    mask = trues(nX, nY, nZ)
    mask[1, 1, 1] = mask[end, end, end] = false
    x = randn(rng, Complex{T}, length(mask))

    masked = vec(mask)
    bf = HighOrderMRI.basisfunc_spha(grid.x[masked], grid.y[masked], grid.z[masked], collect(1:9))
    temporal = permutedims(reshape(Float64.(kspha), 9, :))
    phase = temporal * Float64.(bf)' + vec(Float64.(times)) * vec(Float64.(fieldmap))[masked]'
    encoding = cis.(2π .* phase) ./ sqrt(count(mask))
    sensitivities = reshape(csm, :, nCha)[masked, :]
    exact = vcat([encoding .* transpose(sensitivities[:, c]) for c in 1:nCha]...)
    return (; grid, kspha, times, fieldmap, csm, mask, x, exact)
end

function test_joint_operator(arrayType, nZ; gpus=[0], normal_distribution=:single, nCha=2)
    data = joint_operator_test_problem(nZ; nCha)
    report = Ref{Any}()
    op = HighOrderLowRankOp(data.grid, data.kspha, data.times;
        data.fieldmap, data.csm, data.mask, arrayType, gpus,
        shared_basis_method=:joint, shared_basis_tol=1f-3,
        shared_rank_max=32, joint_snapshots=32, joint_samples=64,
        joint_roi_tol=1f-3, joint_basis_report=report,
        normal_distribution, rsvd_chunk=23,
    )
    try
        @test op isa HighOrderLowRankOp
        @test report[].passed
        @test size(op) == (57 * nCha, length(data.mask))
        @test size(op.basis, 2) == report[].rank
        x = arrayType(data.x)
        y = arrayType(randn(Random.Xoshiro(913), ComplexF32, size(op, 1)))
        forward = op * x
        backward = op' * y
        reference = data.exact * data.x[vec(data.mask)]
        @test norm(Array(forward) - reference) / norm(reference) < 2e-3
        back_reference = data.exact' * Array(y)
        @test norm(Array(backward)[vec(data.mask)] - back_reference) / norm(back_reference) < 2e-3
        @test all(iszero, Array(backward)[.!vec(data.mask)])
        lhs, rhs = dot(forward, y), dot(x, backward)
        @test abs(lhs - rhs) / max(abs(lhs), abs(rhs)) < 1e-4
        weights = arrayType(ComplexF32.(range(0.5f0, 1f0; length=57)))
        E = ∘(WeightingOp(ComplexF32; weights, rep=nCha), op)
        normal = normalOperator(E)
        reference_normal = E' * (E * x)
        for repeat in 1:2
            result = normal * x
            @test norm(result - reference_normal) / norm(reference_normal) < 1e-4
        end
        close(op)
        @test op.normal_backend.operator === nothing
        close(op)
    finally
        close(op)
    end
end

@testset "Joint operator integration" begin
    @testset "CPU nZ=$nZ" for nZ in (1, 8)
        test_joint_operator(Array, nZ)
    end
    d = joint_operator_test_problem(1)
    @test_throws ArgumentError HighOrderLowRankOp(d.grid, d.kspha, d.times;
        d.fieldmap, d.csm, d.mask, shared_basis_method=:joint, global_basis_tol=1f-2)
    @test_throws ArgumentError HighOrderLowRankOp(d.grid, d.kspha, d.times;
        d.fieldmap, d.csm, d.mask, shared_basis_method=:joint, rsvd_distribution=:voxel)
    @test_throws ArgumentError HighOrderLowRankOp(d.grid, d.kspha, d.times;
        d.fieldmap, d.csm, d.mask, shared_basis_method=:unsupported)

    if CUDA.functional() && Base.find_package("NonuniformFFTs") !== nothing
        previous_backend = AbstractNFFTs.active_backend()
        try
            @eval import NonuniformFFTs
            AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
            @testset "CUDA nZ=$nZ" for nZ in (1, 8)
                test_joint_operator(CuArray, nZ; gpus=[Int(CUDA.deviceid(CUDA.device()))])
            end
        finally
            AbstractNFFTs.set_active_backend!(previous_backend)
        end
    end
end
