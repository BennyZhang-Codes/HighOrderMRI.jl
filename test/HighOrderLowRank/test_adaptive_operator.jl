function test_adaptive_operator(arrayType, nZ; gpus=[0], normal_distribution=:single)
    d = joint_operator_test_problem(nZ)
    report = Ref{Any}()
    options = (; selection_pairs=4000, acceptance_pairs=12000)
    op = HighOrderLowRankOp(d.grid, d.kspha, d.times;
        d.fieldmap, d.csm, d.mask, arrayType, gpus, normal_distribution,
        shared_basis_method=:adaptive, epsilon_H=0.01, adaptive_options=options,
        joint_basis_report=report, rsvd_chunk=23)
    try
        @test report[].empirical_status === :passed
        @test report[].certification_status === :not_requested
        @test !hasproperty(report[], :q) && !hasproperty(report[], :S)
        @test size(op.basis, 2) == report[].rank
        x = arrayType(d.x)
        forward = Array(op*x)
        @test norm(forward-d.exact*d.x[vec(d.mask)])/norm(forward) < 1e-2
        y = arrayType(randn(Random.Xoshiro(55), ComplexF32, size(op,1)))
        backward = Array(op'*y)
        left, right = dot(ComplexF64.(forward),ComplexF64.(Array(y))), dot(ComplexF64.(d.x),ComplexF64.(backward))
        @test abs(left-right)/max(abs(left),abs(right)) < 1e-4
        @test all(iszero, backward[.!vec(d.mask)])
        weights = arrayType(ComplexF32.(range(0.5, 1; length=57)))
        E = WeightingOp(ComplexF32; weights, rep=2)∘op
        N = normalOperator(E)
        @test norm(Array(N*x)-Array(E'*(E*x)))/norm(Array(N*x)) < 1e-4
        close(op);close(op)
    finally
        close(op)
    end
    @test_throws ArgumentError HighOrderLowRankOp(d.grid, d.kspha, d.times;
        d.fieldmap, d.csm, d.mask, epsilon_H=0.01)
    @test_throws ArgumentError HighOrderLowRankOp(d.grid, d.kspha, d.times;
        d.fieldmap, d.csm, d.mask, shared_basis_method=:adaptive, epsilon_H=NaN,
        joint_basis_report=report)
    @test report[].empirical_status === :not_evaluated
    @test_throws ArgumentError HighOrderLowRankOp(d.grid, d.kspha, d.times;
        d.fieldmap, d.csm, d.mask, shared_basis_method=:adaptive)
end

@testset "Adaptive operator integration" begin
    for nZ in (1, 8)
        test_adaptive_operator(Array, nZ)
    end
    d = joint_operator_test_problem(1)
    @test_throws ArgumentError HighOrderLowRankOp(d.grid, d.kspha, d.times;
        d.fieldmap, d.csm, d.mask, shared_basis_method=:adaptive, epsilon_H=0.01,
        joint_roi_tol=1f-2)
    failure_report = Ref{Any}()
    @test_throws ErrorException HighOrderLowRankOp(d.grid, d.kspha, d.times;
        d.fieldmap, d.csm, d.mask, shared_basis_method=:adaptive, epsilon_H=1e-8,
        shared_rank_max=1, adaptive_options=(; max_attempts=1, selection_pairs=4000, acceptance_pairs=12000),
        joint_basis_report=failure_report)
    @test failure_report[].construction_status === :budget_exhausted
    @test failure_report[].rank == 0
    if CUDA.functional() && Base.find_package("NonuniformFFTs") !== nothing
        previous = AbstractNFFTs.active_backend()
        try
            @eval import NonuniformFFTs
            AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
            test_adaptive_operator(CuArray, 8; gpus=[Int(CUDA.deviceid(CUDA.device()))])
            if get(ENV, "HIGHORDERMRI_TEST_MULTIGPU_NORMAL", "false") == "true"
                gpus = parse.(Int, split(ENV["HIGHORDERMRI_TEST_GPUS"], ','))
                test_adaptive_operator(CuArray, 8; gpus, normal_distribution=:rank)
            end
        finally
            AbstractNFFTs.set_active_backend!(previous)
        end
    end
end
