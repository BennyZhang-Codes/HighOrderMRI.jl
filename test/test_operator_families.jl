using Test, HighOrderMRI, CUDA, LinearAlgebra, Random
using AbstractNFFTs, LinearOperatorCollection, LinearOperators

function test_lowrank_family(arrayType, nZ)
    d = joint_operator_test_problem(nZ)
    common = (; d.fieldmap, d.csm, d.mask, arrayType, L_rank=6,
        rsvd_oversample=3, rsvd_seed=1234, rsvd_chunk=23,
        shared_rank_max=32, shared_basis_tol=1f-3)
    choices = (
        (; method=:per_readout),
        (; method=:incremental_shared),
        (; method=:direct_shared),
        (; method=:direct_shared, strategy=:global),
    )
    for selection in choices
        direct = get(selection, :method, nothing) === :direct_shared
        global_strategy = get(selection, :strategy, nothing) === :global
        options = global_strategy ? (; epsilon_H=0.01,
            adaptive_options=(; selection_pairs=1000, acceptance_pairs=4000)) :
            direct ? (; joint_snapshots=32, joint_samples=64) : (;)
        report = Ref{Any}()
        op = HighOrderLowRankOp(d.grid, d.kspha, d.times;
            common..., selection..., options..., joint_basis_report=report)
        try
            @test op isa HighOrderLowRankOp
            @test op.basis_layout === (selection.method === :per_readout ? :per_readout : :shared)
            x = arrayType(d.x)
            y = arrayType(randn(Random.Xoshiro(91), ComplexF32, size(op, 1)))
            forward, backward = op*x, op'*y
            @test norm(Array(forward)-d.exact*d.x[vec(d.mask)])/norm(forward) < 0.03
            @test norm(Array(backward)[vec(d.mask)]-d.exact'*Array(y))/norm(backward) < 0.03
            @test all(iszero, Array(backward)[.!vec(d.mask)])
            @test abs(dot(forward,y)-dot(x,backward))/max(abs(dot(forward,y)),abs(dot(x,backward))) < 1e-4
            D = arrayType(Float32.(range(0.3,1; length=size(op,1))))
            normal = normalOperator(op,D)*x
            @test norm(Array(normal)-Array(op'*(D.*forward)))/norm(normal) < 1e-4
            if direct
                @test global_strategy ? report[].empirical_status === :passed : report[].passed
            end
        finally
            close(op)
        end
    end
    # The default selects Incremental Shared; the convenience method adds one
    # readout dimension without changing the model or vector ordering.
    default = HighOrderLowRankOp(d.grid,d.kspha,d.times; common...)
    selected = HighOrderLowRankOp(d.grid,d.kspha,d.times;
        common..., method=:incremental_shared)
    single = HighOrderLowRankOp(d.grid,d.kspha[:,:,1],d.times[:,1];
        common..., method=:incremental_shared)
    try
        @test Array(default.q) == Array(selected.q)
        @test Array(default.basis) == Array(selected.basis)
        @test default.normal_backend.method === :standard
        @test default.normal_backend.distribution === :single
        @test single.nDyn == 1 && single.basis_layout === :shared
    finally
        close(default); close(selected); close(single)
    end
end

function test_explicit_family(arrayType)
    d = joint_operator_test_problem(8)
    ns, nt = size(d.times,1), length(d.times)
    rows = vcat(collect(1:ns), collect(nt+1:nt+ns))
    A = d.exact[rows,:]
    common = (; d.fieldmap, d.csm, d.mask)
    choices = arrayType === Array ? ((; implementation=:array, arrayType=Array, nBlock=5),) : (
        (; implementation=:kernel, backend=:direct),
        (; implementation=:kernel, backend=:blocked, phase_backend=:gemm, block_size=7),
        (; implementation=:kernel, backend=:cached_full, phase_backend=:gemm, block_size=7),
    )
    for choice in choices
        op = HighOrderExplicitOp(d.grid,d.kspha[:,:,1],d.times[:,1]; common...,choice...)
        try
            @test op isa HighOrderExplicitOp
            x = arrayType(d.x)
            y = arrayType(randn(Random.Xoshiro(107),ComplexF32,size(op,1)))
            forward, backward = op*x, op'*y
            @test norm(Array(forward)-A*d.x[vec(d.mask)])/norm(forward) < 3e-5
            @test norm(Array(backward)[vec(d.mask)]-A'*Array(y))/norm(backward) < 3e-5
            @test all(iszero,Array(backward)[.!vec(d.mask)])
            D = arrayType(Float32.(range(0.1,1;length=size(op,1))))
            normal = normalOperator(op,D)*x
            @test norm(Array(normal)[vec(d.mask)]-A'*(Array(D).*(A*d.x[vec(d.mask)])))/norm(normal) < 3e-5
        finally
            close(op)
        end
    end
end

@testset "Explicit and LowRank family interfaces" begin
    @test !isdefined(HighOrderMRI, :HighOrderOp)
    @test !isdefined(HighOrderMRI, :HighOrderKernelOp)
    @test !isdefined(HighOrderMRI, :lowrank_method_options)
    @test !isdefined(HighOrderMRI, :highorder_lowrank_operator)
    @test :ArrayExplicitOp ∉ names(HighOrderMRI)
    @test :CUDAExplicitOp ∉ names(HighOrderMRI)
    test_explicit_family(Array)
    for nZ in (1,8)
        test_lowrank_family(Array,nZ)
    end
    d = joint_operator_test_problem(1)
    f(; kwargs...) = HighOrderLowRankOp(d.grid,d.kspha,d.times; kwargs...)
    @test_throws ArgumentError f(method=:unknown)
    @test_throws ArgumentError f(strategy=:global)
    @test_throws ArgumentError f(method=:per_readout,strategy=:global)
    @test_throws ArgumentError f(method=:direct_shared,strategy=:unknown)
    @test_throws MethodError f(basis_layout=:shared)
    @test_throws MethodError f(shared_basis_method=:rsvd)
    @test_throws MethodError f(method=:direct_shared,shared_basis_method=:adaptive)
    @test_throws ArgumentError HighOrderExplicitOp(d.grid,d.kspha[:,:,1],d.times[:,1];implementation=:unknown)
    if CUDA.functional() && Base.find_package("NonuniformFFTs") !== nothing
        previous = AbstractNFFTs.active_backend()
        try
            @eval import NonuniformFFTs
            AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
            test_explicit_family(CuArray)
            test_lowrank_family(CuArray,8)
        finally
            AbstractNFFTs.set_active_backend!(previous)
        end
    end
end
