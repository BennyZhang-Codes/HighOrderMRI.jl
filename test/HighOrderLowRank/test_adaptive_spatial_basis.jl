using Random, Serialization, Statistics

function adaptive_basis_test_problem(::Type{T}) where T
    rng = Random.Xoshiro(871)
    nSam, nDyn, nVox = 29, 5, 47
    times = reshape(collect(range(T(0.001), T(0.011); length=nSam*nDyn)), nSam, nDyn) .* T(0.25)
    fieldmap = T.(randn(rng, Float32, nVox)) .* T(80)
    kspha = T.(randn(rng, Float32, 2, nSam, nDyn)) .* T(0.05) .* T(0.25)
    bf = T.(randn(rng, Float32, nVox, 2)) .* T(0.1)
    temporal = hcat(vec(Float64.(times)), permutedims(reshape(Float64.(kspha), 2, :)))
    spatial = hcat(Float64.(fieldmap), Float64.(bf))
    exact = cis.(2π .* (temporal * spatial'))
    options = (; snapshots=20, sample_count=32, max_samples=47, fit_pool_size=47,
        max_snapshots=40, chunk_size=31, initial_parent=12, parent_increment=4,
        selection_pairs=4000, acceptance_pairs=12000, diagnostic_rows=145,
        selection_seed=2026111201, acceptance_seed=2026111202, diagnostic_seed=2026111203)
    return (; times, fieldmap, bf, kspha, temporal, spatial, exact, options)
end

function test_adaptive_basis(arrayType, ::Type{T}) where T
    d = adaptive_basis_test_problem(T)
    original = deepcopy(d)
    build(; kwargs...) = HighOrderMRI.joint_adaptive_spatial_basis(d.times, d.fieldmap, d.bf, d.kspha;
        arrayType, epsilon_H=0.015, merge(d.options, (; kwargs...))...)
    result = build()
    @test result.construction_status === :computed
    @test result.empirical_status === :passed
    @test result.error_metric === :global_relative_frobenius
    @test result.certification_status === result.external_audit_status === :not_requested
    @test result.epsilon_H == 0.015
    @test result.q isa Matrix{Complex{T}} && result.S isa Matrix{Complex{T}}
    @test size(result.q) == (145, result.rank)
    @test size(result.S) == (47, result.rank)
    prediction = ComplexF64.(result.q) * ComplexF64.(result.S)'
    @test norm(prediction-d.exact)/norm(d.exact) < 0.015
    optimal = svdvals(d.exact)
    optimal_rank = findfirst(r -> norm(optimal[r+1:end])/norm(optimal) <= 0.015, eachindex(optimal))
    @test result.rank >= optimal_rank
    @test norm(ComplexF64.(result.S)'*ComplexF64.(result.S)-I) < 100eps(T)*result.rank
    @test d == original
    mktemp() do path, io
        close(io)
        serialize(path, result)
        saved = deserialize(path)
        @test saved.q == result.q && saved.S == result.S
        @test saved.certification_status === :not_requested
    end
    repeated = build()
    @test repeated.rank == result.rank
    @test norm(repeated.q*repeated.S'-result.q*result.S')/norm(d.exact) < 1e-6
    if T === Float32
        parent = build(initial_parent=2)
        @test parent.empirical_status === :passed
        @test any(t -> t.action === :expand_parent, parent.trials)
        snapshots = build(snapshots=2)
        @test snapshots.empirical_status === :passed
        @test any(t -> t.action === :expand_K, snapshots.trials)
        fitting = build(sample_count=16, condition_limit=1.01)
        @test fitting.empirical_status === :passed
        @test any(t -> t.action === :expand_J, fitting.trials)
        @test count(t -> t.rebuild, fitting.trials) == 1
    end
    failure = build(rank_max=1, max_attempts=1)
    @test failure.construction_status === :budget_exhausted
    @test failure.empirical_status === :not_passed
    @test failure.failure_reason === :parent_budget
    @test failure.rank == 0 && isnothing(failure.q) && isnothing(failure.S)
    @test failure.certification_status === failure.external_audit_status === :not_requested
    attempts = build(initial_parent=2, max_attempts=1)
    @test attempts.failure_reason === :attempt_budget
    @test isnothing(attempts.q) && isnothing(attempts.S)
    timeout = build(max_seconds=nextfloat(0.0))
    @test timeout.failure_reason === :time_budget && isempty(timeout.trials)
end

@testset "Joint adaptive low-rank basis" begin
    @test HighOrderMRI.adaptive_spatial_basis === HighOrderMRI.joint_adaptive_spatial_basis
    @testset "CPU $T" for T in (Float32, Float64)
        test_adaptive_basis(Array, T)
    end
    @testset "Paired global residual" begin
        temporal = reshape([0.0, 0.3, 0.9, 1.2, 2.0], :, 1)
        spatial = reshape([-1.0, 0.2, 1.0], :, 1)
        exact = cis.(2π .* (temporal*spatial'))
        q = exact
        S = Matrix{ComplexF64}(I, 3, 3)
        pairs = (rows=[1,2,2,5,5,4,3], voxels=[1,2,2,3,1,1,2])
        q[5, 3] *= 7
        e2, se = HighOrderMRI.adaptive_prefix_errors(temporal, spatial, q, S, pairs; all_prefixes=false)
        loss = [abs2(exact[t,v]-cis(2π*temporal[t]*spatial[v])) for (t,v) in zip(pairs.rows,pairs.voxels)]
        @test only(e2) ≈ sum(loss)/length(loss)
        @test only(se) ≈ std(loss)/sqrt(length(loss))
        q2 = fill(0.7+0im, 5, 1);S2 = ones(ComplexF64, 3, 1)
        z2, _ = HighOrderMRI.adaptive_prefix_errors(zeros(5,1), zeros(3,1), q2, S2, pairs)
        @test only(z2) ≈ 0.09
        @test_throws BoundsError HighOrderMRI.adaptive_prefix_errors(temporal, spatial, q, S, (rows=[1,6], voxels=[1,2]))
    end
    @testset "Input validation and constant phase" begin
        d = adaptive_basis_test_problem(Float32)
        f(; kwargs...) = HighOrderMRI.adaptive_spatial_basis(d.times, d.fieldmap, d.bf, d.kspha;
            epsilon_H=0.01, kwargs...)
        for tolerance in (0, -0.1, 1, Inf, NaN)
            @test_throws ArgumentError HighOrderMRI.adaptive_spatial_basis(d.times, d.fieldmap, d.bf, d.kspha; epsilon_H=tolerance)
        end
        @test_throws ArgumentError f(max_attempts=0)
        @test_throws ArgumentError f(chunk_size=0)
        @test_throws ArgumentError f(selection_pairs=1)
        @test_throws ArgumentError f(memory_fraction=1.1)
        @test_throws ArgumentError f(snapshots=32, max_snapshots=16)
        @test_throws ArgumentError f(sample_count=32, max_samples=16)
        @test_throws DimensionMismatch HighOrderMRI.adaptive_spatial_basis(d.times, d.fieldmap, d.bf[1:end-1,:], d.kspha; epsilon_H=0.01)
        bad = copy(d.times);bad[1] = NaN32
        @test_throws ArgumentError HighOrderMRI.adaptive_spatial_basis(bad, d.fieldmap, d.bf, d.kspha; epsilon_H=0.01)
        constant = HighOrderMRI.adaptive_spatial_basis(zeros(Float32, 1, 1), zeros(Float32, 4),
            zeros(Float32, 4, 0), zeros(Float32, 0, 1, 1); epsilon_H=1e-4, chunk_size=3)
        @test constant.rank == 1
        @test constant.q*constant.S' ≈ ones(ComplexF32, 1, 4)
    end
    if CUDA.functional()
        @testset "CUDA Float32" begin
            test_adaptive_basis(CuArray, Float32)
        end
    end
end
