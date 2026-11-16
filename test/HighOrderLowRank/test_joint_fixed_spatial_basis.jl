function test_joint_fixed_basis(arrayType, ::Type{T}) where T
    d = adaptive_basis_test_problem(T)
    original = deepcopy(d)
    build(; kwargs...) = HighOrderMRI.joint_fixed_spatial_basis(
        d.times, d.fieldmap, d.bf, d.kspha; arrayType, rank=4,
        snapshots=20, sample_count=32, fit_pool_size=47, chunk_size=31, kwargs...)
    snapshot = build()
    @test snapshot.rank == snapshot.parent_rank == 4
    @test snapshot.ordering === :snapshot
    @test snapshot.q isa Matrix{Complex{T}} && snapshot.S isa Matrix{Complex{T}}
    @test snapshot.construction_status === :computed
    @test snapshot.empirical_status === :not_evaluated
    @test snapshot.certification_status === snapshot.external_audit_status === :not_requested
    # Independent FP64 phase matrix and least-squares solve, including the
    # conjugation needed by H ≈ q*S'. This does not reuse the fitting map.
    reference_q = (ComplexF64.(snapshot.S[snapshot.fit_indices, :]) \
        d.exact[:, snapshot.fit_indices]')'
    @test norm(snapshot.q-reference_q)/norm(reference_q) < 100eps(T)
    @test norm(ComplexF64.(snapshot.S)'*snapshot.S-I) < 100eps(T)*4

    parent = HighOrderMRI.joint_fixed_spatial_basis(d.times, d.fieldmap, d.bf, d.kspha;
        arrayType, rank=8, parent_rank=8, ordering=:coefficient,
        snapshots=20, sample_count=32, fit_pool_size=47, chunk_size=31)
    compressed = build(ordering=:coefficient, parent_rank=8)
    @test parent.snapshot_indices == compressed.snapshot_indices
    @test parent.fit_indices == compressed.fit_indices
    @test compressed.parent_rank == 8 && compressed.rank == 4
    full = ComplexF64.(parent.q)*ComplexF64.(parent.S)'
    prediction = ComplexF64.(compressed.q)*ComplexF64.(compressed.S)'
    spectrum = svd(full)
    optimal = spectrum.U[:, 1:4]*Diagonal(spectrum.S[1:4])*spectrum.V[:, 1:4]'
    @test norm(prediction-optimal)/norm(full) < 200eps(T)
    @test norm(prediction-full)/norm(full) ≈ norm(spectrum.S[5:end])/norm(spectrum.S) rtol=2e-4 atol=200eps(T)
    @test d == original
end

@testset "Fixed Joint controls" begin
    @testset "CPU $T" for T in (Float32, Float64)
        test_joint_fixed_basis(Array, T)
    end
    d = adaptive_basis_test_problem(Float32)
    f(; kwargs...) = HighOrderMRI.joint_fixed_spatial_basis(
        d.times, d.fieldmap, d.bf, d.kspha; kwargs...)
    @test_throws ArgumentError f(rank=0)
    @test_throws ArgumentError f(rank=4, parent_rank=3)
    @test_throws ArgumentError f(rank=4, parent_rank=8)
    @test_throws ArgumentError f(rank=4, ordering=:unknown)
    @test_throws ArgumentError f(rank=4, sample_count=3)
    @test_throws DimensionMismatch HighOrderMRI.joint_fixed_spatial_basis(
        d.times, d.fieldmap, d.bf[1:end-1, :], d.kspha; rank=4)
    times, fieldmap = zeros(Float32, 1, 1), zeros(Float32, 8)
    bf, coefficients = zeros(Float32, 8, 0), zeros(Float32, 0, 1, 1)
    constant = HighOrderMRI.joint_fixed_spatial_basis(times, fieldmap, bf, coefficients; rank=1)
    @test constant.q*constant.S' ≈ ones(ComplexF32, 1, 8)
    @test_throws ErrorException HighOrderMRI.joint_fixed_spatial_basis(
        zeros(Float32, 4, 2), fieldmap, bf, zeros(Float32, 0, 4, 2); rank=2)
    if CUDA.functional()
        @testset "CUDA Float32" begin
            test_joint_fixed_basis(CuArray, Float32)
        end
    end
end
