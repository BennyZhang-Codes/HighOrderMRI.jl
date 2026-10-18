using Random

function joint_basis_test_problem(::Type{T}) where T
    rng = Random.Xoshiro(731)
    nSam, nVox, nDyn, nTerm = 67, 193, 3, 3
    times = T.(rand(rng, nSam, nDyn) .* 0.02)
    fieldmap = T.(randn(rng, nVox) .* 2)
    bf = T.(randn(rng, nVox, nTerm) .* 0.1)
    kspha = T.(randn(rng, nTerm, nSam, nDyn) .* 0.1)
    temporal = hcat(vec(Float64.(times)), permutedims(reshape(Float64.(kspha), nTerm, :)))
    spatial = hcat(Float64.(fieldmap), Float64.(bf))
    exact = cis.(2π .* (temporal * spatial'))
    return (; times, fieldmap, bf, kspha, exact)
end

function test_joint_factors(arrayType, ::Type{T}) where T
    data = joint_basis_test_problem(T)
    report_ref = Ref{Any}()
    original = deepcopy(data)
    q = basis = nothing
    try
        q, basis, report = HighOrderMRI.joint_spatial_basis(
            data.times, data.fieldmap, data.bf, data.kspha;
            arrayType, rank_max=32, tol=T(1e-2), snapshots=64,
            sample_count=64, chunk_size=23, seed=1234, roi_tol=T(1e-2), report_ref,
        )
        approximation = Array(q) * Array(basis)'
        @test q isa arrayType
        @test basis isa arrayType
        @test eltype(q) == eltype(basis) == Complex{T}
        @test size(q) == (201, report.rank)
        @test size(basis) == (193, report.rank)
        @test report_ref[] === report
        @test report.passed && report.stage === :audit
        @test maximum(report.validation_errors) <= report.selection_tolerance < report.tolerance
        @test maximum(report.audit_errors) <= 1e-2
        @test maximum(report.roi_errors) <= 1e-2
        @test norm(approximation - data.exact) / norm(data.exact) < 1e-2
        @test norm(Array(basis)' * Array(basis) - I) < 100eps(T) * report.rank
        @test isempty(intersect(report.fit_indices, report.validation_voxels))
        @test isempty(intersect(report.fit_indices, report.audit_voxels))
        @test isempty(intersect(report.snapshot_indices, vec(report.validation_rows)))
        @test isempty(intersect(vec(report.validation_rows), vec(report.audit_rows)))
        @test report.disjoint_times
        for d in 1:3
            rows = report.audit_rows[:, d]
            columns = report.audit_voxels
            exact_error = norm(approximation[rows, columns] - data.exact[rows, columns]) /
                sqrt(length(rows) * length(columns))
            @test report.audit_errors[d] ≈ exact_error rtol=1e-4 atol=10eps(T)
        end
        @test data == original

        # Independent complex input probes check the adjoint and all Gram
        # cross terms, including ranks on opposite sides of a block boundary.
        rng = Random.Xoshiro(937)
        x = randn(rng, ComplexF64, 193)
        y = randn(rng, ComplexF64, 201)
        forward = approximation * x
        backward = approximation' * y
        @test dot(forward, y) ≈ dot(x, backward) rtol=1e-12
        @test norm(approximation' * forward - data.exact' * (data.exact * x)) /
            norm(data.exact' * (data.exact * x)) < 2e-2
    finally
        HighOrderMRI.joint_synchronize(arrayType)
        HighOrderMRI.joint_release!(q)
        HighOrderMRI.joint_release!(basis)
    end

    # A failed bounded search must return diagnostics, never partial factors.
    for repeat in 1:2
        @test_throws ErrorException HighOrderMRI.joint_spatial_basis(
            data.times, data.fieldmap, data.bf, data.kspha;
            arrayType, rank_max=1, tol=T(1e-5), snapshots=16,
            sample_count=16, chunk_size=7, report_ref,
        )
        @test !report_ref[].passed && report_ref[].stage === :rank_limit
    end

    # With this seed, sample 2 is used for development and sample 3 for audit.
    # Only the unseen audit row has a nonconstant phase: developing a rank-one
    # model succeeds, but returning its completed factors would be incorrect.
    @test_throws ErrorException HighOrderMRI.joint_spatial_basis(
        reshape(T[0, 0, 1, 0], 4, 1), collect(range(-one(T), one(T); length=193)),
        zeros(T, 193, 0), zeros(T, 0, 4, 1);
        arrayType, tol=T(1e-2), sample_count=32, chunk_size=7, seed=1234, report_ref,
    )
    @test report_ref[].stage === :audit && !report_ref[].passed
    @test maximum(report_ref[].validation_errors) < 1e-2
    @test maximum(report_ref[].audit_errors) > 1e-2

    # Constant phase has one direction, including repeated snapshot geometry
    # and the smallest supported spatial partition with no residual terms.
    q, basis, report = HighOrderMRI.joint_spatial_basis(
        zeros(T, 1, 1), zeros(T, 4), zeros(T, 4, 0), zeros(T, 0, 1, 1);
        arrayType, tol=T(1e-5), chunk_size=3,
    )
    try
        @test report.rank == report.snapshots == 1
        @test report.passed && !report.disjoint_times
        @test Array(q) * Array(basis)' ≈ ones(Complex{T}, 1, 4) atol=10eps(T)
    finally
        HighOrderMRI.joint_synchronize(arrayType)
        HighOrderMRI.joint_release!(q)
        HighOrderMRI.joint_release!(basis)
    end
end

@testset "Joint shared spatial basis" begin
    @testset "CPU $T" for T in (Float32, Float64)
        test_joint_factors(Array, T)
    end

    @testset "Input validation" begin
        d = joint_basis_test_problem(Float32)
        f = HighOrderMRI.joint_spatial_basis
        @test_throws DimensionMismatch f(d.times, d.fieldmap, d.bf[1:end-1, :], d.kspha)
        @test_throws DimensionMismatch f(d.times, d.fieldmap, d.bf, d.kspha[:, 1:end-1, :])
        @test_throws ArgumentError f(d.times, d.fieldmap, d.bf, d.kspha; tol=Float32(NaN))
        @test_throws ArgumentError f(d.times, d.fieldmap, d.bf, d.kspha; roi_tol=-1f0)
        @test_throws ArgumentError f(d.times, d.fieldmap, d.bf, d.kspha; rank_max=0)
        @test_throws ArgumentError f(d.times, d.fieldmap, d.bf, d.kspha; chunk_size=0)
        d.fieldmap[1] = Inf32
        @test_throws ArgumentError f(d.times, d.fieldmap, d.bf, d.kspha)
    end

    @testset "Independent high-B0 gate" begin
        times = repeat(reshape(collect(range(0.0, 0.1; length=131)), :, 1), 1, 4)
        fieldmap = zeros(4097)
        bf = zeros(4097, 0)
        kspha = zeros(0, 131, 4)
        report_ref = Ref{Any}()
        seed = 8
        order = randperm(Random.Xoshiro(seed), length(fieldmap))
        available = order[length(fieldmap) ÷ 2 + 1:end]
        n_random = min(1024, length(available) ÷ 2)
        audit_order = randperm(Random.Xoshiro(seed + 1), length(available))
        # Keep the isolated outlier outside the fit and both random checks;
        # its exact index may change with Julia's seeded permutation.
        outliers = setdiff(available[n_random+1:end], available[audit_order[1:n_random]])
        outlier = first(outliers)
        fieldmap[outlier] = 100
        q, basis, report = @test_logs (:warn, r"high-B0 region") HighOrderMRI.joint_spatial_basis(
            times, fieldmap, bf, kspha; rank_max=1, tol=0.08,
            snapshots=16, sample_count=32, chunk_size=77, seed, report_ref,
        )
        @test outlier in report.validation_roi && outlier in report.audit_roi
        @test report.passed && maximum(report.roi_errors) > 0.08
        @test_throws ErrorException HighOrderMRI.joint_spatial_basis(
            times, fieldmap, bf, kspha; rank_max=1, tol=0.08, roi_tol=0.08,
            snapshots=16, sample_count=32, chunk_size=77, seed, report_ref,
        )
        @test !report_ref[].passed
    end

    if CUDA.functional()
        @testset "CUDA $T" for T in (Float32, Float64)
            test_joint_factors(CuArray, T)
        end
    end
end
