using Random
using Statistics

@testset "Coil compression" begin
    @testset "rank and energy selection" begin
        calibration = Matrix(Diagonal(ComplexF64[4, 3, 1, 0.1]))

        rank_transform = fit_coil_compression(calibration; coil_dim = 2, n_virtual_coils = 2)
        @test size(rank_transform) == (4, 2)
        @test rank_transform.compression_matrix' * rank_transform.compression_matrix ≈ Matrix{ComplexF64}(I, 2, 2)
        @test rank_transform.retained_energy ≈ 25 / 26.01
        @test rank_transform.singular_values ≈ [4, 3, 1, 0.1]
        @test occursin("4 => 2", sprint(show, rank_transform))

        energy_transform = fit_coil_compression(calibration; energy_threshold = 0.95)
        @test size(energy_transform) == (4, 2)
        @test energy_transform.retained_energy >= 0.95

        @test_throws ArgumentError fit_coil_compression(calibration)
        @test_throws ArgumentError fit_coil_compression(calibration; n_virtual_coils = 2, energy_threshold = 0.95)
        @test_throws ArgumentError fit_coil_compression(calibration; n_virtual_coils = 0)
        @test_throws ArgumentError fit_coil_compression(calibration; energy_threshold = 1.1)
        @test_throws ArgumentError fit_coil_compression(zeros(ComplexF64, 8, 4); n_virtual_coils = 2)
    end

    @testset "noise covariance and prewhitening" begin
        Random.seed!(20260805)
        @test noise_prewhitening_scale_factor(4e-6, 2e-6) ≈ 2.0
        @test noise_prewhitening_scale_factor(4.0f-6, 2.0f-6; receiver_bandwidth_ratio = 0.8f0) ≈ 1.6f0
        @test_throws ArgumentError noise_prewhitening_scale_factor(0.0, 2e-6)
        @test_throws ArgumentError noise_prewhitening_scale_factor(4e-6, -2e-6)
        @test_throws ArgumentError noise_prewhitening_scale_factor(4e-6, 2e-6; receiver_bandwidth_ratio = Inf)

        noise_data = randn(ComplexF64, 101, 3)
        covariance = estimate_noise_covariance(noise_data; coil_dim = 2)
        centered_noise = noise_data .- mean(noise_data; dims = 1)
        manual_covariance = centered_noise' * centered_noise / 100
        @test covariance ≈ manual_covariance
        @test covariance ≈ covariance'

        calibration = randn(ComplexF64, 40, 3)
        prescribed_covariance = ComplexF64[
            4.0 0.7+0.2im 0.1-0.1im
            0.7-0.2im 2.0 0.3+0.05im
            0.1+0.1im 0.3-0.05im 1.0
        ]
        transform = fit_coil_compression(calibration; n_virtual_coils = 2, noise_covariance = prescribed_covariance)
        compressed_noise_covariance = transform.compression_matrix' * prescribed_covariance * transform.compression_matrix
        @test compressed_noise_covariance ≈ Matrix{ComplexF64}(I, 2, 2)
        @test transform.noise_covariance ≈ prescribed_covariance
        @test occursin("noise-whitened SVD", sprint(show, transform))

        whitening_matrix = cholesky(Hermitian(prescribed_covariance)).U \ Matrix{ComplexF64}(I, 3, 3)
        reference_singular_values = svdvals(calibration * whitening_matrix)
        @test transform.singular_values ≈ reference_singular_values

        scale_factor = noise_prewhitening_scale_factor(4e-6, 2e-6; receiver_bandwidth_ratio = 0.8)
        corrected_transform = fit_coil_compression(calibration; n_virtual_coils = 2, noise_covariance = prescribed_covariance, prewhitening_scale_factor = scale_factor)
        effective_covariance = prescribed_covariance / scale_factor
        @test corrected_transform.compression_matrix' * effective_covariance * corrected_transform.compression_matrix ≈ Matrix{ComplexF64}(I, 2, 2)
        @test corrected_transform.compression_matrix' * prescribed_covariance * corrected_transform.compression_matrix ≈ scale_factor .* Matrix{ComplexF64}(I, 2, 2)
        @test corrected_transform.noise_covariance ≈ prescribed_covariance
        @test corrected_transform.prewhitening_scale_factor ≈ scale_factor
        @test occursin("prewhitening_scale_factor", sprint(show, corrected_transform))

        copied_transform = CoilCompressionTransform(copy(corrected_transform.compression_matrix), copy(corrected_transform.singular_values), 
            corrected_transform.retained_energy, copy(corrected_transform.noise_covariance))
        @test copied_transform.prewhitening_scale_factor ≈ scale_factor

        corrected_whitening_matrix = sqrt(scale_factor) .* whitening_matrix
        corrected_reference_singular_values = svdvals(calibration * corrected_whitening_matrix)
        @test corrected_transform.singular_values ≈ corrected_reference_singular_values
        @test corrected_transform.retained_energy ≈ transform.retained_energy

        @test_throws DimensionMismatch fit_coil_compression(calibration; n_virtual_coils = 2, noise_covariance = Matrix{ComplexF64}(I, 2, 2))
        @test_throws ArgumentError fit_coil_compression(
            calibration;
            n_virtual_coils = 2,
            noise_covariance = ComplexF64[
                1 1 0
                0 1 0
                0 0 1
            ],
        )
        @test_throws ArgumentError fit_coil_compression(calibration; n_virtual_coils = 2, noise_covariance = zeros(ComplexF64, 3, 3))
        @test_throws ArgumentError fit_coil_compression(calibration; n_virtual_coils = 2, prewhitening_scale_factor = 2)
        @test_throws ArgumentError fit_coil_compression(calibration; n_virtual_coils = 2, noise_covariance = prescribed_covariance, prewhitening_scale_factor = 0)
        @test_throws ArgumentError estimate_noise_covariance(zeros(ComplexF64, 1, 3))
    end

    @testset "arbitrary coil dimension" begin
        Random.seed!(20260803)
        data = randn(ComplexF64, 7, 4, 3)
        transform = fit_coil_compression(data; coil_dim = 2, n_virtual_coils = 2)
        compressed = apply_coil_compression(data, transform; coil_dim = 2)

        data_matrix = reshape(permutedims(data, (1, 3, 2)), :, 4)
        manual = permutedims(reshape(data_matrix * transform.compression_matrix, 7, 3, 2), (1, 3, 2))
        @test size(compressed) == (7, 2, 3)
        @test compressed ≈ manual
        @test_throws ArgumentError apply_coil_compression(data, transform; coil_dim = 4)
        @test_throws DimensionMismatch apply_coil_compression(randn(ComplexF64, 7, 5), transform; coil_dim = 2)
    end

    @testset "shared data and CSM transform" begin
        Random.seed!(20260804)
        data = randn(ComplexF64, 11, 4)
        csm = randn(ComplexF64, 5, 6, 2, 4)
        scale_factor = noise_prewhitening_scale_factor(2e-6, 4e-6; receiver_bandwidth_ratio = 0.9)
        compressed_data, compressed_csm, transform = compress_coils(
            data,
            csm;
            data_coil_dim = 2,
            csm_coil_dim = 4,
            n_virtual_coils = 3,
            noise_covariance = ComplexF64[
                2.0 0.1 0.0 0.0
                0.1 1.5 0.2 0.0
                0.0 0.2 1.2 0.1
                0.0 0.0 0.1 1.0
            ],
            prewhitening_scale_factor = scale_factor,
        )
        @test size(compressed_data) == (11, 3)
        @test size(compressed_csm) == (5, 6, 2, 3)
        @test transform.prewhitening_scale_factor ≈ scale_factor
        @test compressed_data ≈ apply_coil_compression(data, transform; coil_dim = 2)
        @test compressed_csm ≈ apply_coil_compression(csm, transform; coil_dim = 4)
    end

    @testset "low-rank operator compatibility" begin
        grid, kspha, times, fieldmap, csm, mask, recon_terms = highorder_lowrank_test_data()
        n_sample, n_dynamic = size(times)
        n_coil = size(csm, 3)
        image = Complex{T}.(reshape(T.(1:prod(grid.matrixSize)), :))

        original_operator = HighOrderLowRankOp(grid, kspha, times; fieldmap, csm,
            mask, recon_terms, L_rank = 2, rsvd_seed = 0, rsvd_finalize = :gram)
        original_data = reshape(original_operator * image, n_sample * n_dynamic, n_coil)

        full_transform = fit_coil_compression(original_data; n_virtual_coils = n_coil)
        full_data = apply_coil_compression(original_data, full_transform; coil_dim = 2)
        full_csm = apply_coil_compression(csm, full_transform; coil_dim = 3)
        full_operator = HighOrderLowRankOp(grid, kspha, times; fieldmap, csm = full_csm,
            mask, recon_terms, L_rank = 2, rsvd_seed = 0, rsvd_finalize = :gram)
        @test vec(full_data) ≈ full_operator * image rtol = 1.0f-4 atol = 1.0f-5

        # Compare the complete encoding matrices: a random adjoint probe can
        # land near the operator null space and amplify harmless setup error.
        image_basis = Matrix{Complex{T}}(I, size(original_operator, 2), size(original_operator, 2))
        original_matrix = mapreduce(column -> original_operator * column, hcat, eachcol(image_basis))
        full_matrix = mapreduce(column -> full_operator * column, hcat, eachcol(image_basis))
        physical_to_virtual = kron(
            transpose(full_transform.compression_matrix),
            Matrix{Complex{T}}(I, n_sample * n_dynamic, n_sample * n_dynamic),
        )
        @test full_matrix ≈ physical_to_virtual * original_matrix rtol = 1.0f-4 atol = 1.0f-5
        @test adjoint(full_operator) * (full_operator * image) ≈ adjoint(original_operator) * (original_operator * image) rtol = 1.0f-4 atol = 1.0f-5

        compressed_data, compressed_csm, _ = compress_coils(original_data, csm; n_virtual_coils = 1)
        compressed_operator = HighOrderLowRankOp(grid, kspha, times; fieldmap, csm = compressed_csm,
            mask, recon_terms, L_rank = 2, rsvd_seed = 0, rsvd_finalize = :gram)

        @test vec(compressed_data) ≈ compressed_operator * image rtol = 1.0f-4 atol = 1.0f-5
        close(original_operator)
        close(full_operator)
        close(compressed_operator)
    end
end
