@testset "Image metrics" begin
    reference = ComplexF64[1 + 1im 2 - 1im; -0.5 + 0.25im 3 + 2im]

    @testset "raw complex NRMSE" begin
        @test raw_complex_nrmse(reference, reference) == 0

        reconstruction = 2im .* reference
        expected = norm(vec(reconstruction) - vec(reference)) / norm(vec(reference))
        @test raw_complex_nrmse(reconstruction, reference) ≈ expected
        @test raw_complex_nrmse(reconstruction, reference) > 0

        @test_throws DimensionMismatch raw_complex_nrmse(zeros(2), zeros(2, 1))
        @test_throws DomainError raw_complex_nrmse(ones(2), zeros(2))
        @test_throws DomainError raw_complex_nrmse([NaN, 1.0], ones(2))
    end

    @testset "complex alignment" begin
        reconstruction = 2im .* reference
        @test complex_alignment_scale(reconstruction, reference) ≈ -0.5im
        @test aligned_complex_nrmse(reconstruction, reference) ≈ 0 atol=1e-14
        @test aligned_complex_nrmse(zeros(size(reference)), reference) == 1
    end

    @testset "magnitude NRMSE" begin
        @test magnitude_nrmse(reference, reference) == 0
        @test magnitude_nrmse(cis(0.7) .* reference, reference) ≈ 0 atol=1e-14
        @test magnitude_nrmse(1.1 .* reference, reference) ≈ 0.1
        @test magnitude_nrmse(2im .* reference, reference; align=true) ≈ 0 atol=1e-14
    end

    @testset "magnitude SSIM" begin
        image_reference = reshape(collect(range(0.0, 1.0; length=16^2)), 16, 16)
        @test magnitude_ssim(image_reference, image_reference) ≈ 1
        @test magnitude_ssim(cis(0.9) .* image_reference, image_reference) ≈ 1
        @test magnitude_ssim(1.5 .* image_reference, image_reference) < 1
        @test magnitude_ssim(
            1.5 .* image_reference,
            image_reference;
            align=true,
        ) ≈ 1
    end

    @testset "reference-first compatibility API" begin
        reconstruction = 2im .* reference
        @test HO_NRMSE(reference, reconstruction) ≈
              raw_complex_nrmse(reconstruction, reference)
        @test HO_NRMSE(reference, reconstruction; scale=true) ≈ 0 atol=1e-14
        @test HO_MSE(reference, reference) == 0
        @test HO_RMSE(reference, reference) == 0
    end
end
