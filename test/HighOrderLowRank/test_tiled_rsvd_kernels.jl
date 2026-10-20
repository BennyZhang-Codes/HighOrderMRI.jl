@testset "tiled rSVD kernels against dense products" begin
    if CUDA.functional()
        rng = MersenneTwister(1984)
        for FT in (Float32, Float64), M in (1, 12)
            nSam, nVox = 67, 263
            times = FT(0.02) .* rand(rng, FT, nSam)
            fieldmap = randn(rng, FT, nVox)
            bf = FT(0.02) .* randn(rng, FT, nVox, M)
            kspha = randn(rng, FT, M, nSam)
            E = cis.(FT(2π) .* (times * transpose(fieldmap) + transpose(kspha) * transpose(bf)))
            td, fd, bd = CuArray(times), CuArray(fieldmap), CuArray(bf)
            tolerance = FT === Float32 ? FT(1e-4) : FT(1e-12)
            for L in (1, 11, 17, 33)
                omega = randn(rng, Complex{FT}, nVox, L)
                Q = randn(rng, Complex{FT}, nSam, L)
                wd, qd = CuArray(omega), CuArray(Q)
                W = CUDA.zeros(Complex{FT}, nSam, L)
                B = CUDA.zeros(Complex{FT}, nVox, L)
                for transposed in (false, true)
                    kd = CuArray(transposed ? permutedims(kspha) : kspha)
                    for partitions in (1, 3, 5)
                        partials = CuArray{Complex{FT}}(undef, nSam, L, partitions)
                        HighOrderMRI.run_kernel_rsvd_forward_tiled!(
                            W, partials, wd, td, fd, bd, kd; kspha_transposed=transposed)
                        @test norm(Array(W) - E * omega) / norm(E * omega) < tolerance
                    end
                    for lanes in (1, 2, 4, 8, 16)
                        HighOrderMRI.run_kernel_rsvd_adjoint!(
                            B, qd, td, fd, bd, kd; threads=128,
                            kspha_transposed=transposed, lanes_per_voxel=lanes)
                        @test norm(Array(B) - adjoint(E) * Q) / norm(adjoint(E) * Q) < tolerance
                    end
                    @test_throws ArgumentError HighOrderMRI.run_kernel_rsvd_adjoint!(
                        B, qd, td, fd, bd, kd; lanes_per_voxel=3,
                        kspha_transposed=transposed)
                end
            end
        end
    end
end
