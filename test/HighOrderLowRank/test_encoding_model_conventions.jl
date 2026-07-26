function encoding_reference_forward(
    grid,
    kspha::AbstractArray{T,3},
    times::AbstractMatrix{T},
    fieldmap::AbstractArray{T},
    csm::AbstractArray{Complex{T}},
    mask::AbstractArray{Bool},
    x::AbstractVector{Complex{T}},
) where T<:AbstractFloat
    nTerm, nSam, nDyn = size(kspha)
    nCha = size(csm, ndims(csm))
    mask_flat = vec(mask)
    nVox = count(mask_flat)

    bf = HighOrderMRI.basisfunc_spha(
        grid.x[mask_flat],
        grid.y[mask_flat],
        grid.z[mask_flat],
        collect(1:nTerm),
    )
    fieldmap_masked = vec(fieldmap)[mask_flat]
    csm_masked = reshape(csm, :, nCha)[mask_flat, :]
    x_masked = x[mask_flat]

    out = zeros(Complex{T}, nSam, nDyn, nCha)
    for dyn = 1:nDyn
        kspha_dyn = @view kspha[:, :, dyn]
        times_dyn = @view times[:, dyn]
        phase =
            bf * kspha_dyn .+
            fieldmap_masked * transpose(times_dyn)
        encoding = cis.(T(2π) .* phase)

        for cha = 1:nCha
            @views out[:, dyn, cha] .=
                transpose(encoding) * (x_masked .* csm_masked[:, cha]) /
                sqrt(T(nVox))
        end
    end

    return vec(out)
end


function encoding_convention_problem(; nCha::Int=2, nDyn::Int=2)
    nX, nY, nZ = 12, 13, 1
    nSam, nTerm = 6, 9
    grid = Grid(nX, nY, nZ, T(0.8), T(1.3), one(T))

    kspha = zeros(T, nTerm, nSam, nDyn)
    times = zeros(T, nSam, nDyn)
    for dyn = 1:nDyn
        times[:, dyn] .= range(zero(T), T(1.2e-3); length=nSam)
        kspha[1, :, dyn] .= T(1e-2 * (dyn - 1))
        kspha[2, :, dyn] .= range(T(-0.04), T(0.05); length=nSam)
        kspha[3, :, dyn] .=
            range(T(0.03), T(-0.02); length=nSam) .* T(dyn)
        kspha[5, :, dyn] .=
            range(T(-5e-4), T(5e-4); length=nSam)
        kspha[9, :, dyn] .= T(2e-4 * dyn)
    end

    fieldmap = reshape(
        range(T(-20), T(20); length=nX * nY),
        nX,
        nY,
    )
    csm = Array{Complex{T}}(undef, nX, nY, nCha)
    for cha = 1:nCha
        magnitude = one(T) .+ T(0.03 * cha) .* reshape(
            range(-one(T), one(T); length=nX * nY),
            nX,
            nY,
        )
        phase = T(0.05 * cha) .* reshape(
            range(-one(T), one(T); length=nX * nY),
            nX,
            nY,
        )
        @views csm[:, :, cha] .= magnitude .* cis.(phase)
    end

    mask = trues(nX, nY)
    mask[1, 1] = false
    mask[end, end] = false
    x = complex.(
        collect(range(T(0.1), one(T); length=nX * nY)),
        collect(range(T(-0.2), T(0.2); length=nX * nY)),
    )

    return (; grid, kspha, times, fieldmap, csm, mask, x)
end


@testset "Encoding model conventions" begin
    @testset "Grid coordinates and spherical-harmonic order" begin
        grid = Grid(6, 5, 3, T(0.8), T(1.25), T(2.0))

        expected_x = T.((collect(1:6) .- T(3.5)) .* T(0.8))
        expected_y = T.((collect(1:5) .- T(3.0)) .* T(1.25))
        expected_z = T.((collect(1:3) .- T(2.0)) .* T(2.0))
        @test unique(grid.x) ≈ expected_x
        @test unique(grid.y) ≈ expected_y
        @test unique(grid.z) ≈ expected_z
        @test sum(expected_x) ≈ zero(T) atol=eps(T)
        @test sum(expected_y) ≈ zero(T) atol=eps(T)
        @test sum(expected_z) ≈ zero(T) atol=eps(T)

        x = T[2]
        y = T[3]
        z = T[5]
        basis = HighOrderMRI.basisfunc_spha(x, y, z, collect(1:9))
        expected_basis = T[
            1,
            2,
            3,
            5,
            6,
            15,
            3 * 25 - (4 + 9 + 25),
            10,
            4 - 9,
        ]
        @test vec(basis) == expected_basis
    end

    @testset "recon_terms is validated and input preserving" begin
        kspha = reshape(T.(1:(9 * 5)), 9, 5)
        original = copy(kspha)
        k_nominal = fill(T(-3), 3, 5)

        @test HighOrderMRI.prep_kspha(kspha, k_nominal, 9) == original
        @test kspha == original

        selected = HighOrderMRI.prep_kspha(
            kspha,
            k_nominal,
            9;
            recon_terms="010",
        )
        @test all(iszero, @view selected[1, :])
        @test view(selected, 2:4, :) == view(original, 2:4, :)
        @test all(iszero, @view selected[5:9, :])
        @test kspha == original

        no_zeroth_correction = HighOrderMRI.prep_kspha(
            kspha,
            k_nominal,
            9;
            recon_terms="011",
        )
        @test all(iszero, @view no_zeroth_correction[1, :])
        @test view(no_zeroth_correction, 2:9, :) ==
              view(original, 2:9, :)

        nominal = HighOrderMRI.prep_kspha(
            kspha,
            k_nominal,
            9;
            recon_terms="000",
        )
        @test view(nominal, 2:4, :) == k_nominal
        @test_throws ArgumentError HighOrderMRI.prep_kspha(
            kspha,
            k_nominal,
            9;
            recon_terms="11",
        )
        @test_throws ArgumentError HighOrderMRI.prep_kspha(
            kspha,
            k_nominal,
            9;
            recon_terms="012",
        )
    end

    @testset "LowRank raw complex convention against dense reference" begin
        data = encoding_convention_problem()
        nSam = size(data.kspha, 2)

        # Full row rank removes LowRank truncation from this convention test.
        op = HighOrderLowRankOp(
            data.grid,
            data.kspha,
            data.times;
            fieldmap=data.fieldmap,
            csm=data.csm,
            mask=data.mask,
            L_rank=nSam,
            rsvd_seed=2026,
            rsvd_oversample=0,
            rsvd_finalize=:gram,
            rsvd_backend=:chunked,
            shared_rank_max=nSam * size(data.kspha, 3),
            shared_basis_tol=T(1e-5),
            nfft_center_correction=true,
        )

        expected_nodes = similar(op.nfft_traj)
        expected_nodes[1, :, :] .=
            -data.kspha[2, :, :] .* T(data.grid.Δx)
        expected_nodes[2, :, :] .=
            -data.kspha[3, :, :] .* T(data.grid.Δy)
        @test op.nfft_traj ≈ expected_nodes

        y_reference = encoding_reference_forward(
            data.grid,
            data.kspha,
            data.times,
            data.fieldmap,
            data.csm,
            data.mask,
            data.x,
        )
        y_lowrank = op * data.x
        raw_complex_error =
            norm(y_lowrank - y_reference) /
            max(norm(y_reference), eps(T))
        @test raw_complex_error < T(2e-3)

        y_probe = complex.(
            collect(range(T(-0.5), T(0.5); length=length(y_reference))),
            collect(range(T(0.2), T(-0.2); length=length(y_reference))),
        )
        lhs = dot(y_lowrank, y_probe)
        rhs = dot(data.x, adjoint(op) * y_probe)
        adjoint_error =
            abs(lhs - rhs) / max(abs(lhs), abs(rhs), eps(T))
        @test adjoint_error < T(1e-4)
    end

    @testset "Explicit CUDA operator against dense reference" begin
        if CUDA.functional()
            data = encoding_convention_problem(; nCha=32, nDyn=1)
            gpu_id = Int(CUDA.deviceid(CUDA.device()))
            op = HighOrderOp_Kernel(
                data.grid,
                data.kspha[:, :, 1],
                data.times[:, 1];
                fieldmap=data.fieldmap,
                csm=data.csm,
                mask=data.mask,
                gpus=[gpu_id],
            )

            y_reference = encoding_reference_forward(
                data.grid,
                data.kspha,
                data.times,
                data.fieldmap,
                data.csm,
                data.mask,
                data.x,
            )
            x_gpu = CuArray(data.x)
            y_explicit = Array(op * x_gpu)
            raw_complex_error =
                norm(y_explicit - y_reference) /
                max(norm(y_reference), eps(T))
            @test raw_complex_error < T(2e-5)

            y_probe = complex.(
                collect(range(T(-0.5), T(0.5); length=length(y_reference))),
                collect(range(T(0.2), T(-0.2); length=length(y_reference))),
            )
            y_probe_gpu = CuArray(y_probe)
            lhs = dot(y_explicit, y_probe)
            rhs = dot(data.x, Array(adjoint(op) * y_probe_gpu))
            adjoint_error =
                abs(lhs - rhs) / max(abs(lhs), abs(rhs), eps(T))
            @test adjoint_error < T(1e-4)
        end
    end
end
