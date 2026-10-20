using Test, LinearAlgebra, Random, AbstractNFFTs, NFFT, CUDA

function concomitant_test_problem(
    T           :: Type{<:AbstractFloat}                                              ;
    nTerm       :: Int                       = 16                                     ,
    nZ          :: Int                       = 10                                     ,
    )
    rng = Random.Xoshiro(723)
    grid = Grid(12, 13, nZ, T(0.8), T(1.3), T(1.1))
    nZ == 1 && (grid.z .= T(0.4))
    nPixel = prod(grid.matrixSize)
    nSam, nDyn, nCha = 6, 2, 2
    mask = trues(grid.matrixSize)
    mask[1] = mask[end] = false
    kspha = randn(rng, T, nTerm, nSam, nDyn) .* T(1e-5)
    kspha[1:4, :, :] .*= T(1000)
    times = rand(rng, T, nSam, nDyn) .* T(1e-3)
    fieldmap = randn(rng, T, grid.matrixSize...) .* T(5)
    csm = randn(rng, Complex{T}, grid.matrixSize..., nCha)
    concomitant = randn(rng, T, 4, nSam, nDyn) .* T(1e-3)
    image = randn(rng, Complex{T}, nPixel)
    signal = randn(rng, Complex{T}, nSam * nDyn * nCha)
    return (; grid, kspha, times, fieldmap, csm, mask, concomitant, image, signal)
end

function concomitant_dense_encoding(data, concomitant=data.concomitant)
    T = eltype(data.times)
    nTerm, nSam, nDyn = size(data.kspha)
    nCha = size(data.csm, 4)
    grid = data.grid
    nPixel = prod(grid.matrixSize)
    bf = basisfunc_spha(grid.x, grid.y, grid.z, collect(1:nTerm))
    encoding = zeros(Complex{T}, nSam * nDyn * nCha, nPixel)
    for cha = 1:nCha, dyn = 1:nDyn
        phase = bf * data.kspha[:, :, dyn] .+ vec(data.fieldmap) * transpose(data.times[:, dyn])
        if !isnothing(concomitant)
            for sam = 1:nSam
                phase[:, sam] .+= concomitant[1, sam, dyn] .* grid.z .^ 2 .+
                    concomitant[2, sam, dyn] .* (grid.x .^ 2 .+ grid.y .^ 2) .+
                    concomitant[3, sam, dyn] .* grid.x .* grid.z .+
                    concomitant[4, sam, dyn] .* grid.y .* grid.z
            end
        end
        rows = (cha - 1) * nSam * nDyn + (dyn - 1) * nSam .+ (1:nSam)
        encoding[rows, :] = transpose(cis.(T(2π) .* phase)) .*
            transpose(reshape(data.csm, nPixel, nCha)[:, cha]) ./ sqrt(T(count(data.mask)))
    end
    encoding[:, .!vec(data.mask)] .= 0
    return encoding
end

function concomitant_nfft_reference(data, op)
    # Keep the existing NFFT interpolation error out of the phase-model test.
    # Each Fourier row comes from an adjoint impulse; all residual phases and
    # coil/mask factors come independently from the dense physical encoding.
    T = eltype(data.times)
    nSam, nDyn = size(data.times)
    nCha = size(data.csm, 4)
    grid = data.grid
    nDim = grid.nZ == 1 ? 2 : 3
    coordinates = hcat(grid.x, grid.y, grid.z)[:, 1:nDim]
    offsets = T[iseven(n) ? -0.5 : 0 for n in grid.matrixSize][1:nDim]
    spacing = T[grid.Δx, grid.Δy, grid.Δz][1:nDim]
    encoding = concomitant_dense_encoding(data)
    shape = grid.nZ == 1 ? (grid.nX, grid.nY) : grid.matrixSize
    image = zeros(Complex{T}, shape)
    for dyn = 1:nDyn
        plan = op.basis_layout === :shared ? op.nfftplan : op.readout_plans[dyn]
        signal = zeros(Complex{T}, op.basis_layout === :shared ? nSam * nDyn : nSam)
        for sam = 1:nSam
            fill!(signal, 0)
            index = op.basis_layout === :shared ? (dyn - 1) * nSam + sam : sam
            signal[index] = 1
            mul!(image, adjoint(plan), signal)
            k = data.kspha[2:1+nDim, sam, dyn]
            phase = coordinates * k
            correction = cis(T(2π) * sum((-k .* spacing) .* offsets))
            factor = conj.(vec(image)) .* cis.(-T(2π) .* phase) .* correction
            for cha = 1:nCha
                row = (cha - 1) * nSam * nDyn + (dyn - 1) * nSam + sam
                encoding[row, :] .*= factor
            end
        end
    end
    return encoding
end

@testset "Optional concomitant phase" begin
    previous_backend = AbstractNFFTs.active_backend()
    AbstractNFFTs.set_active_backend!(NFFT.backend())
    try
        @test basisfunc_spha([1., 2.], [3., 4.], [5., 6.], 17) == [-24., -96.]
        for T in (Float32, Float64), nTerm in (9, 16)
            data = concomitant_test_problem(T; nTerm)
            nSam = size(data.times, 1)
            kspha = reshape(data.kspha, nTerm, :)
            times = vec(data.times)
            concomitant = Float64.(reshape(data.concomitant, 4, :))
            original = copy(concomitant)
            common = (; fieldmap=data.fieldmap, csm=data.csm, mask=data.mask)
            reference = concomitant_dense_encoding(data)
            op = HighOrderOp(data.grid, kspha, times;
                common..., concomitant, arrayType=Array, nBlock=3)
            tolerance = T === Float64 ? 1e-11 : 1e-5
            @test norm(op * data.image - reference * data.image) / norm(reference * data.image) < tolerance
            @test norm(op' * data.signal - reference' * data.signal) / norm(reference' * data.signal) < tolerance
            @test concomitant == original

            for term = 1:4
                isolated = zeros(T, size(data.concomitant))
                isolated[term, :, :] .= data.concomitant[term, :, :]
                op = HighOrderOp(data.grid, kspha, times;
                    common..., concomitant=reshape(isolated, 4, :), arrayType=Array)
                reference = concomitant_dense_encoding(data, isolated)
                @test norm(op * data.image - reference * data.image) / norm(reference * data.image) < tolerance
            end
            default = HighOrderOp(data.grid, kspha, times; common..., arrayType=Array)
            absent = HighOrderOp(data.grid, kspha, times; common..., concomitant=nothing, arrayType=Array)
            @test default * data.image == absent * data.image
            @test_throws DimensionMismatch HighOrderOp(data.grid, kspha, times;
                common..., concomitant=concomitant[1:3, :], arrayType=Array)
            @test_throws ArgumentError HighOrderOp(data.grid, kspha, times;
                common..., concomitant=complex.(concomitant), arrayType=Array)
            derivative = HighOrderOp(data.grid, kspha, times;
                common..., kspha_dt=kspha, arrayType=Array)
            derivative_c = HighOrderOp(data.grid, kspha, times;
                common..., concomitant, kspha_dt=kspha, arrayType=Array)
            @test derivative * data.image == derivative_c * data.image
            @test derivative' * data.signal == derivative_c' * data.signal

            flags = repeat("0", nTerm == 9 ? 3 : 4)
            selected = merge(data, (; kspha=zero(data.kspha)))
            op = HighOrderOp(data.grid, kspha, times;
                common..., concomitant, recon_terms=flags,
                k_nominal=zeros(T, 3, length(times)), arrayType=Array)
            reference = concomitant_dense_encoding(selected)
            @test norm(op * data.image - reference * data.image) / norm(reference * data.image) < tolerance

            lowrank = (; common..., L_rank=nSam, rsvd_oversample=0, rsvd_seed=1234,
                rsvd_backend=:chunked, shared_rank_max=2nSam, shared_basis_tol=T(1e-10))
            for layout in (:shared, :per_readout), finalize in (:svd, :gram)
                op = HighOrderLowRankOp(data.grid, data.kspha, data.times;
                    lowrank..., basis_layout=layout, rsvd_finalize=finalize, concomitant=data.concomitant)
                reference = concomitant_nfft_reference(data, op)
                forward, backward = op * data.image, op' * data.signal
                tolerance = T === Float64 ? 1e-9 : 2e-4
                @test norm(forward - reference * data.image) / norm(reference * data.image) < tolerance
                @test norm(backward - reference' * data.signal) / norm(reference' * data.signal) < tolerance
                @test abs(dot(forward, data.signal) - dot(data.image, backward)) /
                    max(abs(dot(forward, data.signal)), eps(T)) < (T === Float64 ? 1e-10 : 2e-5)
                @test all(iszero, backward[.!vec(data.mask)])
                close(op)
            end
            default = HighOrderLowRankOp(data.grid, data.kspha, data.times; lowrank...)
            absent = HighOrderLowRankOp(data.grid, data.kspha, data.times; lowrank..., concomitant=nothing)
            @test default.q == absent.q
            @test default.basis == absent.basis
            close(default)
            close(absent)
            single = HighOrderLowRankOp(data.grid, data.kspha[:, :, 1], data.times[:, 1];
                lowrank..., concomitant=data.concomitant[:, :, 1])
            @test size(single, 1) == nSam * size(data.csm, 4)
            single3 = HighOrderLowRankOp(data.grid, data.kspha[:, :, 1:1], data.times[:, 1:1];
                lowrank..., concomitant=data.concomitant[:, :, 1:1])
            @test single.q == single3.q
            @test single.basis == single3.basis
            close(single)
            close(single3)
            @test_throws DimensionMismatch HighOrderLowRankOp(data.grid, data.kspha, data.times;
                lowrank..., concomitant=data.concomitant[1:3, :, :])
            @test_throws DimensionMismatch HighOrderLowRankOp(data.grid, data.kspha, data.times;
                lowrank..., concomitant=data.concomitant[:, :, 1])
        end

        @testset "Prepared arrays and residual model" begin
            data = concomitant_test_problem(Float64)
            bf = basisfunc_spha(data.grid.x[vec(data.mask)], data.grid.y[vec(data.mask)],
                data.grid.z[vec(data.mask)], collect(1:16))
            prepared, coefficients = HighOrderMRI.prep_concomitant(bf, data.kspha, data.grid, vec(data.mask))
            @test prepared === bf
            @test coefficients === data.kspha
            original = deepcopy(data)
            prepared, coefficients = HighOrderMRI.prep_concomitant(bf, data.kspha, data.grid, vec(data.mask);
                concomitant=@view(data.concomitant[:, :, :]))
            @test data.kspha == original.kspha
            @test data.concomitant == original.concomitant
            @test size(prepared, 2) == size(coefficients, 1) == 20

            selected = merge(data, (; kspha=zero(data.kspha)))
            op = HighOrderLowRankOp(data.grid, data.kspha, data.times;
                fieldmap=data.fieldmap, csm=data.csm, mask=data.mask, concomitant=data.concomitant,
                recon_terms="0000", k_nominal=zeros(3, size(data.times)...),
                L_rank=6, rsvd_oversample=0, shared_rank_max=12, shared_basis_tol=1e-10)
            reference = concomitant_nfft_reference(selected, op)
            @test norm(op * data.image - reference * data.image) / norm(reference * data.image) < 1e-9
            close(op)

            data = merge(data, (; kspha=data.kspha[:, 1:2, :], times=data.times[1:2, :],
                concomitant=data.concomitant[:, 1:2, :]))
            op = HighOrderLowRankOp(data.grid, data.kspha, data.times;
                fieldmap=data.fieldmap, csm=data.csm, mask=data.mask, concomitant=data.concomitant,
                shared_basis_method=:joint, shared_rank_max=4, shared_basis_tol=1e-8,
                joint_snapshots=4, joint_samples=16)
            reference = concomitant_nfft_reference(data, op)
            @test norm(op * data.image - reference * data.image) / norm(reference * data.image) < 1e-8
            close(op)
        end

        @testset "Single-slice residual model" begin
            data = concomitant_test_problem(Float64; nZ=1)
            op = HighOrderLowRankOp(data.grid, data.kspha, data.times;
                fieldmap=data.fieldmap, csm=data.csm, mask=data.mask, concomitant=data.concomitant,
                L_rank=6, rsvd_oversample=0, shared_rank_max=12, shared_basis_tol=1e-10)
            reference = concomitant_nfft_reference(data, op)
            @test norm(op * data.image - reference * data.image) / norm(reference * data.image) < 1e-9
            @test norm(op' * data.signal - reference' * data.signal) / norm(reference' * data.signal) < 1e-9
            close(op)
        end

        @testset "CUDA phase preparation" begin
            if CUDA.functional()
                gpu_id = Int(CUDA.deviceid(CUDA.device()))
                data = concomitant_test_problem(Float32)
                kspha = reshape(data.kspha, 16, :)
                times = vec(data.times)
                common = (; fieldmap=data.fieldmap, csm=data.csm, mask=data.mask)
                op = HighOrderKernelOp(data.grid, kspha, times;
                    common..., concomitant=reshape(data.concomitant, 4, :), gpus=[gpu_id])
                reference = concomitant_dense_encoding(data)
                @test norm(op * data.image - reference * data.image) / norm(reference * data.image) < 3e-5
                @test norm(op' * data.signal - reference' * data.signal) / norm(reference' * data.signal) < 3e-5
                @testset "Resident concomitant phase" begin
                    x, y = CuArray(data.image), CuArray(data.signal)
                    D = Float32.(range(0.2, 1.1; length=length(y)))
                    for backend in (:direct, :blocked, :cached, :cached_full), phase_backend in (:kernel, :gemm)
                        resident = HighOrderKernelOp(data.grid, kspha, times;
                            common..., concomitant=reshape(data.concomitant, 4, :),
                            gpus=[gpu_id], backend, phase_backend, block_size=5)
                        try
                            @test norm(Array(resident*x) - reference*data.image) / norm(reference*data.image) < 3e-5
                            @test norm(Array(resident'*y) - reference'*data.signal) / norm(reference'*data.signal) < 3e-5
                            normal = HighOrderMRI.normalOperator(resident, D)
                            expected = reference' * (D .* (reference * data.image))
                            @test norm(Array(normal*x) - expected) / norm(expected) < 3e-5
                        finally
                            close(resident)
                        end
                    end
                    absent = HighOrderKernelOp(data.grid, kspha, times;
                        common..., gpus=[gpu_id], backend=:blocked)
                    zeroed = HighOrderKernelOp(data.grid, kspha, times;
                        common..., gpus=[gpu_id], backend=:blocked, concomitant=zeros(Float32, 4, length(times)))
                    try
                        @test Array(absent*x) ≈ Array(zeroed*x) rtol=3e-5
                        @test size(absent.gpu_operator.state.bf, 2) == 16
                        @test size(zeroed.gpu_operator.state.bf, 2) == 20
                    finally
                        close(absent)
                        close(zeroed)
                    end
                end
                @testset "Resident FP64 and worker preparation" begin
                    precise = concomitant_test_problem(Float64)
                    reference64 = concomitant_dense_encoding(precise)
                    x64, y64 = CuArray(precise.image), CuArray(precise.signal)
                    D64 = collect(range(0.2, 1.1; length=length(y64)))
                    for constructor in (HighOrderMRI.ExplicitGPUOp, HighOrderMRI.ExplicitMultiGPUOp),
                        phase_backend in (:kernel, :gemm)
                        # One visible device exercises worker-side sample slicing
                        # without requiring a second GPU in the test environment.
                        resident = constructor(precise.grid, reshape(precise.kspha, 16, :), vec(precise.times);
                            fieldmap=precise.fieldmap, csm=precise.csm, mask=precise.mask,
                            concomitant=reshape(precise.concomitant, 4, :), gpus=[gpu_id],
                            backend=:blocked, phase_backend, block_size=5)
                        try
                            @test norm(Array(resident*x64) - reference64*precise.image) / norm(reference64*precise.image) < 1e-11
                            @test norm(Array(resident'*y64) - reference64'*precise.signal) / norm(reference64'*precise.signal) < 1e-11
                            normal = HighOrderMRI.normalOperator(resident, D64)
                            expected = reference64' * (D64 .* (reference64 * precise.image))
                            @test norm(Array(normal*x64) - expected) / norm(expected) < 1e-11
                        finally
                            close(resident)
                        end
                    end
                end
                derivative = HighOrderKernelOp(data.grid, kspha, times;
                    common..., kspha_dt=kspha, gpus=[gpu_id])
                derivative_c = HighOrderKernelOp(data.grid, kspha, times;
                    common..., kspha_dt=kspha, concomitant=reshape(data.concomitant, 4, :), gpus=[gpu_id])
                @test derivative * data.image == derivative_c * data.image
                @test derivative' * data.signal == derivative_c' * data.signal

                data = concomitant_test_problem(Float32; nZ=1)
                for layout in (:shared, :per_readout)
                    cpu = HighOrderLowRankOp(data.grid, data.kspha, data.times;
                        fieldmap=data.fieldmap, csm=data.csm, mask=data.mask, concomitant=data.concomitant,
                        L_rank=6, rsvd_oversample=0, basis_layout=layout,
                        shared_rank_max=12, shared_basis_tol=0f0)
                    gpu = HighOrderLowRankOp(data.grid, data.kspha, data.times;
                        fieldmap=data.fieldmap, csm=data.csm, mask=data.mask, concomitant=data.concomitant,
                        L_rank=6, rsvd_oversample=0, basis_layout=layout,
                        shared_rank_max=12, shared_basis_tol=0f0,
                        arrayType=CuArray, gpus=[gpu_id], rsvd_backend=:kernel, rsvd_finalize=:gram)
                    @test norm(Array(gpu * CuArray(data.image)) - cpu * data.image) / norm(cpu * data.image) < 2e-3
                    @test norm(Array(gpu' * CuArray(data.signal)) - cpu' * data.signal) / norm(cpu' * data.signal) < 2e-3
                    close(cpu)
                    close(gpu)
                end
            else
                @test_skip "CUDA is not functional"
            end
        end
    finally
        AbstractNFFTs.set_active_backend!(previous_backend)
    end
end
