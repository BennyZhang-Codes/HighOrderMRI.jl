using CUDA

function explicit_highorder_test_problem(; n_coil::Int = 3)
    scalar_type = Float32
    n_x, n_y, n_z = 4, 3, 2
    n_sample, n_term = 7, 16
    grid = Grid(n_x, n_y, n_z, scalar_type(0.7), scalar_type(0.9), scalar_type(1.2))

    times = collect(range(zero(scalar_type), scalar_type(1.3e-3); length = n_sample))
    kspha = zeros(scalar_type, n_term, n_sample)
    kspha[1, :] .= range(scalar_type(-0.01), scalar_type(0.015); length = n_sample)
    kspha[2, :] .= range(scalar_type(-0.08), scalar_type(0.07); length = n_sample)
    kspha[3, :] .= range(scalar_type(0.06), scalar_type(-0.05); length = n_sample)
    kspha[4, :] .= range(scalar_type(-0.03), scalar_type(0.04); length = n_sample)
    for term = 5:9
        kspha[term, :] .=
            scalar_type(4e-4 * (term - 4)) .*
            sin.(range(zero(scalar_type), scalar_type(π); length = n_sample),)
    end
    for term = 10:16
        kspha[term, :] .=
            scalar_type(2e-5 * (term - 9)) .*
            cos.(range(zero(scalar_type), scalar_type(π); length = n_sample),)
    end

    fieldmap = reshape(
        collect(range(scalar_type(-25), scalar_type(20); length = n_x * n_y * n_z)),
        n_x,
        n_y,
        n_z,
    )
    normalized_x = reshape(grid.x ./ maximum(abs, grid.x), n_x, n_y, n_z)
    normalized_y = reshape(grid.y ./ maximum(abs, grid.y), n_x, n_y, n_z)
    normalized_z = reshape(grid.z ./ maximum(abs, grid.z), n_x, n_y, n_z)
    csm = Array{Complex{scalar_type}}(undef, n_x, n_y, n_z, n_coil)
    for coil = 1:n_coil
        angle = scalar_type(2π * (coil - 1) / n_coil)
        magnitude =
            one(scalar_type) .+
            scalar_type(0.08) .* (cos(angle) .* normalized_x .+ sin(angle) .* normalized_y)
        phase = scalar_type(0.12 * coil) .* (normalized_y .- normalized_z)
        @views csm[:, :, :, coil] .= magnitude .* cis.(phase)
    end

    mask = trues(n_x, n_y, n_z)
    mask[1, 1, 1] = false
    mask[end, end, end] = false
    image =
        complex.(
            collect(
                range(scalar_type(0.1), one(scalar_type); length = prod(grid.matrixSize)),
            ),
            collect(
                range(scalar_type(-0.2), scalar_type(0.15); length = prod(grid.matrixSize)),
            ),
        )

    return (; grid, kspha, times, fieldmap, csm, mask, image)
end

function explicit_highorder_dense_matrix(data)
    scalar_type = eltype(data.times)
    n_term, n_sample = size(data.kspha)
    n_coil = size(data.csm, ndims(data.csm))
    mask_flat = vec(data.mask)
    mask_indices = findall(mask_flat)
    n_voxel = length(mask_indices)

    basis = HighOrderMRI.basisfunc_spha(
        data.grid.x[mask_flat],
        data.grid.y[mask_flat],
        data.grid.z[mask_flat],
        collect(1:n_term),
    )
    phase = basis * data.kspha .+ vec(data.fieldmap)[mask_flat] * transpose(data.times)
    encoding = transpose(cis.(scalar_type(2π) .* phase)) ./ sqrt(scalar_type(n_voxel))
    csm_masked = reshape(data.csm, :, n_coil)[mask_flat, :]

    matrix = zeros(Complex{scalar_type}, n_sample * n_coil, prod(data.grid.matrixSize))
    for coil = 1:n_coil
        rows = (coil-1)*n_sample+1:coil*n_sample
        @views matrix[rows, mask_indices] .= encoding .* transpose(csm_masked[:, coil])
    end
    return matrix
end

@testset "Explicit high-order operators" begin
    data = explicit_highorder_test_problem()
    dense_matrix = explicit_highorder_dense_matrix(data)
    n_sample = size(data.kspha, 2)
    n_coil = size(data.csm, ndims(data.csm))
    probe = complex.(collect(range(-0.4f0, 0.5f0; length = n_sample * n_coil)), collect(range(0.3f0, -0.2f0; length = n_sample * n_coil)))

    @testset "HighOrderOp CPU and block partition" begin
        blocked_operator = HighOrderOp(
            data.grid,
            data.kspha,
            data.times;
            fieldmap = data.fieldmap,
            csm = data.csm,
            mask = data.mask,
            nBlock = 4,
            arrayType = Array,
        )
        single_block_operator = HighOrderOp(
            data.grid,
            data.kspha,
            data.times;
            fieldmap = data.fieldmap,
            csm = data.csm,
            mask = data.mask,
            nBlock = 1,
            arrayType = Array,
        )

        expected_forward = dense_matrix * data.image
        expected_adjoint = adjoint(dense_matrix) * probe
        blocked_forward = blocked_operator * data.image
        blocked_adjoint = adjoint(blocked_operator) * probe

        @test size(blocked_operator) == size(dense_matrix)
        @test blocked_forward ≈ expected_forward rtol = 3.0f-6 atol = 2.0f-6
        @test blocked_adjoint ≈ expected_adjoint rtol = 3.0f-6 atol = 2.0f-6
        @test single_block_operator * data.image ≈ blocked_forward rtol = 3.0f-6 atol = 2.0f-6
        @test adjoint(single_block_operator) * probe ≈ blocked_adjoint rtol = 3.0f-6 atol = 2.0f-6
        @test dot(blocked_forward, probe) ≈ dot(data.image, blocked_adjoint) rtol = 3.0f-6 atol = 2.0f-6
        @test all(iszero, blocked_adjoint[.!vec(data.mask)])
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError HighOrderOp(
            data.grid,
            data.kspha,
            data.times;
            arrayType = BitArray,
        )

        @test_throws AssertionError HighOrderOp(
            data.grid,
            zeros(Float32, 10, n_sample),
            data.times;
            arrayType = Array,
        )
        @test_throws ArgumentError HighOrderKernelOp(
            data.grid,
            data.kspha,
            data.times;
            arrayType = Array,
        )
        @test_throws ArgumentError HighOrderKernelOp(
            data.grid,
            data.kspha,
            data.times;
            gpus = Int[],
        )
        @test_throws ArgumentError HighOrderKernelOp(
            data.grid,
            data.kspha,
            data.times;
            gpus = [0, 0],
        )
    end

    @testset "CUDA array and kernel implementations" begin
        if CUDA.functional()
            gpu_id = Int(CUDA.deviceid(CUDA.device()))
            cuda_operator = HighOrderOp(
                data.grid,
                data.kspha,
                data.times;
                fieldmap = data.fieldmap,
                csm = data.csm,
                mask = data.mask,
                nBlock = 4,
                arrayType = CuArray,
            )
            kernel_operator = HighOrderKernelOp(
                data.grid,
                data.kspha,
                data.times;
                fieldmap = data.fieldmap,
                csm = data.csm,
                mask = data.mask,
                gpus = [gpu_id],
            )

            image_gpu = CuArray(data.image)
            probe_gpu = CuArray(probe)
            expected_forward = dense_matrix * data.image
            expected_adjoint = adjoint(dense_matrix) * probe
            cuda_forward = Array(cuda_operator * image_gpu)
            cuda_adjoint = Array(adjoint(cuda_operator) * probe_gpu)
            kernel_forward = Array(kernel_operator * image_gpu)
            kernel_adjoint = Array(adjoint(kernel_operator) * probe_gpu)

            @test cuda_forward ≈ expected_forward rtol = 3.0f-5 atol = 2.0f-5
            @test cuda_adjoint ≈ expected_adjoint rtol = 3.0f-5 atol = 2.0f-5
            @test kernel_forward ≈ expected_forward rtol = 3.0f-5 atol = 2.0f-5
            @test kernel_adjoint ≈ expected_adjoint rtol = 3.0f-5 atol = 2.0f-5
            @test kernel_forward ≈ cuda_forward rtol = 3.0f-5 atol = 2.0f-5
            @test kernel_adjoint ≈ cuda_adjoint rtol = 3.0f-5 atol = 2.0f-5
            @test dot(kernel_forward, probe) ≈ dot(data.image, kernel_adjoint) rtol = 3.0f-5 atol =
                2.0f-5

            gpu_ids = map(device -> Int(CUDA.deviceid(device)), CUDA.devices())
            if length(gpu_ids) >= 2
                multi_gpu_operator = HighOrderKernelOp(
                    data.grid,
                    data.kspha,
                    data.times;
                    fieldmap = data.fieldmap,
                    csm = data.csm,
                    mask = data.mask,
                    gpus = gpu_ids[1:2],
                )
                @test Array(multi_gpu_operator * image_gpu) ≈ kernel_forward rtol = 3.0f-5 atol =
                    2.0f-5
                @test Array(adjoint(multi_gpu_operator) * probe_gpu) ≈ kernel_adjoint rtol =
                    3.0f-5 atol = 2.0f-5
            end
        else
            @test_skip "CUDA is not functional"
        end
    end
end
