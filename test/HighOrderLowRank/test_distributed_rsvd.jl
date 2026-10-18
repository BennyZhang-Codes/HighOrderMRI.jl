@testset "distributed rSVD" begin
    @testset "shared basis retains energy below rank-scaled threshold" begin
        T = Float32
        L = 80
        V = zeros(Complex{T}, L, L)
        V[1, 1] = sqrt(T(1e9))
        for j in 2:21
            V[j, j] = sqrt(T(5000))
        end
        shared = HighOrderMRI.SharedSpatialBasis(V, T, L, L, 1, L, T(1e-3))
        workspace = HighOrderMRI.SharedBasisUpdateWorkspace(V, T, L, L, L)
        reported, added = HighOrderMRI.update_shared_basis!(
            shared, workspace, V, 1, T(sum(abs2, V)),
        )
        reconstructed = shared.basis[:, 1:shared.rank] *
            shared.coeff[1:shared.rank, :, 1]
        actual = norm(V - reconstructed) / norm(V)
        @test added == 21
        @test reported <= shared.tol
        @test actual <= shared.tol
    end

    @testset "shared basis resolves cancellation-dominated tail" begin
        nVox, L = 256, 24
        U = ComplexF64[cis(2π * (v - 1) * (j - 1) / nVox) / sqrt(nVox)
            for v in 1:nVox, j in 1:L]
        W = ComplexF64[cis(2π * (r - 1) * (j - 1) / L) / sqrt(L)
            for r in 1:L, j in 1:L]
        spectrum = vcat(1e8, fill(11.0, 20), zeros(3))
        backends = CUDA.functional() ? (Array, CuArray) : (Array,)
        for FT in (Float32, Float64), backend in backends
            @testset "$FT / $backend" begin
                V = Complex{FT}.(U * Diagonal(sqrt.(spectrum)) * adjoint(W))
                input = backend(V)
                shared = HighOrderMRI.SharedSpatialBasis(input, FT, nVox, L, 1, L, FT(1e-3))
                workspace = HighOrderMRI.SharedBasisUpdateWorkspace(input, FT, nVox, L, L)
                @test (workspace.residual64 === workspace.residual) == (FT === Float64)
                reported, added = HighOrderMRI.update_shared_basis!(
                    shared, workspace, input, 1, FT(sum(abs2, V)),
                )
                reconstructed = Array(shared.basis[:, 1:shared.rank]) *
                    Array(shared.coeff[1:shared.rank, :, 1])
                actual = norm(V - reconstructed) / norm(V)
                @test eltype(shared.basis) === Complex{FT}
                @test eltype(shared.coeff) === Complex{FT}
                @test added == 12
                @test reported <= shared.tol
                @test actual <= shared.tol
                @test abs(reported - actual) < FT(1e-5)
            end
        end
    end

    warning_message =
        "Julia default thread count is smaller than the number of GPU workers; " *
        "multi-GPU execution may be CPU-thread limited"
    insufficient = @test_logs (:warn, warning_message) HighOrderMRI.warn_if_insufficient_gpu_worker_threads(
        2;
        operation=:test,
        default_threads=1,
    )
    @test !insufficient
    @test HighOrderMRI.warn_if_insufficient_gpu_worker_threads(
        2;
        operation=:test,
        default_threads=2,
    )

    if CUDA.functional() && length(collect(CUDA.devices())) >= 4
        data = rsvd_test_problem()
        test_gpus = collect(0:3)
        workspace = HighOrderMRI.DistributedRSVDWorkspace(
            data.fieldmap_masked,
            data.bf_err,
            data.nSam,
            data.L_total,
            data.L_rank,
            test_gpus,
        )
        @test size(workspace.kspha_t_host) == (data.nSam, size(data.kspha_err, 1))
        @test all(
            size(shard.kspha_t) == size(workspace.kspha_t_host)
            for shard in workspace.shards
        )
        @test workspace.rsvd_backend === :kernel
        @test workspace.chunk_size == 0
        @test all(
            size(shard.phase) == (data.nSam, 0) && size(shard.encoding) == (data.nSam, 0)
            for shard in workspace.shards
        )
        distributed_shared = nothing
        workspaces_released = false

        try
            @testset "worker failure recovery" begin
                worker_outputs = Vector{Nothing}(undef, length(workspace.workers))
                injected_error = try
                    HighOrderMRI.run_on_workers!(
                        worker_outputs,
                        workspace.workers;
                        phase=:injected_failure,
                    ) do i
                        i == 2 && error("injected worker failure")
                        nothing
                    end
                    nothing
                catch err
                    err
                end

                @test injected_error isa HighOrderMRI.DistributedGPUWorkerError
                if injected_error isa HighOrderMRI.DistributedGPUWorkerError
                    @test injected_error.gpu_id == test_gpus[2]
                    @test injected_error.worker_index == 2
                    @test injected_error.phase === :injected_failure
                    @test injected_error.captured.ex isa ErrorException
                    @test injected_error.captured.ex.msg == "injected worker failure"
                end

                HighOrderMRI.run_on_workers!(
                    worker_outputs,
                    workspace.workers;
                    phase=:recovery_check,
                ) do _
                    nothing
                end
                @test all(isnothing, worker_outputs)
            end

            @testset "distributed forward" begin
                omega = randn(Complex{T}, data.nVox, data.L_total)
                W_ref = data.E_ref * omega
                W = HighOrderMRI.distributed_rsvd_forward!(
                    workspace, data.times[:, 1], data.kspha_err; omega,
                )
                forward_multi_error = norm(W - W_ref) / max(norm(W_ref), eps(T))

                @test forward_multi_error < T(1e-4)
            end

            @testset "distributed finalization" begin
                omega = randn(Complex{T}, data.nVox, data.L_total)
                timing = HighOrderMRI.DistributedRSVDTiming()
                u_multi, energy_multi = HighOrderMRI.perform_rsvd_multi_gpu!(
                    workspace,
                    data.times[:, 1],
                    data.kspha_err;
                    seed=17,
                    omega,
                    timing,
                )
                v_multi = HighOrderMRI.gather_distributed_v_scaled(workspace)

                W_reference = data.E_ref * omega
                qr_reference = qr(W_reference)
                Q_seed = Matrix{Complex{T}}(I, data.nSam, data.L_total)
                Q_reference = Matrix(qr_reference.Q * Q_seed)
                B_reference = adjoint(data.E_ref) * Q_reference
                gram_reference = adjoint(B_reference) * B_reference
                gram_reference = (gram_reference + adjoint(gram_reference)) * T(0.5)
                eig_reference = eigen(Hermitian(gram_reference))
                order = sortperm(real.(eig_reference.values); rev=true)
                values_reference = max.(T.(real.(eig_reference.values[order])), zero(T))
                Z_reference = eig_reference.vectors[:, order[1:data.L_rank]]
                u_reference = Q_reference * Z_reference
                v_reference = B_reference * Z_reference

                E_multi = u_multi * adjoint(v_multi)
                E_reference = u_reference * adjoint(v_reference)
                distributed_rsvd_error =
                    norm(E_multi - E_reference) / max(norm(E_reference), eps(T))
                energy_reference = sum(values_reference[1:data.L_rank])
                @test distributed_rsvd_error < T(1e-3)
                @test energy_multi ≈ energy_reference rtol=T(1e-3)
                @test timing.n_calls == 1
                @test timing.forward_time >= 0.0
                @test timing.qr_time >= 0.0
                @test timing.adjoint_gram_time >= 0.0
                @test timing.finalize_time >= 0.0
                @test HighOrderMRI.distributed_rsvd_total_time(timing) >= 0.0

                HighOrderMRI.reset_distributed_rsvd_timing!(timing)
                @test timing.n_calls == 0
                @test HighOrderMRI.distributed_rsvd_total_time(timing) == 0.0
            end

            @testset "distributed shared spatial basis" begin
                nDyn = size(data.times, 2)
                max_rank = min(data.nVox, data.L_rank * nDyn)
                shared_tol = T(1e-2)
                distributed_shared = HighOrderMRI.DistributedSharedSpatialBasis(
                    workspace, nDyn, max_rank, shared_tol,
                )
                @test all(
                    CUDA.device(distributed_shared.shards[i].residual64) ==
                    CUDA.device(workspace.shards[i].v_scaled) &&
                    CUDA.device(distributed_shared.shards[i].gram) ==
                    CUDA.device(workspace.shards[i].v_scaled)
                    for i in eachindex(workspace.shards)
                )
                reference_shared = HighOrderMRI.SharedSpatialBasis(
                    data.times[:, 1], T, data.nVox, data.L_rank,
                    nDyn, max_rank, shared_tol,
                )
                reference_workspace = HighOrderMRI.SharedBasisUpdateWorkspace(
                    data.times[:, 1], T, data.nVox, data.L_rank, max_rank,
                )

                for dyn = 1:nDyn
                    kspha_dyn = data.kspha[5:end, :, dyn]
                    omega_dyn = randn(Complex{T}, data.nVox, data.L_total)
                    _, total_energy = HighOrderMRI.perform_rsvd_multi_gpu!(
                        workspace, data.times[:, dyn], kspha_dyn;
                        seed=17 + dyn - 1,
                        omega=omega_dyn,
                    )
                    v_reference = HighOrderMRI.gather_distributed_v_scaled(workspace)
                    reference_error, reference_added = HighOrderMRI.update_shared_basis!(
                        reference_shared,
                        reference_workspace,
                        v_reference,
                        dyn,
                        total_energy,
                    )
                    distributed_error, distributed_added =
                        HighOrderMRI.update_distributed_shared_basis!(
                            distributed_shared, workspace, dyn, total_energy,
                        )
                    distributed_basis_now = HighOrderMRI.gather_distributed_shared_basis(
                        distributed_shared, workspace,
                    )
                    C_now = @view distributed_shared.coeff[1:distributed_shared.rank, :, dyn]
                    actual_error = norm(v_reference - distributed_basis_now * C_now) /
                        norm(v_reference)
                    @test distributed_added == reference_added
                    @test reference_error <= shared_tol
                    @test distributed_error <= shared_tol
                    @test actual_error <= shared_tol + T(5e-5)
                    @test distributed_error ≈ reference_error rtol=T(1e-3) atol=T(1e-3)
                end

                distributed_basis = HighOrderMRI.gather_distributed_shared_basis(
                    distributed_shared, workspace,
                )
                @test distributed_shared.rank == reference_shared.rank

                for dyn = 1:nDyn
                    V_reference = zeros(Complex{T}, data.nVox, data.L_rank)
                    HighOrderMRI.reconstruct_spatial_factors!(
                        V_reference, reference_shared, dyn,
                    )
                    rank = distributed_shared.rank
                    C_distributed = @view distributed_shared.coeff[1:rank, :, dyn]
                    V_distributed = distributed_basis * C_distributed
                    distributed_shared_error =
                        norm(V_distributed - V_reference) /
                        max(norm(V_reference), eps(T))
                    @test distributed_shared_error < T(1e-4)
                end
            end

            @testset "explicit workspace release" begin
                HighOrderMRI.release_distributed_workspaces!(
                    workspace, distributed_shared,
                )
                workspaces_released = true
                @test all(shard -> shard.released, workspace.shards)
                @test all(shard -> shard.released, distributed_shared.shards)
            end
        finally
            try
                if !workspaces_released
                    HighOrderMRI.release_distributed_workspaces!(
                        workspace, distributed_shared,
                    )
                end
            finally
                HighOrderMRI.shutdown_distributed_workers!(workspace.workers)
            end
        end

        @testset "chunked backend" begin
            # Four shards split 15 voxels unevenly, and a two-voxel chunk
            # exercises both full and tail chunks on every GPU.
            chunk_size = 2
            omega = randn(Complex{T}, data.nVox, data.L_total)

            kernel_workspace = HighOrderMRI.DistributedRSVDWorkspace(
                data.fieldmap_masked,
                data.bf_err,
                data.nSam,
                data.L_total,
                data.L_rank,
                test_gpus;
                rsvd_backend=:kernel,
            )
            kernel_W = nothing
            kernel_reconstruction = nothing
            kernel_energy = zero(T)
            try
                kernel_W = HighOrderMRI.distributed_rsvd_forward!(
                    kernel_workspace, data.times[:, 1], data.kspha_err; omega,
                )
                u_kernel, kernel_energy = HighOrderMRI.perform_rsvd_multi_gpu!(
                    kernel_workspace, data.times[:, 1], data.kspha_err; omega,
                )
                v_kernel = HighOrderMRI.gather_distributed_v_scaled(kernel_workspace)
                kernel_reconstruction = u_kernel * adjoint(v_kernel)
            finally
                HighOrderMRI.release_distributed_workspaces!(kernel_workspace)
                HighOrderMRI.shutdown_distributed_workers!(kernel_workspace.workers)
            end

            chunked_workspace = HighOrderMRI.DistributedRSVDWorkspace(
                data.fieldmap_masked,
                data.bf_err,
                data.nSam,
                data.L_total,
                data.L_rank,
                test_gpus;
                rsvd_backend=:chunked,
                rsvd_chunk=chunk_size,
            )
            chunked_released = false
            try
                @test chunked_workspace.rsvd_backend === :chunked
                @test chunked_workspace.chunk_size == chunk_size
                @test all(
                    size(shard.phase) == (data.nSam, min(chunk_size, length(shard.voxels))) &&
                    size(shard.encoding) == (data.nSam, min(chunk_size, length(shard.voxels)))
                    for shard in chunked_workspace.shards
                )

                chunked_W = HighOrderMRI.distributed_rsvd_forward!(
                    chunked_workspace, data.times[:, 1], data.kspha_err; omega,
                )
                u_chunked, chunked_energy = HighOrderMRI.perform_rsvd_multi_gpu!(
                    chunked_workspace, data.times[:, 1], data.kspha_err; omega,
                )
                v_chunked = HighOrderMRI.gather_distributed_v_scaled(chunked_workspace)
                chunked_reconstruction = u_chunked * adjoint(v_chunked)

                W_reference = data.E_ref * omega
                @test norm(chunked_W - W_reference) / max(norm(W_reference), eps(T)) < T(1e-4)
                @test chunked_W ≈ kernel_W rtol=T(1e-4) atol=T(1e-5)
                @test norm(chunked_reconstruction - kernel_reconstruction) /
                      max(norm(kernel_reconstruction), eps(T)) < T(1e-3)
                @test chunked_energy ≈ kernel_energy rtol=T(1e-3)

                HighOrderMRI.release_distributed_workspaces!(chunked_workspace)
                chunked_released = true
                @test all(shard -> shard.released, chunked_workspace.shards)
            finally
                try
                    if !chunked_released
                        HighOrderMRI.release_distributed_workspaces!(chunked_workspace)
                    end
                finally
                    HighOrderMRI.shutdown_distributed_workers!(chunked_workspace.workers)
                end
            end
        end

        @testset "distributed shared Gram spectrum and buffer ownership" begin
            # Uneven voxel shards must retain the same small spectral tail.
            nVox, L = 259, 24
            U = ComplexF64[cis(2π * (v - 1) * (j - 1) / nVox) / sqrt(nVox)
                for v in 1:nVox, j in 1:L]
            W = ComplexF64[cis(2π * (r - 1) * (j - 1) / L) / sqrt(L)
                for r in 1:L, j in 1:L]
            spectrum = vcat(1e8, fill(11.0, 20), zeros(3))
            for FT in (Float32, Float64)
                @testset "$FT" begin
                    V = Complex{FT}.(U * Diagonal(sqrt.(spectrum)) * adjoint(W))
                    gram_workspace = HighOrderMRI.DistributedRSVDWorkspace(
                        zeros(FT, nVox), zeros(FT, nVox, 1), L, L, L, test_gpus,
                    )
                    gram_shared = nothing
                    try
                        gram_shared = HighOrderMRI.DistributedSharedSpatialBasis(
                            gram_workspace, 2, L, FT(1e-3),
                        )
                        @test all(shard ->
                            (shard.residual64 === shard.residual) == (FT === Float64),
                            gram_shared.shards,
                        )
                        outputs = Vector{Nothing}(undef, length(gram_workspace.workers))
                        HighOrderMRI.run_on_workers!(outputs, gram_workspace.workers) do i
                            shard = gram_workspace.shards[i]
                            CUDA.@sync copyto!(shard.v_scaled, V[shard.voxels, :])
                            nothing
                        end
                        for dyn in 1:2
                            reported, added = HighOrderMRI.update_distributed_shared_basis!(
                                gram_shared, gram_workspace, dyn, FT(sum(abs2, V)),
                            )
                            basis = HighOrderMRI.gather_distributed_shared_basis(
                                gram_shared, gram_workspace,
                            )
                            coeff = gram_shared.coeff[1:gram_shared.rank, :, dyn]
                            actual = norm(V - basis * coeff) / norm(V)
                            @test eltype(basis) === Complex{FT}
                            @test eltype(coeff) === Complex{FT}
                            @test added == (dyn == 1 ? 12 : 0)
                            @test reported <= gram_shared.tol
                            @test actual <= gram_shared.tol
                            @test abs(reported - actual) < FT(1e-5)
                        end
                        HighOrderMRI.release_distributed_workspaces!(gram_workspace, gram_shared)
                        HighOrderMRI.release_distributed_workspaces!(gram_workspace, gram_shared)
                        @test all(shard -> shard.released, gram_workspace.shards)
                        @test all(shard -> shard.released, gram_shared.shards)
                    finally
                        try
                            HighOrderMRI.release_distributed_workspaces!(gram_workspace, gram_shared)
                        finally
                            HighOrderMRI.shutdown_distributed_workers!(gram_workspace.workers)
                        end
                    end
                end
            end
        end
    end
end
