@testset "distributed rSVD" begin
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

            @testset "distributed forward and adjoint" begin
                omega = randn(Complex{T}, data.nVox, data.L_total)
                W_ref = data.E_ref * omega
                W = HighOrderMRI.distributed_rsvd_forward!(
                    workspace, data.times[:, 1], data.kspha_err; omega,
                )
                forward_multi_error = norm(W - W_ref) / max(norm(W_ref), eps(T))

                Q = randn(Complex{T}, data.nSam, data.L_total)
                B_ref = adjoint(data.E_ref) * Q
                gram_ref = adjoint(B_ref) * B_ref
                B, gram = HighOrderMRI.distributed_rsvd_adjoint!(workspace, Q)
                adjoint_multi_error = norm(B - B_ref) / max(norm(B_ref), eps(T))
                gram_multi_error = norm(gram - gram_ref) / max(norm(gram_ref), eps(T))

                @show forward_multi_error adjoint_multi_error gram_multi_error
                @test forward_multi_error < T(1e-4)
                @test adjoint_multi_error < T(1e-4)
                @test gram_multi_error < T(1e-4)
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
                    fastmath=true,
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
                @show distributed_rsvd_error
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
                    @test distributed_added == reference_added
                    @test reference_error <= shared_tol
                    @test distributed_error <= shared_tol
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
                    @show dyn distributed_shared_error
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
    end
end
