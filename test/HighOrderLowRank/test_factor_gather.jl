@testset "double-buffered factor gather" begin
    if get(ENV, "HIGHORDERMRI_TEST_MULTIGPU_NORMAL", "false") == "true"
        gpu_ids = parse.(Int, split(get(ENV, "HIGHORDERMRI_TEST_GPUS", "0,1"), ','))
        CUDA.device!(first(gpu_ids))
        data = rsvd_test_problem()
        workspace = HighOrderMRI.DistributedRSVDWorkspace(data.fieldmap_masked,
            data.bf_err, data.nSam, data.L_total, data.L_rank, gpu_ids)
        try
            @testset "drain and slot reuse, nDyn=$nDyn" for nDyn in (1, 2, 7)
                destination = fill(ComplexF32(NaN), data.nVox, nDyn * data.L_rank)
                expected = similar(destination)
                gather = HighOrderMRI.DistributedFactorGather(destination, workspace)
                @test all(length(d.host_parts) == 2 for d in gather.downloads)
                try
                    for dyn in 1:nDyn
                        HighOrderMRI.perform_rsvd_multi_gpu!(workspace,
                            data.times[:, 1], data.kspha_err .* T(dyn);
                            seed=20 + dyn, v_read_finished=gather.v_read_finished)
                        columns = ((dyn - 1) * data.L_rank + 1):(dyn * data.L_rank)
                        expected[:, columns] .= HighOrderMRI.gather_distributed_v_scaled(workspace)
                        HighOrderMRI.submit_factor_gather!(gather, dyn)
                        @test gather.pending <= 2
                    end
                    HighOrderMRI.drain_factor_gather!(gather)
                    @test destination == expected
                    @test gather.pending == 0
                    @test gather.submitted == nDyn
                    @test gather.host_write_s >= 0
                    @test gather.device_copy_s >= 0
                    @test_throws BoundsError HighOrderMRI.submit_factor_gather!(gather, nDyn + 1)
                finally
                    close(gather)
                end
                close(gather)
                @test all(istaskdone(w.task) for w in gather.collectors)
                @test_throws ErrorException HighOrderMRI.submit_factor_gather!(gather, 1)
            end
            @testset "collector failure propagation and shutdown" begin
                gather = HighOrderMRI.DistributedFactorGather(
                    zeros(ComplexF32, data.nVox, data.L_rank), workspace)
                for worker in gather.collectors
                    put!(worker.jobs, () -> error("injected factor collector failure"))
                end
                gather.pending = 1
                @test_throws HighOrderMRI.DistributedGPUWorkerError HighOrderMRI.drain_factor_gather!(gather)
                @test gather.pending == 0
                close(gather)
                @test all(istaskdone(w.task) for w in gather.collectors)
            end
        finally
            HighOrderMRI.release_distributed_workspaces!(workspace)
            HighOrderMRI.shutdown_distributed_workers!(workspace.workers)
        end

        @eval import NonuniformFFTs
        previous_backend = AbstractNFFTs.active_backend()
        try
            AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
            @testset "full operator factors, nZ=$nZ" for nZ in (1, 10)
                problem = per_readout_test_problem(; nZ, nDyn=7)
                common = (;fieldmap=problem.fieldmap, csm=problem.csm, mask=problem.mask,
                    arrayType=CuArray, gpus=gpu_ids, L_rank=2, rsvd_oversample=2,
                    rsvd_seed=2026, rsvd_finalize=:gram, rsvd_backend=:kernel,
                    rsvd_distribution=:voxel, basis_layout=:per_readout,
                    per_readout_storage=:cpu)
                synchronous = HighOrderLowRankOp(problem.grid,problem.kspha,problem.times;
                    common...,per_readout_gather=:synchronous)
                pipelined = HighOrderLowRankOp(problem.grid,problem.kspha,problem.times;
                    common...,per_readout_gather=:pipeline)
                @test Array(synchronous.q) == Array(pipelined.q)
                @test synchronous.basis == pipelined.basis
                x = CuArray(problem.x)
                @test norm(Array(synchronous*x-pipelined*x))/norm(Array(synchronous*x)) < 1f-5
                close(synchronous)
                close(pipelined)
            end
        finally
            AbstractNFFTs.set_active_backend!(previous_backend)
        end
    end
end
