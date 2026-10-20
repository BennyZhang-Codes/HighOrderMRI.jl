@testset "regenerated factor precision" begin
    if get(ENV,"HIGHORDERMRI_TEST_MULTIGPU_NORMAL","false")=="true"
        gpu_ids=parse.(Int,split(get(ENV,"HIGHORDERMRI_TEST_GPUS","0,1"),','))
        CUDA.device!(first(gpu_ids))
        @testset "$FT" for FT in (Float32,Float64)
            rng=Random.Xoshiro(2026)
            nVox,nSam,nDyn,rank,total=37,8,7,3,5
            times=repeat(reshape(collect(range(zero(FT),FT(0.002);length=nSam)),:,1),1,nDyn)
            field=FT(20).*randn(rng,FT,nVox)
            bf=FT(0.02).*randn(rng,FT,nVox,5)
            kspha=randn(rng,FT,5,nSam,nDyn)
            source=HighOrderMRI.RegeneratedSpatialBasis(times,field,bf,kspha,rank,total,0)
            randn!(rng,source.Q);randn!(rng,source.Z)
            buffers=[CuMatrix{Complex{FT}}(undef,nVox,rank) for _ in 1:2]
            CUDA.synchronize()
            stream=CuStream(;flags=CUDA.STREAM_NON_BLOCKING)
            generator=HighOrderMRI.SpatialFactorGenerator(source,buffers,stream)
            ready=[CuEvent() for _ in 1:2];consumed=[CuEvent() for _ in 1:2]
            started=[CuEvent() for _ in 1:2];done=[CuEvent() for _ in 1:2]
            used=falses(2)
            try
                for dyn in 1:nDyn
                    slot=mod1(dyn,2)
                    HighOrderMRI.submit_generated_factor!(generator,slot,dyn,stream,
                        consumed[slot],used[slot],started[slot],done[slot],ready[slot])
                    CUDA.wait(ready[slot],CUDA.stream())
                    actual=Array(buffers[slot])
                    phase=Float64.(times[:,dyn])*transpose(Float64.(field))+
                        Float64.(source.kspha_t[:,:,dyn])*transpose(Float64.(bf))
                    expected=(adjoint(cis.(2π.*phase))*ComplexF64.(source.Q[:,:,dyn])) *
                        ComplexF64.(source.Z[:,:,dyn])
                    error=norm(actual-expected)/norm(expected)
                    @test error < (FT===Float32 ? 3e-6 : 1e-12)
                    CUDA.record(consumed[slot],CUDA.stream());used[slot]=true
                end
            finally
                CUDA.synchronize(stream);CUDA.synchronize()
                close(generator)
                foreach(CUDA.unsafe_free!,buffers)
            end
            close(generator)
            @test istaskdone(generator.worker.task)
            @test generator.released
        end
    end
end
