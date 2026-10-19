@testset "paired adjoint dense and original grouped reference" begin
    if CUDA.functional()
    rng=MersenneTwister(1029)
    for FT in (Float32,Float64), M in (1,5,12), nVox in (1,65,263)
        nSam=67;times=FT(.02).*rand(rng,FT,nSam);field=randn(rng,FT,nVox)
        bf=FT(.02).*randn(rng,FT,nVox,M);k=randn(rng,FT,M,nSam)
        E=cis.(FT(2π).*(times*transpose(field)+transpose(k)*transpose(bf)))
        td,fd,bd=CuArray(times),CuArray(field),CuArray(bf)
        tol=FT===Float32 ? FT(1e-4) : FT(1e-12)
        for L in (1,11,15,17,33), transposed in (false,true)
            Q=randn(rng,Complex{FT},nSam,L);qd=CuArray(Q);kd=CuArray(transposed ? permutedims(k) : k)
            old=CUDA.zeros(Complex{FT},nVox,L);new=similar(old)
            HighOrderMRI.run_kernel_rsvd_adjoint!(old,qd,td,fd,bd,kd;threads=256,lanes_per_voxel=2,kspha_transposed=transposed)
            HighOrderMRI.run_kernel_rsvd_adjoint!(new,qd,td,fd,bd,kd;threads=128,lanes_per_voxel=2,voxels_per_group=2,kspha_transposed=transposed)
            a,b=Array(old),Array(new);ref=adjoint(E)*Q
            @test norm(b-ref)/norm(ref)<tol
            @test a==b
        end
    end
end
end
