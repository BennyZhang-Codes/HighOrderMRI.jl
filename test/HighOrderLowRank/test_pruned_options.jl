@testset "optional normal execution configuration" begin
    @test isempty(HighOrderMRI.HighOrderNormalBackend(:single,[0],false,nothing,nothing).options)
    @test isempty(HighOrderMRI.HighOrderNormalBackend(:channel,[0,1],false,nothing,nothing,100,0,:batched,8).options)
    @test HighOrderMRI.validate_highorder_normal_options(:standard,(;)) === nothing
    @test HighOrderMRI.validate_highorder_normal_options(:batched,(fft=:cufft,)) === nothing
    @test_throws ArgumentError HighOrderMRI.validate_highorder_normal_options(:standard,(fft=:vkfft_pruned,))
    @test_throws ArgumentError HighOrderMRI.validate_highorder_normal_options(:batched,(fft=:unknown,))
    @test_throws ArgumentError HighOrderMRI.validate_highorder_normal_options(:batched,(unknown=true,))
    @test_throws ArgumentError HighOrderMRI.validate_highorder_normal_options(:batched,(fft=:vkfft_pruned,))
    @test_throws ArgumentError HighOrderMRI.validate_highorder_normal_options(:batched,(fft=:cufft,coefficient_cache_bytes=0))
    @test_throws ArgumentError HighOrderMRI.validate_highorder_normal_options(:batched,(native_ptx="unused.ptx",))
    @test_throws ArgumentError HighOrderMRI.validate_highorder_normal_options(:batched,(vkfft_library="unused.so",))
    @test_throws ArgumentError HighOrderMRI.validate_highorder_normal_options(:batched,(coefficient_cache_bytes=0,))
end
