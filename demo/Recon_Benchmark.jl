using HighOrderMRI
using RegularizedLeastSquares
using MAT
using CUDA
using BenchmarkTools

T             = Float32;
path          = joinpath(@__DIR__, "")
data_mat      = "7T_2D_Spiral_1p0_200_r4.mat" 
data_file     = joinpath(path, data_mat)

@info "data file: $(data_file)"
data          = matread(data_file);

csm           = data["gre_csm"];            # coil sensitivity map
b0            = data["gre_b0"];             # ΔB0 map
mask          = data["gre_mask"];           # mask

kdata         = data["kdata"];              # spiral k-space data
datatime      = data["datatime"];           # time stamps of k-space data
matrixSize    = data["matrixSize"];         # matrix size
FOV           = data["FOV"];                # field of view

k0_ecc        = data["k0_adc"];             # b0 compensation of scanner from ECC model
dt_Measured   = data["dt_Measured"];
ksphaMeasured = data["ksphaMeasured"];     # coefficients of the field dynamics
startMeasured = data["startMeasured"];     # start time of the field dynamics
tauMeasured   = data["tauMeasured"];       # synchronization delay between the Measured trajectory and the MRI data
dt_adc        = data["dt_adc"];

# comparion of the field dynamics with or without synchronization
# plt_ksphas([InterpTrajTime(ksphaMeasured, dt_Measured, startMeasured, datatime), InterpTrajTime(ksphaMeasured, dt_Measured, startMeasured+tauMeasured*dt_adc, datatime)], dt_adc)
datatime      = vec(datatime);
ksphaMeasured = InterpTrajTime(ksphaMeasured, dt_Measured, startMeasured + tauMeasured * dt_adc, datatime);

# prepare some parameters for reconstruction
# 1. gridding 
Δx, Δy, Δz    = T.(FOV ./ matrixSize);
nX, nY, nZ    = matrixSize;
gridding      = Grid(nX, nY, nZ, Δx, Δy, Δz; exchange_xy=true, reverse_x=false, reverse_y=true)

# 2. sampling density 
weightMeasured = SampleDensity(ksphaMeasured'[2:3,:], (nX, nY));

###########################################################################
# recon with field dynamics measured by Dynamic Field Camera
###########################################################################
kdata = data["kdata"];
kdata = kdata ./ exp.(2π*1im.*k0_ecc)';
kdata = kdata .* exp.(-2π*1im.*ksphaMeasured[:, 1]);

labelMeasured = [    "Measured"];
recons        = [        "0111"];
kdatas        = [         kdata];
weights       = [weightMeasured]; 
b0s           = [            b0];
imgMeasured   = Array{Complex{T},3}(undef, nX, nY, length(labelMeasured));
idxs           = collect(1:length(labelMeasured));

idx = 1; label = labelMeasured[1]; recon = recons[1]; kdata = kdatas[1]; weight = weights[1]; b0 = b0s[1];

@info "[$(idx)] $(label) $(recon)"

use_gpu = true;
verbose = false;

recParams = Dict{Symbol,Any}()
recParams[:reconSize]  = (nX, nY)
recParams[:reg]        = L2Regularization(1.e-9)  
recParams[:iterations] = 20
recParams[:solver]     = CGNR


###########################################################################
# Run for precomplie
###########################################################################
@info "HighOderOp: Array-based, 1 GPU"
CUDA.device!(0)
GC.gc(); CUDA.reclaim();
Op1 = HighOrderOp(gridding, T.(ksphaMeasured'), T.(datatime); 
        recon_terms=recon, csm=Complex{T}.(csm), fieldmap=T.(b0), 
        use_gpu=use_gpu, verbose=verbose, nBlock=100);
CUDA.@time x1 = recon_HOOp(Op1, Complex{T}.(kdata), Complex{T}.(weight), recParams);
fig1 = plt_image(abs.(x1); title="HighOderOp: Array-based, 1 GPU", vmaxp=99.9, width=20, height=20)

@info "HighOderOp: Array-based, 1 GPU, with mask"
CUDA.device!(0)
GC.gc(); CUDA.reclaim();
Op2 = HighOrderOp(gridding, T.(ksphaMeasured'), T.(datatime); 
        recon_terms=recon, csm=Complex{T}.(csm), fieldmap=T.(b0), mask=mask, 
        use_gpu=use_gpu, verbose=verbose, nBlock=100);
CUDA.@time x2 = recon_HOOp(Op2, Complex{T}.(kdata), Complex{T}.(weight), recParams);
fig2 = plt_image(abs.(x2); title="HighOderOp: Array-based, 1 GPU, with mask", vmaxp=99.9, width=20, height=20)

@info "HighOderOp: kernel-based, 1 GPU"
GC.gc(); CUDA.reclaim();
Op3 = HighOrderOp_Kernel(gridding, T.(ksphaMeasured'), T.(datatime); 
        recon_terms=recon, csm=Complex{T}.(csm), fieldmap=T.(b0), 
        use_gpu=use_gpu, verbose=verbose, gpus=[0]);
CUDA.@time x3 = recon_HOOp(Op3, Complex{T}.(kdata), Complex{T}.(weight), recParams);
fig3 = plt_image(abs.(x3); title="HighOderOp: Kernel-based, 1 GPU", vmaxp=99.9, width=20, height=20)

@info "HighOderOp: kernel-based, 8 GPUs"
GC.gc(); CUDA.reclaim();
Op4 = HighOrderOp_Kernel(gridding, T.(ksphaMeasured'), T.(datatime); 
        recon_terms=recon, csm=Complex{T}.(csm), fieldmap=T.(b0), 
        use_gpu=use_gpu, verbose=verbose, gpus=[0,1,2,3,4,5,6,7]);
CUDA.@time x4 = recon_HOOp(Op4, Complex{T}.(kdata), Complex{T}.(weight), recParams);
fig4 = plt_image(abs.(x4); title="HighOderOp: Kernel-based, 8 GPUs", vmaxp=99.9, width=20, height=20)

@info "HighOderOp: kernel-based, 8 GPUs, with mask"
GC.gc(); CUDA.reclaim();
Op5 = HighOrderOp_Kernel(gridding, T.(ksphaMeasured'), T.(datatime); 
        recon_terms=recon, csm=Complex{T}.(csm), fieldmap=T.(b0), mask=mask,
        use_gpu=use_gpu, verbose=verbose, gpus=[0,1,2,3,4,5,6,7]);
CUDA.@time x5 = recon_HOOp(Op5, Complex{T}.(kdata), Complex{T}.(weight), recParams);
fig5 = plt_image(abs.(x5); title="HighOderOp: Kernel-based, 8 GPUs, with mask", vmaxp=99.9, width=20, height=20)


###########################################################################
# Benckmark test
###########################################################################
@info "Benckmark test - HighOderOp: Array-based, 1 GPU"
CUDA.device!(0)
test = () -> begin
    CUDA.@sync recon_HOOp(Op1, Complex{T}.(kdata), Complex{T}.(weight), recParams)
end
t_array = run(@benchmarkable test() samples=10 evals=1 seconds=3600 setup=(GC.gc(); CUDA.reclaim()))


@info "Benckmark test - HighOderOp: Array-based, 1 GPU, mask"
CUDA.device!(0)
test = () -> begin
    CUDA.@sync recon_HOOp(Op2, Complex{T}.(kdata), Complex{T}.(weight), recParams)
end
t_array_mask = run(@benchmarkable test() samples=10 evals=1 seconds=3600 setup=(GC.gc(); CUDA.reclaim()))


@info "Benckmark test - HighOderOp: kernel-based, 1 GPU"
test = () -> begin
    CUDA.@sync recon_HOOp(Op3, Complex{T}.(kdata), Complex{T}.(weight), recParams)
end
t_kernel = run(@benchmarkable test() samples=10 evals=1 seconds=3600 setup=(GC.gc(); CUDA.reclaim()))


@info "Benckmark test - HighOderOp: kernel-based, 8 GPUs"
test = () -> begin
    CUDA.@sync recon_HOOp(Op4, Complex{T}.(kdata), Complex{T}.(weight), recParams)
end
t_kernel_8 = run(@benchmarkable test() samples=10 evals=1 seconds=3600 setup=(GC.gc(); CUDA.reclaim()))


@info "Benckmark test - HighOderOp: kernel-based, 8 GPUs, with mask"
test = () -> begin
    CUDA.@sync recon_HOOp(Op5, Complex{T}.(kdata), Complex{T}.(weight), recParams)
end
t_kernel_8_mask = run(@benchmarkable test() samples=10 evals=1 seconds=3600 setup=(GC.gc(); CUDA.reclaim()))

# Judgement
m_array         = median(t_array)
m_array_mask    = median(t_array_mask)
m_kernel        = median(t_kernel)
m_kernel_8      = median(t_kernel_8)
m_kernel_8_mask = median(t_kernel_8_mask)

judgement = judge(m_kernel, m_array)
judgement = judge(m_kernel_8, m_kernel)
judgement = judge(m_kernel_8, m_array)
judgement = judge(m_kernel_8_mask, m_array)
