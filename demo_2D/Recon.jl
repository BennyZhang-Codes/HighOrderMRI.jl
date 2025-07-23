using HighOrderMRI
using RegularizedLeastSquares
using MRIOperators, MRIReco
using MAT
using CUDA
CUDA.device!(0)

T             = Float64;
path          = joinpath(@__DIR__)
# data_mat      = "7T_2D_Spiral_1p0_200_r4.mat" 
data_mat      = "7T_2D_EPI_1p0_200_r4.mat"  
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
dt_Nominal    = data["dt_Nominal"];
ksphaMeasured = data["ksphaMeasured"];     # coefficients of the field dynamics
startMeasured = data["startMeasured"];     # start time of the field dynamics

ksphaNominal  = data["ksphaNominal"];       # nominal kspace trajectory, kx, ky
startNominal  = data["startNominal"];       # start time of the nominal trajectory

tauNominal    = data["tauNominal"];        # synchronization delay between the nominal trajectory and the MRI data  
tauMeasured   = data["tauMeasured"];       # synchronization delay between the Measured trajectory and the MRI data

dt_adc        = data["dt_adc"];

# comparion of the field dynamics with or without synchronization
# plt_ksphas([InterpTrajTime(ksphaMeasured, dt_Measured, startMeasured, datatime), InterpTrajTime(ksphaMeasured, dt_Measured, startMeasured+tauMeasured*dt_adc, datatime)], dt_adc)
# plt_ksphas([InterpTrajTime(ksphaNominal , dt_Nominal , startNominal , datatime), InterpTrajTime(ksphaNominal , dt_Nominal , startNominal +tauNominal *dt_adc, datatime)], dt_adc)
datatime      = vec(datatime);
ksphaMeasured = InterpTrajTime(ksphaMeasured, dt_Measured, startMeasured + tauMeasured * dt_adc, datatime);
ksphaNominal  = InterpTrajTime(ksphaNominal , dt_Nominal , startNominal  + tauNominal  * dt_adc, datatime);

# prepare some parameters for reconstruction
# 1. gridding 
Δx, Δy, Δz    = T.(FOV ./ matrixSize);
nX, nY, nZ    = matrixSize;
gridding      = Grid(nX, nY, nZ, Δx, Δy, Δz; exchange_xy=true, reverse_x=false, reverse_y=true)

# 2. sampling density 
weightMeasured = SampleDensity(ksphaMeasured'[2:3,:], (nX, nY));
weightNominal  = SampleDensity( ksphaNominal'[2:3,:], (nX, nY));

use_gpu = true;
verbose = false;
nBlock  = 40;

solver = CGNR; reg = L2Regularization(1.e-6); iter = 10;
recParams = Dict{Symbol,Any}()
recParams[:reconSize]      = (nX, nY)
recParams[:reg] = reg  # ["L2", "L1", "L21", "TV", "LLR", "Positive", "Proj", "Nuclear"]
recParams[:iterations]     = iter
recParams[:solver]         = solver
recParams[:csm] = csm



###########################################################################
# recon with field dynamics measured by Dynamic Field Camera
###########################################################################
kdata = data["kdata"];
kdata = kdata .* exp.(-2π*1im.*k0_ecc)';
kdata = kdata .* exp.(-2π*1im.*ksphaMeasured[:, 1]);

labelMeasured = [    "Measured"];
recons        = [        "0111"];
kdatas        = [         kdata];
weights       = [weightMeasured]; 
b0s           = [            b0];
imgMeasured   = Array{Complex{T},3}(undef, nX, nY, length(labelMeasured));
idxs           = collect(1:length(labelMeasured));
for (idx, label, recon, kdata, weight, b0) in zip(idxs, labelMeasured, recons, kdatas, weights, b0s)
    @info "[$(idx)] $(label) $(recon)"
    HOOp = HighOrderOp(gridding, T.(ksphaMeasured'), T.(datatime); recon_terms=recon, 
        nBlock=nBlock, csm=Complex{T}.(csm), fieldmap=T.(b0), use_gpu=use_gpu, verbose=verbose);
    @time x = recon_HOOp(HOOp, Complex{T}.(kdata), Complex{T}.(weight), recParams);
    imgMeasured[:, :, idx] = x;
    fig = plt_image(abs.(x); title=label, vmaxp=99.9)
    fig.savefig("$(path)/result/$(data_mat[1:end-4])_$(label).png", dpi=300, transparent=false, bbox_inches="tight", pad_inches=0.0)
end


###########################################################################
# recon with nominal trajectory
###########################################################################
kdata = data["kdata"];

labelNominal  = [    "Nominal"];
recons        = [        "010"];
kdatas        = [        kdata];
weights       = [weightNominal]; 
b0s           = [           b0];
imgNominal    = Array{Complex{T},3}(undef, nX, nY, length(labelNominal));
idxs           = collect(1:length(labelNominal));
for (idx, label, recon, kdata, weight, b0) in zip(idxs, labelNominal, recons, kdatas, weights, b0s)
    @info "[$(idx)] $(label) $(recon)"
    HOOp = HighOrderOp(gridding, T.(ksphaNominal'), T.(datatime); recon_terms=recon, 
        nBlock=nBlock, csm=Complex{T}.(csm), fieldmap=T.(b0), use_gpu=use_gpu, verbose=verbose);
    @time x = recon_HOOp(HOOp, Complex{T}.(kdata), Complex{T}.(weight), recParams);
    imgNominal[:, :, idx] = x;
    fig = plt_image(abs.(x); title=label, vmaxp=99.9)
    fig.savefig("$(path)/result/$(data_mat[1:end-4])_$(label).png", dpi=300, transparent=false, bbox_inches="tight", pad_inches=0.0)
end

###########################################################################
# recon with fast Higher Order Op
###########################################################################

# NOTE: Sense maps and B0 maps are rotated by 90 degrees to match the orientation of the k-space data. This will be fixed soon. 
kdata = data["kdata"];
kdata = kdata .* exp.(-2π*1im.*k0_ecc)';
kdata = kdata .* exp.(-2π*1im.*ksphaMeasured[:, 1]);

gridding      = Grid(nX, nY, nZ, Δx, Δy, Δz; exchange_xy=true, reverse_x=false, reverse_y=true)

labelMeasured = [    "Measured fast"];
recons        = [        "0111"];
kdatas        = [         kdata];
weights       = [weightMeasured]; 
b0s           = [            b0];
imgMeasuredFast   = Array{Complex{T},3}(undef, nX, nY, length(labelMeasured));
idxs          = collect(1:length(labelMeasured));

rot_csm = cat([rotl90(Complex{T}.(csm)[:, :, i]) for i in 1:size(csm, 3)]..., dims=3)

for (idx, label, recon, kdata, weight, b0) in zip(idxs, labelMeasured, recons, kdatas, weights, b0s)
    @info "[$(idx)] $(label) $(recon)"
    fHOOp = fastHighOrderOp(gridding, T.(ksphaMeasured'), T.(datatime); recon_terms=recon, 
        nBlock=nBlock, csm=Complex{T}.(rot_csm), fieldmap=rotl90(T.(b0)), use_gpu=use_gpu, verbose=verbose);
    @time x = recon_fHOOp(fHOOp, Complex{T}.(kdata), Complex{T}.(weight), recParams);
    x = rotr90(x);
    imgMeasuredFast[:, :, idx] = x;
    fig = plt_image(abs.(x); title=label, vmaxp=99.9)
    fig.savefig("$(path)/result/$(data_mat[1:end-4])_$(label).png", dpi=300, transparent=false, bbox_inches="tight", pad_inches=0.0)
end

diff_im = angle.(imgMeasuredFast[:, :, 1])- angle.(imgMeasured[:, :, 1])

fig = plt_image(diff_im; title="Difference", vmaxp=99.9)