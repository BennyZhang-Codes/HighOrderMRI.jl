using HighOrderMRI
using RegularizedLeastSquares
using MAT
using CUDA
CUDA.device!(0)

T             = Float32;
path          = joinpath(@__DIR__, "")
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

shift_x = 0
shift_y = -1
csm  = permutedims(reverse(circshift(csm , (shift_x, shift_y, 0)), dims=(1)), [2,1,3])
b0   = permutedims(reverse(circshift(b0  , (shift_x, shift_y   )), dims=(1)), [2,1  ])
mask = permutedims(reverse(circshift(mask, (shift_x, shift_y   )), dims=(1)), [2,1  ])

# comparion of the field dynamics with or without synchronization
# plt_ksphas([InterpTrajTime(ksphaMeasured, dt_Measured, startMeasured, datatime), InterpTrajTime(ksphaMeasured, dt_Measured, startMeasured+tauMeasured*dt_adc, datatime)], dt_adc)
# plt_ksphas([InterpTrajTime(ksphaNominal , dt_Nominal , startNominal , datatime), InterpTrajTime(ksphaNominal , dt_Nominal , startNominal +tauNominal *dt_adc, datatime)], dt_adc)
datatime      = vec(datatime);
ksphaMeasured = InterpTrajTime(ksphaMeasured, dt_Measured, startMeasured + tauMeasured * dt_adc, datatime);
ksphaNominal  = InterpTrajTime(ksphaNominal , dt_Nominal , startNominal  + tauNominal  * dt_adc, datatime);


## prepare some parameters for reconstruction
# 1. gridding 
Δx, Δy, Δz    = T.(FOV ./ matrixSize);
nX, nY, nZ    = matrixSize;
gridding      = Grid(nX, nY, nZ, Δx, Δy, Δz)

# 2. sampling density 
weightMeasured = samplingDensity(ksphaMeasured'[2:3,:], (nX, nY));
weightNominal  = samplingDensity( ksphaNominal'[2:3,:], (nX, nY));

arrayType = CuArray;
verbose   = false;
nBlock    = 20;

recParams = Dict{Symbol,Any}()
recParams[:reconSize]      = (nX, nY)
recParams[:reg]            = L2Regularization(1.e-9)
recParams[:iterations]     = 20
recParams[:solver]         = CGNR


## recon with field dynamics measured by Dynamic Field Camera
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
for (idx, label, recon, kdata, weight, b0) in zip(idxs, labelMeasured, recons, kdatas, weights, b0s)
    @info "[$(idx)] $(label) $(recon)"
    HOOp = HighOrderOp(gridding, T.(ksphaMeasured'), T.(datatime); recon_terms=recon, 
        nBlock=nBlock, csm=Complex{T}.(csm), fieldmap=T.(b0), arrayType=arrayType, verbose=verbose);
    @time x = recon_HOOp(HOOp, arrayType(Complex{T}.(kdata)), arrayType(Complex{T}.(weight)), recParams);
    imgMeasured[:, :, idx] = mapslices(rotl90, x; dims=(1,2));
    fig = plt_image(abs.(x); title=label, vmaxp=99.9, width=8)
    # fig.savefig("$(path)/result/$(data_mat[1:end-4])_$(label).png", dpi=300, transparent=false, bbox_inches="tight", pad_inches=0.0)
end


## recon with nominal trajectory
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
        nBlock=nBlock, csm=Complex{T}.(csm), fieldmap=T.(b0), arrayType=arrayType, verbose=verbose);
    @time x = recon_HOOp(HOOp, arrayType(Complex{T}.(kdata)), arrayType(Complex{T}.(weight)), recParams);
    imgNominal[:, :, idx] = mapslices(rotl90, x; dims=(1,2));
    fig = plt_image(abs.(x); title=label, vmaxp=99.9, width=8)
    # fig.savefig("$(path)/result/$(data_mat[1:end-4])_$(label).png", dpi=300, transparent=false, bbox_inches="tight", pad_inches=0.0)
end

## Coil compression
kdata = data["kdata"]
nVirtualCoils = 16
kdata_cc, csm_cc, coil_transform = compress_coils(Complex{T}.(kdata), Complex{T}.(csm); data_coil_dim=2, csm_coil_dim=3, n_virtual_coils=nVirtualCoils)
@info "Coil compression" physical_coils=size(coil_transform, 1) virtual_coils=size(coil_transform, 2) retained_energy=coil_transform.retained_energy

kdata_cc = kdata_cc ./ exp.(2π*1im.*k0_ecc)';
kdata_cc = kdata_cc .* exp.(-2π*1im.*ksphaMeasured[:, 1]);

label  = "Measured_CoilCompressed_$(nVirtualCoils)";
recon  = "0111";
weight = weightMeasured; 

@info "$(label) $(recon)"
HOOp = HighOrderOp(gridding, T.(ksphaMeasured'), T.(datatime); recon_terms=recon, 
    nBlock=nBlock, csm=Complex{T}.(csm_cc), fieldmap=T.(b0), arrayType=arrayType, verbose=verbose);
@time x = recon_HOOp(HOOp, arrayType(Complex{T}.(kdata_cc)), arrayType(Complex{T}.(weight)), recParams);
img = mapslices(rotl90, x; dims=(1,2));
fig = plt_image(abs.(img); title=label, vmaxp=99.9, width=8)
# fig.savefig("$(path)/result/$(data_mat[1:end-4])_$(label).png", dpi=300, transparent=false, bbox_inches="tight", pad_inches=0.0)
