using HighOrderMRI
using RegularizedLeastSquares
using MAT
using CUDA
CUDA.device!(0)

T             = Float32;
path          = joinpath(@__DIR__, "")
data_mat      = "7T_2D_Spiral_1p0_200_r4.mat" 
# data_mat      = "7T_2D_EPI_1p0_200_r4.mat"  
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
nBlock  = 20;

solver = CGNR; reg = L2Regularization(1.e-9); iter = 20;
recParams = Dict{Symbol,Any}()
recParams[:reconSize]      = (nX, nY)
recParams[:reg] = reg  # ["L2", "L1", "L21", "TV", "LLR", "Positive", "Proj", "Nuclear"]
recParams[:iterations]     = iter
recParams[:solver]         = solver


## recon with field dynamics measured by Dynamic Field Camera
kdata = data["kdata"];
kdata = kdata ./ exp.(2π*1im.*k0_ecc)';
kdata = kdata .* exp.(-2π*1im.*ksphaMeasured[:, 1]);

## NFFT
using MRIReco
using PyPlot 
pygui(true)

csm = reshape(csm, nX, nY, nZ,  size(csm)[end]);
kspha = ksphaMeasured;

C = maximum(2*abs.(kspha[:,2:4][:]));  #Normalize k-space to -.5 to .5 for NUFFT
C = 1000;

nSample_adc = 29000;


k_norm = T.(kspha[:,2:3]' ./ C)

shift_x = 1.0
shift_y = 1.0

tr = Trajectory("2dEPI", k_norm,  T.(datatime), T.(0e-3), T.(nSample_adc*dt_adc), 1, nSample_adc, 1, false, false);

# tr = Trajectory(T.(kspha[:,2:3]' ./ C), 1, nSample_adc; times=T.(datatime), TE=T.(0e-3), AQ=T.(nSample_adc*dt_adc), numSlices=1, cartesian=false, circular=false);
dat         = Array{Array{Complex{T},2},3}(undef,1,1,1);
dat[1,1,1]  = kdata[:,:];
acqData     = AcquisitionData(tr, dat, encodingSize=(nX, nY));
##
solver = CGNR; reg = L2Regularization(1.e-9); iter = 20;
params = Dict{Symbol, Any}()
params[:arrayType]   = CuArray
params[:reco]        = "multiCoil"
params[:reconSize]   = (nX, nY)
params[:reg]         = reg  
params[:iterations]  = iter
params[:solver]      = solver
params[:densityWeighting] = true;
params[:senseMaps]   = Complex{T}.(permutedims(reverse(circshift(csm, (0, -shift_y, 0, 0)), dims=(2)), [2,1,3,4])); #permutedims(reverse(csm, dims=(1,2,3)), [1,2,3,4]));
params[:correctionMap] = Complex{T}.(-1im*2π*permutedims(reverse(circshift(b0, (0, -shift_y)), dims=(2)), [2,1]));
params[:alpha] = 1.75
params[:m] = 4.0
params[:K] = 28
# params[:method] = "nfft"
# AbstractNFFTs.set_active_backend!(NFFT.backend())
# AbstractNFFTs.set_active_backend!(NonuniformFFTs.backend())
# AbstractNFFTs.active_backend()

@time Ireco = reconstruction(acqData, params);
x = Array(Ireco[:,:,1,1,1,1]);
x = mapslices(rotr90, x; dims=(1,2));
fig = plt_image(abs.(x);  vmaxp=99.9, width=20, height=20)
# fig.savefig("$(path)/$(data_mat[1:end-4])_Nufft_measured_wb0.png", dpi=300, transparent=false, bbox_inches="tight", pad_inches=0.0)

##
HOOp = HighOrderOp_Kernel(gridding, T.(ksphaMeasured'), T.(datatime); recon_terms="0111", 
    nBlock=nBlock, csm=Complex{T}.(csm), fieldmap=T.(b0), use_gpu=use_gpu, verbose=verbose);
@time x = recon_HOOp(HOOp, Complex{T}.(kdata), Complex{T}.(weightMeasured), recParams);
fig = plt_image(abs.(x); vmaxp=99.9, width=20, height=20)
fig.savefig("$(path)/$(data_mat[1:end-4])_HighOrderOp_measured_wb0_0111.png", dpi=300, transparent=false, bbox_inches="tight", pad_inches=0.0)