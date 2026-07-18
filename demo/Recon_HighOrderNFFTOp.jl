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


## recon with field dynamics measured by Dynamic Field Camera
using PyPlot
pygui(true)

kdata = data["kdata"];
kdata = kdata ./ exp.(2π*1im.*k0_ecc)';
kdata = kdata .* exp.(-2π*1im.*ksphaMeasured[:, 1]);

weight = weightMeasured;

verbose = true;
# nBlock  = 20;
recon   = "0111";

solver = CGNR; reg = L2Regularization(1.e-9); iter = 20;
recParams = Dict{Symbol,Any}()
recParams[:reconSize]      = (nX, nY)
recParams[:reg] = reg  # ["L2", "L1", "L21", "TV", "LLR", "Positive", "Proj", "Nuclear"]
recParams[:iterations]     = iter
recParams[:solver]         = solver
## HighOrderLowRankOp
shift_x = 0
shift_y = -1

gridding  = Grid(nX, nY, nZ, Δx, Δy, Δz; exchange_xy=false, reverse_x=true, reverse_y=true)

gridding  = Grid(nX=nX, nY=nY, nZ=nZ, Δx=Δx, Δy=Δy, Δz=Δz, x=gridding.x, y=gridding.y, z=gridding.z)


arrayType = CuArray;
@CUDA.time HOOp = HighOrderLowRankOp(gridding, T.(ksphaMeasured'), T.(datatime); 
                recon_terms=recon, 
                csm=Complex{T}.(permutedims(reverse(circshift(csm, (shift_x, shift_y, 0)), dims=(2)), [2,1,3])), 
                fieldmap=T.(permutedims(reverse(circshift(b0, (shift_x, shift_y)), dims=(2)), [2,1])), 
                # mask=permutedims(reverse(circshift(mask, (shift_x, shift_y)), dims=(2)), [2,1]), 
                verbose=verbose,
                L_rank = 15,
                arrayType=arrayType);

@CUDA.time x = recon_HOOp(HOOp, arrayType{eltype(HOOp)}(kdata), arrayType{eltype(HOOp)}(weight), recParams);
# @time x = recon_HOOp(HOOp, Complex{T}.(kdata), Complex{T}.(weight), recParams);
# imgMeasured[:, :, idx] = x;
x1 = mapslices(rotr90, Array(x); dims=(1,2));
fig = plt_image(abs.(x1); vmaxp=99.9, width=20, height=20)
# fig.savefig("$(path)/$(data_mat[1:end-4])_MaxGIRFOp_measured_wb0_1100_L15_x$(shift_x)_y$(shift_y).png", dpi=300, transparent=false, bbox_inches="tight", pad_inches=0.0)
# fig = plt_image(angle.(x1); vmaxp=99.9)                                                                                                                                                