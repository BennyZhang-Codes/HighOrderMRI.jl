using HighOrderMRI
using RegularizedLeastSquares
using MAT
using CUDA

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
