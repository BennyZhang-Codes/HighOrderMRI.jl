@info size(ksphaMeasured)

using LinearAlgebra, Plots, PlotlyJS, MosaicViews, HighOrderMRI

plotlyjs()

LinearAlgebra.BLAS.set_num_threads(64)

using HighOrderMRI
using RegularizedLeastSquares
using MRIOperators, MRIReco
using MAT
using CUDA
CUDA.device!(0)

T = Float64;
path = joinpath(@__DIR__)
data_mat = "7T_2D_Spiral_1p0_200_r4.mat"
# data_mat = "demo_2D/7T_2D_EPI_1p0_200_r4.mat"
data_file = joinpath(path, data_mat)

@info "data file: $(data_file)"
data = matread(data_file);

csm = data["gre_csm"];            # coil sensitivity map
b0 = data["gre_b0"];             # ΔB0 map
mask = data["gre_mask"];           # mask

kdata = data["kdata"];              # spiral k-space data
datatime = data["datatime"];           # time stamps of k-space data (Check by div of AQ window length by nSamples)
matrixSize = data["matrixSize"];         # matrix size
FOV = data["FOV"];                # field of view

# extended time constant model (bunch of e^-exp filters (CSC or hardware parameter that controls this))
# Check basic corrections node to see that it can be turned on or off or if it's applied at that point 
k0_ecc = data["k0_adc"];             # b0 compensation of scanner from ECC model
dt_Measured = data["dt_Measured"];
dt_Nominal = data["dt_Nominal"];
ksphaMeasured = data["ksphaMeasured"];     # coefficients of the field dynamics
startMeasured = data["startMeasured"];     # start time of the field dynamics

ksphaNominal = data["ksphaNominal"];       # nominal kspace trajectory, kx, ky
startNominal = data["startNominal"];       # start time of the nominal trajectory

tauNominal = data["tauNominal"];        # synchronization delay between the nominal trajectory and the MRI data  
tauMeasured = data["tauMeasured"];       # synchronization delay between the Measured trajectory and the MRI data

dt_adc = data["dt_adc"];

# comparion of the field dynamics with or without synchronization
# plt_ksphas([InterpTrajTime(ksphaMeasured, dt_Measured, startMeasured, datatime), InterpTrajTime(ksphaMeasured, dt_Measured, startMeasured+tauMeasured*dt_adc, datatime)], dt_adc)
# plt_ksphas([InterpTrajTime(ksphaNominal , dt_Nominal , startNominal , datatime), InterpTrajTime(ksphaNominal , dt_Nominal , startNominal +tauNominal *dt_adc, datatime)], dt_adc)
datatime = vec(datatime);
ksphaMeasured = InterpTrajTime(ksphaMeasured, dt_Measured, startMeasured + tauMeasured * dt_adc, datatime);
ksphaNominal = InterpTrajTime(ksphaNominal, dt_Nominal, startNominal + tauNominal * dt_adc, datatime);

# prepare some parameters for reconstruction
# 1. gridding 
Δx, Δy, Δz = T.(FOV ./ matrixSize);
nX, nY, nZ = matrixSize;
gridding = Grid(nX, nY, nZ, Δx, Δy, Δz; exchange_xy=true, reverse_x=false, reverse_y=true)

# 2. sampling density 
weightMeasured = SampleDensity(ksphaMeasured'[2:3, :], (nX, nY));
weightNominal = SampleDensity(ksphaNominal'[2:3, :], (nX, nY));

bf = basisfunc_spha(gridding.x,gridding.y,gridding.z,1:16)

non_indices = 5:16
crop_idcs = 1:4000

kspha_cropped = ksphaMeasured[crop_idcs,:]
bf_cropped = bf[:,non_indices]

# set up blocksize
stream_block_size = 1000

full_mat =  cispi.(-2 * bf_cropped * kspha_cropped[:,bf_choice]')

bls_interp, cls_interp = HighOrderMRI.get_bl_cl_coeffs_interp(gridding, kspha_cropped)
bls_isvd, cls_isvd = HighOrderMRI.get_bl_cl_coeffs_isvd(gridding, kspha_cropped)
bls_turnstile, cls_turnstile = HighOrderMRI.get_bl_cl_coeffs_turnstile(gridding, kspha_cropped)

bf_choice = 5:16

@info size(kspha_cropped)
@info size(bf)

bf_cropped = bf[:,bf_choice]
# set up blocksize
stream_block_size = 50

num_L = 50

# Make the turnstile model (streaming rows off my matrix)
row_iterator = HighOrderMRI.row_blocks(kspha_cropped[:,bf_choice], bf_cropped; blocksize=stream_block_size)

# Set up the sketching matrices
Ω, γ = HighOrderMRI.make_standard_normal_sketching_matrices(size(kspha_cropped,1), size(bf_cropped,1), stream_block_size, stream_block_size)
Ψ, Φ = HighOrderMRI.make_standard_normal_sketching_matrices(size(kspha_cropped,1), size(bf_cropped,1), 2*stream_block_size, 2*stream_block_size)

X,Y,Z = HighOrderMRI.streaming_sketch(row_iterator, γ, Ω, Ψ, Φ)
U,s,V = HighOrderMRI.svd_from_sketch(X,Y,Z,Φ,Ψ)

# UsV' = transpose(full_mat)
# conj.(V) * s * transpose(U) yields full_mat

# bls = reshape((V[:,1:num_L].* s[1:num_L]')[:,1:num_L], 200, 200, 1, num_L)
bls = reshape(conj.(V[:,1:num_L])*Diagonal(s[1:num_L]),gridding.nX, gridding.nY,1,num_L)
cls = U[:, 1:num_L]

bls_turnstile = bls_turnstile .* exp(-1im*pi/2)
cls_turnstile = cls_turnstile .* exp(-1im * pi/2)
cls_turnstile = conj.(cls_turnstile)
bls_turnstile = conj.(bls_turnstile)

Plots.heatmap(mosaicview(angle.(bls_interp), nrow = 2, ncol = 5); colormap = :hsv)
Plots.heatmap(mosaicview(angle.(bls_isvd), nrow = 2, ncol = 5); colormap = :hsv)
Plots.heatmap(mosaicview(angle.(bls_turnstile), nrow = 2, ncol = 5); colormap = :hsv)
Plots.heatmap(mosaicview(angle.(bls[:,:,1,1:10]),nrow=2,ncol=5);colormap=:hsv)

Plots.plot(real.(cls_interp[:,1]))
Plots.plot(imag.(cls_interp[:,1]))

Plots.plot(real.(cls_isvd[:,1]))
Plots.plot(imag.(cls_isvd[:,1]))

Plots.plot(real.(cls_turnstile[:,1]))
Plots.plot(imag.(cls_turnstile[:,1]))

Plots.plot(real.(cls[:,1]))
Plots.plot(imag.(cls[:,1]))

Plots.plot(unwrap(angle.(cls_turnstile[:,1])))
Plots.plot(unwrap(angle.(cls_isvd[:,1])))
Plots.plot(unwrap(angle.(cls_interp[:,1])))

Plots.plot(unwrap(angle.(cls[:,1])))
