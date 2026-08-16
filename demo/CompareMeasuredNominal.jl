CUDA.device!(0)

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

    @CUDA.time HOOpKernel = HighOrderKernelOp(gridding, T.(ksphaMeasured'), T.(datatime);
        recon_terms=recon, 
        csm=Complex{T}.(csm), 
        fieldmap=T.(b0), 
        mask=mask, 
        arrayType=arrayType, 
        gpus=[0,1,2,3,4,5,6,7],
        verbose=verbose);

    @CUDA.time x = recon_HOOp(HOOpKernel, Complex{T}.(kdata), Complex{T}.(weight), recParams);
    imgMeasured[:, :, idx] = mapslices(rotl90, x; dims=(1,2));
    fig = plt_image(abs.(imgMeasured[:, :, idx]); title=label, vmaxp=99.9, width=8)
    fig.savefig("$(path)/result/$(data_mat[1:end-4])_$(label).png", dpi=300, transparent=false, bbox_inches="tight", pad_inches=0.0)
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
    @CUDA.time HOOpKernel = HighOrderKernelOp(gridding, T.(ksphaNominal'), T.(datatime);
        recon_terms=recon, 
        csm=Complex{T}.(csm), 
        fieldmap=T.(b0), 
        mask=mask, 
        arrayType=arrayType, 
        gpus=[0,1,2,3,4,5,6,7],
        verbose=verbose);

    @CUDA.time x = recon_HOOp(HOOpKernel, Complex{T}.(kdata), Complex{T}.(weight), recParams);

    imgNominal[:, :, idx] = mapslices(rotl90, x; dims=(1,2));
    fig = plt_image(abs.(imgNominal[:, :, idx]); title=label, vmaxp=99.9, width=8)
    fig.savefig("$(path)/result/$(data_mat[1:end-4])_$(label).png", dpi=300, transparent=false, bbox_inches="tight", pad_inches=0.0)
end
