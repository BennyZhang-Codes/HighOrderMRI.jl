recon  = "0111";
weight = weightMeasured; 

kdata = data["kdata"];
kdata = kdata ./ exp.(2π*1im.*k0_ecc)';
kdata = kdata .* exp.(-2π*1im.*ksphaMeasured[:, 1]);

# Coil compression
nVirtualCoils = 16
kdata_cc, csm_cc, coil_transform = compress_coils(Complex{T}.(kdata), Complex{T}.(csm); data_coil_dim=2, csm_coil_dim=3, n_virtual_coils=nVirtualCoils)
@info "Coil compression" physical_coils=size(coil_transform, 1) virtual_coils=size(coil_transform, 2) retained_energy=coil_transform.retained_energy

## 1. HighOrderOp
begin
    label  = "Measured_HighOrderOp_1111";
    @CUDA.time begin
        HOOp = HighOrderOp(gridding, T.(ksphaMeasured'), T.(datatime); 
            recon_terms=recon, 
            nBlock=nBlock, 
            csm=Complex{T}.(csm), 
            fieldmap=T.(b0), 
            arrayType=arrayType, 
            verbose=verbose)
        x = recon_HOOp(HOOp, arrayType(Complex{T}.(kdata)), arrayType(Complex{T}.(weight)), recParams)
    end;
    fig = plt_image(abs.(mapslices(rotl90, x; dims=(1,2))); title=label, vmaxp=99.9, width=8)
end

begin
    label  = "Measured_HighOrderOp_1111_cc";
    @CUDA.time begin
        HOOp = HighOrderOp(gridding, T.(ksphaMeasured'), T.(datatime); 
            recon_terms=recon, 
            nBlock=nBlock, 
            csm=Complex{T}.(csm_cc), 
            fieldmap=T.(b0), 
            arrayType=arrayType, 
            verbose=verbose)
        x = recon_HOOp(HOOp, arrayType(Complex{T}.(kdata_cc)), arrayType(Complex{T}.(weight)), recParams)
    end;
    fig = plt_image(abs.(mapslices(rotl90, x; dims=(1,2))); title=label, vmaxp=99.9, width=8)
end
# fig.savefig("$(path)/result/$(data_mat[1:end-4])_$(label).png", dpi=300, transparent=false, bbox_inches="tight", pad_inches=0.0)

## 2. HighOrderKernelOp
begin
    label  = "Measured_HighOrderKernelOp_1111";
    @CUDA.time begin
        HOOp_Kernel = HighOrderKernelOp(gridding, T.(ksphaMeasured'), T.(datatime);
            recon_terms=recon, 
            csm=Complex{T}.(csm), 
            fieldmap=T.(b0), 
            mask=mask, 
            arrayType=arrayType, 
            gpus=[0,1,2,3,4,5,6,7],
            verbose=verbose)
        x = recon_HOOp(HOOp_Kernel, Complex{T}.(kdata), Complex{T}.(weight), recParams)
    end;
    fig = plt_image(abs.(mapslices(rotl90, x; dims=(1,2))); title=label, vmaxp=99.9, width=8)
end

begin
    label  = "Measured_HighOrderKernelOp_1111_cc";
    @CUDA.time begin
        HOOp_Kernel = HighOrderKernelOp(gridding, T.(ksphaMeasured'), T.(datatime);
            recon_terms=recon, 
            csm=Complex{T}.(csm_cc), 
            fieldmap=T.(b0), 
            mask=mask, 
            arrayType=arrayType, 
            gpus=[0,1,2,3,4,5,6,7],
            verbose=verbose)
        x = recon_HOOp(HOOp_Kernel, Complex{T}.(kdata_cc), Complex{T}.(weight), recParams)
    end;
    fig = plt_image(abs.(mapslices(rotl90, x; dims=(1,2))); title=label, vmaxp=99.9, width=8)
end

## 3. HighOrderLowRankOp
begin
    label  = "Measured_HighOrderLowRankOp_1111";
    @CUDA.time begin
        @rebuild_HOOp HOOp_LowRank begin
            HighOrderLowRankOp(gridding, T.(ksphaMeasured'), T.(datatime); 
                recon_terms=recon, 
                csm=Complex{T}.(csm), 
                fieldmap=T.(b0), 
                mask=mask, 
                arrayType=arrayType, 
                gpus=[0,1,2,3,4,5,6,7], 
                L_rank=25,
                rsvd_seed=1234, 
                rsvd_chunk=4096, 
                rsvd_oversample=5, 
                rsvd_finalize=:gram,
                rsvd_backend=:kernel, 
                rsvd_distribution=:voxel,
                shared_rank_max=32, 
                shared_basis_tol=T(1e-2), 
                normal_distribution=:channel,
                verbose=verbose);
        end
        @CUDA.time x = recon_HOOp(HOOp_LowRank, arrayType{Complex{T}}(kdata), arrayType{Complex{T}}(weight), recParams)
    end;
    fig = plt_image(abs.(mapslices(rotl90, x; dims=(1,2))); title=label, vmaxp=99.9, width=8)
end


begin
    label  = "Measured_HighOrderLowRankOp_1111_cc";
    @CUDA.time begin
        @rebuild_HOOp HOOp_LowRank begin
            HighOrderLowRankOp(gridding, T.(ksphaMeasured'), T.(datatime); 
                recon_terms=recon, 
                csm=Complex{T}.(csm_cc), 
                fieldmap=T.(b0), 
                mask=mask, 
                arrayType=arrayType, 
                gpus=[0,1,2,3,4,5,6,7], 
                L_rank=25,
                rsvd_seed=1234, 
                rsvd_chunk=4096, 
                rsvd_oversample=5, 
                rsvd_finalize=:gram,
                rsvd_backend=:kernel, 
                rsvd_distribution=:voxel,
                shared_rank_max=32, 
                shared_basis_tol=T(1e-2), 
                normal_distribution=:channel,
                verbose=verbose);
        end
        @CUDA.time x = recon_HOOp(HOOp_LowRank, arrayType{Complex{T}}(kdata_cc), arrayType{Complex{T}}(weight), recParams)
    end;
    fig = plt_image(abs.(mapslices(rotl90, x; dims=(1,2))); title=label, vmaxp=99.9, width=8)
end