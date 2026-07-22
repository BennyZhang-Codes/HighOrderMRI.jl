using CUDA
using AbstractNFFTs

const T = Float32


function highorder_lowrank_test_data(; nDyn::Int=2)
    nX, nY, nZ = 4, 4, 1
    nSam, nCha, nTerm = 12, 2, 9
    recon_terms = "111"

    grid = Grid(nX, nY, nZ, one(T), one(T), one(T))
    kspha = zeros(T, nTerm, nSam, nDyn)
    times = zeros(T, nSam, nDyn)

    for dyn = 1:nDyn
        times[:, dyn] .= range(zero(T), T(1e-3); length=nSam)
        kspha[2, :, dyn] .= range(T(-0.2), T(0.2); length=nSam)
        kspha[3, :, dyn] .= T(0.05 * (dyn - 1))
        kspha[5, :, dyn] .= T(1e-3 * dyn)
    end

    fieldmap = reshape(T.(0:(nX * nY - 1)) ./ T(nX * nY), nX, nY)
    csm = ones(Complex{T}, nX, nY, nCha)
    mask = trues(nX, nY)
    mask[1, 1] = false

    return grid, kspha, times, fieldmap, csm, mask, recon_terms
end


function rsvd_test_problem()
    grid, kspha, times, fieldmap, _, mask, _ = highorder_lowrank_test_data()
    nVox = sum(mask)
    nSam = size(kspha, 2)
    L_rank = 3
    p_oversample = 3
    L_total = L_rank + p_oversample

    fieldmap_masked = vec(fieldmap)[vec(mask)]
    bf = HighOrderMRI.basisfunc_spha(
        grid.x[vec(mask)],
        grid.y[vec(mask)],
        grid.z[vec(mask)],
        collect(1:size(kspha, 1)),
    )
    bf_err = bf[:, 5:end]
    kspha_err = kspha[5:end, :, 1]
    phase_ref =
        reshape(times[:, 1], :, 1) * reshape(fieldmap_masked, 1, :) +
        transpose(kspha_err) * transpose(bf_err)
    E_ref = @. cis(T(2π) * phase_ref)

    return (;
        grid,
        kspha,
        times,
        mask,
        nVox,
        nSam,
        L_rank,
        p_oversample,
        L_total,
        fieldmap_masked,
        bf_err,
        kspha_err,
        E_ref,
    )
end
