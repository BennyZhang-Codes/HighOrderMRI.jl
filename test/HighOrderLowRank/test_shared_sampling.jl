@testset "optional shared coefficient fitting and weighted compression" begin
    rng = MersenneTwister(34817)
    nVox, nTime, rank = 53, 41, 7
    q = randn(rng, ComplexF32, nTime, rank)
    # A nonorthogonal basis exercises both small Grams, rather than assuming I.
    basis = randn(rng, ComplexF32, nVox, rank) * Diagonal(Float32[.3,.5,.7,1,1.2,1.4,2])
    dw = rand(rng, Float64, nTime); dw[1:5:end] .= 0
    cw = rand(rng, Float64, nVox); cw[1:7:end] .= 0
    original = ComplexF64.(q) * ComplexF64.(basis)'
    weighted = sqrt.(dw) .* original .* transpose(sqrt.(cw))
    singular = svdvals(weighted)
    q_copy, b_copy = copy(q), copy(basis)
    for r in (3, rank)
        qr, br, info = HighOrderMRI.global_recompress_shared_basis(q,basis,0f0;
            rank=r,weights=dw,coil_weights=cw,chunk_size=11)
        reconstructed = ComplexF64.(qr) * ComplexF64.(br)'
        error = norm(sqrt.(dw).*(reconstructed-original).*transpose(sqrt.(cw)))/norm(weighted)
        expected = sqrt(sum(abs2,singular[r+1:end])/sum(abs2,singular))
        @test isapprox(error,expected; atol=2e-6)
        @test isapprox(info.relative_error,expected; atol=2e-6)
        @test info.rank == r
        @test q == q_copy && basis == b_copy
        r == rank && @test norm(reconstructed-original)/norm(original) < 3e-6
    end
    @test_throws ArgumentError HighOrderMRI.global_recompress_shared_basis(q,basis,0f0;rank=rank+1)
    @test_throws ArgumentError HighOrderMRI.global_recompress_shared_basis(q,basis,0f0;weights=-dw)
    @test_throws DimensionMismatch HighOrderMRI.global_recompress_shared_basis(q,basis,0f0;weights=dw[1:end-1])

    spatial = randn(rng,Float32,nVox,2) .* .12f0
    temporal = randn(rng,Float32,nTime,2) .* .18f0
    exact = cis.(2π.*(Float64.(temporal)*Float64.(spatial)'))
    b = ComplexF32.(svd(exact).V[:,1:6])
    fitted, info = HighOrderMRI.refit_shared_coefficients(temporal,spatial,b;
        sample_count=23,chunk_size=13,seed=11)
    reference = exact[:,info.fit_indices] * pinv(ComplexF64.(b[info.fit_indices,:])')
    @test norm(ComplexF64.(fitted)-reference)/norm(reference) < 2e-6
    check = HighOrderMRI.shared_basis_phase_check(fitted,b,temporal,spatial;
        exclude_voxels=info.fit_indices,time_samples=nTime,spatial_samples=nVox)
    @test check.spatial_training_disjoint
    heldout = setdiff(1:nVox,info.fit_indices)
    reference_error = norm(ComplexF64.(fitted)*ComplexF64.(b[heldout,:])'-exact[:,heldout])/norm(exact[:,heldout])
    @test isapprox(check.relative_error,reference_error;atol=1e-12)
    @test_throws ArgumentError HighOrderMRI.refit_shared_coefficients(temporal,spatial,b;sample_count=3)

    selections = HighOrderMRI.select_shared_readouts(spatial,temporal[1:39,:],13,3,2;seed=17)
    @test length(unique(selections)) == 2
    @test issorted(selections)
    @test HighOrderMRI.select_shared_readouts(spatial,temporal[1:39,:],13,3,3;seed=17) == 1:3
end

@testset "sampled shared construction retains the complete problem" begin
    rng = MersenneTwister(5913)
    grid = Grid(9,7,5,.001f0,.001f0,.001f0)
    ns,nd,nc = 37,3,3
    k = zeros(Float32,16,ns,nd)
    k[1,:,:] .= .03f0 .* randn(rng,Float32,ns,nd)
    k[2:4,:,:] .= 100f0 .* randn(rng,Float32,3,ns,nd)
    times = reshape(collect(range(.01f0,.013f0;length=ns*nd)),ns,nd)
    fieldmap = reshape(collect(range(-30f0,30f0;length=315)),9,7,5)
    csm = randn(rng,ComplexF32,9,7,5,nc)
    mask = trues(9,7,5); mask[1:17:end] .= false
    report = Ref{Any}()
    options = (;fieldmap,csm,mask,L_rank=3,rsvd_oversample=2,
        shared_rank_max=16,shared_basis_tol=1f-4,rsvd_seed=9)
    @test_throws ArgumentError HighOrderLowRankOp(grid,k,times;options...,shared_readout_count=2)
    @test_throws ArgumentError HighOrderLowRankOp(grid,k,times;options...,normal_method=:unknown)
    @test_throws ArgumentError HighOrderLowRankOp(grid,k,times;options...,normal_method=:batched)
    @test_throws ArgumentError HighOrderLowRankOp(grid,k,times;options...,
        global_basis_rank=2,global_basis_tol=.005f0)
    @test_throws ArgumentError HighOrderLowRankOp(grid,k,times;options...,
        global_basis_weights=ones(Float32,ns*nd))
    @test_throws ArgumentError HighOrderLowRankOp(grid,k,times;options...,
        method=:direct_shared, strategy=:per_readout,global_basis_rank=2)
    op = HighOrderLowRankOp(grid,k,times;options...,shared_readout_count=2,
        shared_fit_samples=64,shared_basis_report=report,global_basis_rank=2,
        global_basis_weights=collect(range(.1f0,1f0;length=ns*nd)),global_basis_coil_weighting=true)
    try
        @test (op.nSam,op.nDyn,op.nCha) == (ns,nd,nc)
        @test size(op.q) == (ns*nd,2)
        @test size(op,1) == ns*nd*nc
        @test op.normal_backend.method === :standard
        @test length(report[].selected_readouts) == 2
        @test report[].fit_samples == 64
        @test report[].phase_check.spatial_training_disjoint
        @test report[].compressed_phase_check.relative_error < .02
        indices = findall(vec(mask))
        fields = HighOrderMRI.basisfunc_spha(grid.x[indices],grid.y[indices],grid.z[indices],collect(1:16))
        phase = Float64.(reshape(k,16,:)')*Float64.(fields)' .+
            Float64.(vec(times))*Float64.(vec(fieldmap)[indices])'
        image = randn(rng,ComplexF32,315)
        expected = cis.(2π.*phase)*(ComplexF64.(image[indices]).*ComplexF64.(reshape(csm,:,nc)[indices,:]))/sqrt(length(indices))
        actual = reshape(op*image,ns*nd,nc)
        @test norm(actual-expected)/norm(expected) < .02
    finally
        close(op)
    end
end


@testset "sampled shared 2D off-centre and concomitant phase" begin
    rng = MersenneTwister(61513)
    dims = (9, 7, 1)
    centred = Grid(dims..., .001f0, .001f0, .001f0)
    grid = Grid(nX=9, nY=7, nZ=1, Δx=.001f0, Δy=.001f0, Δz=.001f0,
        x=centred.x, y=centred.y, z=fill(.002f0, prod(dims)))
    ns, nd, nc = 37, 3, 2
    k = zeros(Float32, 16, ns, nd)
    k[1, :, :] .= .03f0 .* randn(rng, Float32, ns, nd)
    k[2:4, :, :] .= 100f0 .* randn(rng, Float32, 3, ns, nd)
    times = reshape(collect(range(.01f0, .013f0; length=ns*nd)), ns, nd)
    fieldmap = reshape(collect(range(-30f0, 30f0; length=prod(dims))), dims)
    csm = randn(rng, ComplexF32, dims..., nc)
    mask = trues(dims); mask[1:17:end] .= false
    concomitant = 1000f0 .* randn(rng, Float32, 4, ns, nd)
    op = HighOrderLowRankOp(grid, k, times; fieldmap, csm, mask, concomitant,
        L_rank=6, rsvd_oversample=2, shared_rank_max=16, shared_basis_tol=1f-6,
        shared_readout_count=2, shared_fit_samples=48)
    try
        indices = findall(vec(mask))
        x, y, z = Float64.(grid.x[indices]), Float64.(grid.y[indices]), Float64.(grid.z[indices])
        fields = HighOrderMRI.basisfunc_spha(x, y, z, collect(1:16))
        extra = hcat(z.^2, x.^2+y.^2, x.*z, y.*z)
        phase = Float64.(reshape(k,16,:)') * fields' +
            Float64.(reshape(concomitant,4,:)') * extra' +
            Float64.(vec(times)) * Float64.(vec(fieldmap)[indices])'
        image = randn(rng, ComplexF32, prod(dims))
        expected = cis.(2π.*phase) *
            (ComplexF64.(image[indices]).*ComplexF64.(reshape(csm,:,nc)[indices,:])) / sqrt(length(indices))
        actual = reshape(op*image, ns*nd, nc)
        @test size(op.q, 1) == ns*nd
        @test 0 < size(op.q, 2) <= 16
        @test norm(actual-expected)/norm(expected) < 2e-3
    finally
        close(op)
    end
end
