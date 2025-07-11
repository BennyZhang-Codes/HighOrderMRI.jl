function recon_HOOp(HOOp::HOOp{Complex{T}}, Data::AbstractArray{Complex{T},2}, weight::AbstractVector{Complex{T}}, recParams::Dict) where T<:AbstractFloat
    recoParams = merge(defaultRecoParams(), recParams)

    nSample, nCha = size(Data)
    Data = vec(Data) .* repeat(weight, nCha)
    W = WeightingOp(Complex{T}; weights=weight, rep=nCha)
    E = ∘(W, HOOp)
    EᴴE = normalOperator(E)
    solver = createLinearSolver(recParams[:solver], E; AHA=EᴴE, reg=recParams[:reg], recoParams...)
    x = solve!(solver, Data)
    x = reshape(x, recParams[:reconSize])
    return x
end