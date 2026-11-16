using Adapt

function update_nfft_nodes!(
    plan::AbstractNFFTPlan,
    nodes::Matrix{T},
) where T<:AbstractFloat
    if hasproperty(plan, :p)
        # NonuniformFFTs.NFFTPlan stores its point arrays on the execution
        # backend. Its nodes! implementation requires new points to use the
        # same container type as those already held by the plan.
        current_points = plan.p.points
        nodes_backend = if first(current_points) isa CuArray
            CuArray(nodes)
        else
            nodes
        end

        AbstractNFFTs.nodes!(plan, nodes_backend)
    elseif hasproperty(plan, :tmpVec) && plan.tmpVec isa CuArray
        # The GPU implementation always uses FULL precomputation. Only the
        # sparse interpolation matrix depends on the trajectory; FFT plans and
        # their large work buffers can be retained across dynamics.
        _, _, _, _, interpolation = NFFT.precomputation(
            nodes,
            plan.N,
            plan.Ñ,
            plan.params,
        )

        interpolation_gpu = Complex{T}.(Adapt.adapt(CuArray, interpolation))

        plan.J = size(nodes, 2)
        plan.k = nodes
        plan.B = interpolation_gpu
    elseif applicable(AbstractNFFTs.nodes!, plan, nodes)
        AbstractNFFTs.nodes!(plan, nodes)
    else
        error(
            "The active NFFT backend does not support streaming node updates: " *
            string(typeof(plan))
        )
    end

    return plan
end
