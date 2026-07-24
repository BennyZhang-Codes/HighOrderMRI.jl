function prep_kspha(
    kspha         :: AbstractArray{T, 2} , 
    k_nominal     :: AbstractArray{T, 2} , 
    nTerm         :: Int64               ;
    recon_terms   :: Union{Nothing,AbstractString} = nothing,
    verbose       :: Bool   = false      , 
    ) where T<:AbstractFloat
    if isnothing(recon_terms)
        recon_terms = nTerm == 9 ? "111" : "1111"
    end

    expected_length = nTerm == 9 ? 3 : nTerm == 16 ? 4 : 0
    expected_length > 0 || throw(ArgumentError(
        "nTerm must be 9 or 16, got $nTerm",
    ))
    length(recon_terms) == expected_length || throw(ArgumentError(
        "recon_terms must contain $expected_length binary digits for nTerm=$nTerm",
    ))
    all(c -> c == '0' || c == '1', recon_terms) || throw(ArgumentError(
        "recon_terms must contain only '0' and '1', got $(repr(recon_terms))",
    ))

    if nTerm == 9
        t0 = Bool(parse(Int64, recon_terms[1]))
        t1 = Bool(parse(Int64, recon_terms[2]))
        t2 = Bool(parse(Int64, recon_terms[3]))
        t3 = false
    elseif nTerm == 16
        t0 = Bool(parse(Int64, recon_terms[1]))
        t1 = Bool(parse(Int64, recon_terms[2]))
        t2 = Bool(parse(Int64, recon_terms[3]))
        t3 = Bool(parse(Int64, recon_terms[4]))
    end

    # Preparing one reconstruction condition must not alter the coefficients
    # subsequently reused by another condition (for example nominal, explicit,
    # and low-rank comparisons in a benchmark).
    kspha_prepared = copy(kspha)

    if t0 == false
        kspha_prepared[1, :] .= zero(T)
    end
    if t1 == false
        kspha_prepared[2:4, :] .= k_nominal[:, :]
    end
    if t2 == false
        kspha_prepared[5:9, :] .= zero(T)
    end
    if t3 == false && nTerm == 16
        kspha_prepared[10:16, :] .= zero(T)
    end
    if verbose
        @info "kspha prepared for flag: $(recon_terms)" zeroth=t0 first=t1 second=t2 third=t3
    end
    return kspha_prepared
end


function prep_kspha(
    kspha         :: AbstractArray{T, 3} , 
    k_nominal     :: AbstractArray{T, 3} , 
    nTerm         :: Int64               ;
    kwargs...     
    ) where T<:AbstractFloat
    nTerm, nSam, nDyn = size(kspha)
    kspha     = reshape(kspha, (nTerm, nSam*nDyn))
    k_nominal = reshape(k_nominal, (3, nSam*nDyn))

    kspha = prep_kspha(kspha, k_nominal, nTerm; kwargs...)

    kspha = reshape(kspha, (nTerm, nSam, nDyn))
    return kspha
end

