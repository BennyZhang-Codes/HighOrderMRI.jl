function prep_kspha(
    kspha         :: AbstractArray{T, 2} , 
    k_nominal     :: AbstractArray{T, 2} , 
    nTerm         :: Int64               ;
    recon_terms   :: String = nothing    ,
    verbose       :: Bool   = false      , 
    ) where T<:AbstractFloat
    if isnothing(recon_terms)
        recon_terms = nTerm == 9 ? "111" : "1111"
    end
    if nTerm == 9
        @assert length(recon_terms) == 3 "recon_terms must be 3 digits for up to 2nd order terms"
        t0 = Bool(parse(Int64, recon_terms[1]))
        t1 = Bool(parse(Int64, recon_terms[2]))
        t2 = Bool(parse(Int64, recon_terms[3]))
        t3 = false
    elseif nTerm == 16
        @assert length(recon_terms) == 4 "recon_terms must be 4 digits for up to 3rd order terms"
        t0 = Bool(parse(Int64, recon_terms[1]))
        t1 = Bool(parse(Int64, recon_terms[2]))
        t2 = Bool(parse(Int64, recon_terms[3]))
        t3 = Bool(parse(Int64, recon_terms[4]))
    else
        @error "nTerm must be 9 or 16"
    end

    if t0 == false
        kspha[1, :] = kspha[1, :] .* 0
    end
    if t1 == false
        kspha[2:4, :] = k_nominal[:, :]
    end
    if t2 == false
        kspha[5:9, :] = kspha[5:9, :] .* 0
    end
    if t3 == false && nTerm == 16
        kspha[10:16, :] = kspha[10:16, :] .* 0
    end
    if verbose
        @info "kspha prepared for flag: $(recon_terms)" zeroth=t0 first=t1 second=t2 third=t3
    end
    return kspha
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


