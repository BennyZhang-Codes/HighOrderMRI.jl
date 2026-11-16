# Backend-specific plans live in optional package extensions. The normal
# operator keeps ownership and worker scheduling in the existing backend.
function highorder_batched_workspace(plan, q, basis, csm, weights2, mask_idx, grid_size, batch; options=(;))
    throw(ArgumentError("normal_method=:batched requires the supported NonuniformFFTs CUDA backend; load NonuniformFFTs and select its AbstractNFFTs backend"))
end

function highorder_batched_normal! end
function highorder_batched_adjoint! end
function highorder_batched_forward! end


release_highorder_batched_workspace!(workspace) = nothing

function validate_highorder_normal_options(method, options)
    isempty(options) && return nothing
    method === :batched || throw(ArgumentError("normal_options requires normal_method=:batched"))
    allowed = (:fft, :native_ptx, :vkfft_library, :coefficient_cache_bytes)
    all(key -> key in allowed, keys(options)) || throw(ArgumentError("unsupported normal_options key"))
    fft = get(options, :fft, :cufft)
    fft in (:cufft, :vkfft_pruned) || throw(ArgumentError("normal_options.fft must be :cufft or :vkfft_pruned"))
    if fft === :cufft
        keys(options) == (:fft,) || throw(ArgumentError("native resources require fft=:vkfft_pruned"))
    else
        for key in (:native_ptx, :vkfft_library)
            path = get(options, key, nothing)
            path isa AbstractString && isfile(path) || throw(ArgumentError("normal_options.$key must name an existing experimental resource"))
        end
        budget = get(options, :coefficient_cache_bytes, 0)
        budget isa Integer && budget >= 0 || throw(ArgumentError("coefficient_cache_bytes must be a nonnegative integer"))
    end
    return nothing
end
