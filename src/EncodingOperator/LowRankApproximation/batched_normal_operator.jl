# Backend-specific plans live in optional package extensions. The normal
# operator keeps ownership and worker scheduling in the existing backend.
function highorder_batched_workspace(plan, q, basis, csm, weights2, mask_idx, grid_size, batch)
    throw(ArgumentError("normal_method=:batched requires the supported NonuniformFFTs CUDA backend; load NonuniformFFTs and select its AbstractNFFTs backend"))
end

function highorder_batched_normal! end
function highorder_batched_adjoint! end
function highorder_batched_forward! end
