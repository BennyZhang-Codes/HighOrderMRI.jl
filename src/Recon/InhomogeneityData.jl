import MRIReco.MRIOperators: InhomogeneityData

function InhomogeneityData(
    A_k::matT, 
    C_k::matT, 
    times::vecT, 
    Cmap::AbstractArray{Complex{T}, 3}, 
    t_hat::T, 
    z_hat::Complex{T}, 
    method::String
) where {T, matT <: AbstractArray{Complex{T}, 2}, vecT <: AbstractArray{T, 1}}
    cmap_2d = reshape(Cmap, :, 1)
    return InhomogeneityData(A_k, C_k, times, cmap_2d, t_hat, z_hat, method)
end