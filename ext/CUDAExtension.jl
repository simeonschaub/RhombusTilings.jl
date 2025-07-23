module CUDAExtension

using CUDA, KernelAbstractions, RhombusTilings
using CUDA.CUBLAS

function RhombusTilings.batched_det!(res::CuVector{Float32}, A::CuArray{Float32, 3}, ipiv::CuMatrix{Cint}, info::CuVector{Cint})
    m, n = size(A, 1), size(A, 2)
    @assert m == n
    batch_count = size(A, 3)
    @assert length(res) ≥ batch_count
    @assert size(ipiv, 2) ≥ batch_count
    @assert length(info) ≥ batch_count
    CUBLAS.getrf_strided_batched!(A, ipiv, info)

    kernel = det_kernel_pivot!(CUDABackend())
    kernel(res, A, ipiv, info; ndrange = batch_count)
    return res
end

RhombusTilings.requires_pivot(::CUDABackend) = true

end
