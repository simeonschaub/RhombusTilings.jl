module CUDAExtension

using CUDA, KernelAbstractions, RhombusTilings.Slicing
using CUDA.CUBLAS

@kernel function det_kernel_pivot!(res, @Const(A), @Const(ipiv), @Const(info))
    k = @index(Global)
    @inbounds if !iszero(info[k])
        res[k] = 0.0f0
    else
        p = 1.0f0
        s = false
        for i in Cint(1):Cint(size(A, 1))
            p *= A[i, i, k]
            s ⊻= ipiv[i, k] != i
        end
        res[k] = s ? -p : p
    end
end

function Slicing.batched_det!(res::CuVector{Float32}, A::CuArray{Float32, 3}, ipiv::CuMatrix{Cint}, info::CuVector{Cint})
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

Slicing.requires_pivot(::CUDABackend) = true
Slicing.int_type(::OpenCLBackend) = Cint

end
