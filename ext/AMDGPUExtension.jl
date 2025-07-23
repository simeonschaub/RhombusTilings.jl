module AMDGPUExtension

using AMDGPU, KernelAbstractions, RhombusTilings.Slicing
using AMDGPU: rocBLAS, rocSOLVER

@kernel function det_kernel!(res, @Const(A), @Const(info))
    k = @index(Global)
    @inbounds if !iszero(info[k])
        res[k] = 0.0f0
    else
        p = 1.0f0
        for i in Cint(1):Cint(size(A, 1))
            p *= A[i, i, k]
        end
        res[k] = p
    end
end

function Slicing.batched_det!(res::ROCVector{Float32}, A::AnyROCArray{Float32, 3}, ipiv::Nothing, info::ROCVector{Cint})
    m, n = size(A)
    @assert m == n
    lda = max(1, stride(A, 2))
    strideA = stride(A, 3)
    batch_count = size(A, 3)
    @assert length(res) ≥ batch_count
    @assert length(info) ≥ batch_count
    rocSOLVER.rocsolver_sgetf2_npvt_strided_batched(rocBLAS.handle(), m, n, A, lda, strideA, info, batch_count)

    kernel = det_kernel!(ROCBackend())
    kernel(res, A, info; ndrange = batch_count)
    return res
end

Slicing.requires_pivot(::ROCBackend) = false

end
