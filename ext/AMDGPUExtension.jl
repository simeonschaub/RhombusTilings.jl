module AMDGPUExtension

using AMDGPU, KernelAbstractions, RhombusTilings
using AMDGPU: rocBLAS, rocSOLVER

function RhombusTilings.batched_det!(res::ROCVector{Float32}, A::AnyROCArray{Float32, 3}, ipiv::Nothing, info::ROCVector{Cint})
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

RhombusTilings.requires_pivot(::ROCBackend) = false

end
