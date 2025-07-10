using AMDGPU, KernelAbstractions, LinearAlgebra
using AMDGPU: rocBLAS, rocSOLVER

@kernel function det_kernel!(res, @Const(A), @Const(ipiv), @Const(info))
    k = @index(Global)
    @inbounds if !iszero(info[k])
        res[k] = 0f0
    else
        p = 1f0
        s = false
        for i in Cint(1):Cint(size(A, 1))
            p *= A[i, i, k]
            s ⊻= ipiv[i, k] != i
        end
        res[k] = s ? -p : p
    end
end

function batched_det!(res::ROCVector{Float32}, A::AnyROCArray{Float32, 3}, ipiv::ROCMatrix{Cint}, info::ROCVector{Cint})
    m, n = size(A)
    @assert m == n
    lda = max(1, stride(A, 2))
    strideP = stride(A, 2)
    strideA = stride(A, 3)
    batch_count = size(A, 3)
    @assert length(ipiv) ≥ batch_count * n
    @assert size(ipiv, 1) == n
    @assert length(info) ≥ batch_count
    rocSOLVER.rocsolver_sgetrf_strided_batched(rocBLAS.handle(), m, n, A, lda, strideA, ipiv, strideP, info, batch_count)

    kernel = det_kernel!(ROCBackend())
    kernel(res, A, ipiv, info; ndrange = batch_count)
    return res
end

A = rand(0f0:6f0, 6, 6, 1000);
A′ = ROCArray(A);
batched_det!(ROCVector{Float32}(undef, 1000), A′, ROCMatrix{Cint}(undef, 6, 1000), ROCVector{Cint}(undef, 1000))
det.(eachslice(A; dims = 3))
