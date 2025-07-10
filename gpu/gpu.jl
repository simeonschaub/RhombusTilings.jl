using AMDGPU, KernelAbstractions, LinearAlgebra, StaticArrays
using AMDGPU: rocBLAS, rocSOLVER

@kernel function det_kernel!(res, @Const(A), @Const(ipiv), @Const(info))
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

binom(n, k) = n ≥ 0 && 0 ≤ k ≤ n ? binomial(n, k) : zero(n)
@kernel function path_matrices!(
        A::AbstractArray{Float32, 4},
        @Const(src::AbstractVector{SVector{N, NTuple{2, Int}}}),
        @Const(dst::AbstractVector{SVector{N, NTuple{2, Int}}}),
    ) where {N}
    i, j = @index(Global, NTuple)
    s, d = @inbounds src[i], dst[j]
    x_D, x_A = first.(s), first.(d)
    y_D, y_A = last.(s), last.(d)
    inc = StaticArrays.SUnitRange(0, N - 1)
    @inbounds @. A[:, :, i, j] = binom(x_A' - x_D + y_A' - y_D, x_A' - x_D + inc' - inc)
end

function count_paths(
        src::AbstractVector{SVector{N, NTuple{2, Int}}},
        dst::AbstractVector{SVector{N, NTuple{2, Int}}},
    ) where {N}
    m, n = length(src), length(dst)
    A = ROCArray{Float32}(undef, N, N, m, n)
    GC.@preserve src dst begin
        src_gpu = unsafe_wrap(ROCVector{SVector{N, NTuple{2, Int}}}, pointer(src), size(src))
        dst_gpu = unsafe_wrap(ROCVector{SVector{N, NTuple{2, Int}}}, pointer(dst), size(dst))
        path_matrices!(ROCBackend())(A, src_gpu, dst_gpu; ndrange = (m, n))
    end
    res = ROCVector{Float32}(undef, m * n)
    ipiv = ROCMatrix{Cint}(undef, N, m * n)
    info = ROCVector{Cint}(undef, m * n)
    batched_det!(res, reshape(A, N, N, :), ipiv, info)
    AMDGPU.unsafe_free!(A)
    AMDGPU.unsafe_free!(ipiv)
    AMDGPU.unsafe_free!(info)
    return reshape(res, m, n)
end

using Combinatorics

N = 6
DV = [
    (0, 6)  (0, 6)  (0, 6)  (0, 6)  (0, 6)  (0, 6)
    (0, 5)  (0, 5)  (1, 6)  (1, 6)  (1, 6)  (1, 6)
    (0, 4)  (1, 5)  (1, 5)  (2, 6)  (2, 6)  (2, 6)
    (1, 4)  (2, 5)  (2, 5)  (2, 5)  (2, 5)  (2, 5)
    (2, 4)  (2, 4)  (2, 4)  (3, 5)  (3, 5)  (3, 5)
    (2, 3)  (2, 3)  (3, 4)  (3, 4)  (3, 4)  (3, 4)
    (2, 2)  (3, 3)  (3, 3)  (3, 3)  (3, 3)  (4, 4)
    (2, 1)  (3, 2)  (3, 2)  (3, 2)  (4, 3)  (4, 3)
    (3, 1)  (3, 1)  (3, 1)  (4, 2)  (4, 2)  (5, 3)
    (4, 1)  (4, 1)  (4, 1)  (5, 2)  (5, 2)  (5, 2)
    (5, 1)  (5, 1)  (5, 1)  (6, 2)  (6, 2)  (6, 2)
    (5, 0)  (5, 0)  (5, 0)  (6, 1)  (6, 1)  (6, 1)
    (6, 0)  (6, 0)  (6, 0)  (6, 0)  (6, 0)  (6, 0)
]
C = SVector{N}.(with_replacement_combinations(1:(2N + 1), N))
src = [SVector(ntuple(_ -> (0, 0), N))]
dst = reinterpret(reshape, SVector{N, NTuple{2, Int}}, view(DV, :, 1)[reinterpret(reshape, Int, C)])
count_paths(src, dst)

A = rand(0.0f0:6.0f0, 6, 6, 1000);
A′ = ROCArray(A);
batched_det!(ROCVector{Float32}(undef, 1000), A′, ROCMatrix{Cint}(undef, 6, 1000), ROCVector{Cint}(undef, 1000))
det.(eachslice(A; dims = 3))
