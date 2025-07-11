using AMDGPU, KernelAbstractions, LinearAlgebra, StaticArrays, GPUArrays
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
    @assert length(res) ≥ batch_count
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
        @Const(src::AbstractVector{SVector{N, NTuple{2, I}}}),
        @Const(dst::AbstractVector{SVector{N, NTuple{2, I}}}),
    ) where {N, I <: Integer}
    i, j = @index(Global, NTuple)
    s, d = @inbounds src[j], dst[i]
    x_D, x_A = first.(s), first.(d)
    y_D, y_A = last.(s), last.(d)
    inc = SizedVector{N}(I(0):I(N - 1))
    @inbounds @. A[:, :, i, j] = binom(x_A' - x_D + y_A' - y_D, x_A' - x_D + inc' - inc)
end

function count_paths(
        src::ROCVector{SVector{N, NTuple{2, I}}},
        dst::ROCVector{SVector{N, NTuple{2, I}}},
    ) where {N, I <: Integer}
    m, n = length(src), length(dst)
    A = ROCArray{Float32}(undef, N, N, m, n)
    path_matrices!(ROCBackend())(A, src, dst; ndrange = (n, m))

    res = ROCVector{Float32}(undef, m * n)
    ipiv = ROCMatrix{Cint}(undef, N, m * n)
    info = ROCVector{Cint}(undef, m * n)
    batched_det!(res, reshape(A, N, N, :), ipiv, info)
    AMDGPU.unsafe_free!(A)
    AMDGPU.unsafe_free!(ipiv)
    AMDGPU.unsafe_free!(info)

    return reshape(res, m, n)
end

function compute_npaths(
        DV::ROCMatrix{NTuple{2, I}},
        C::ROCVector{SVector{N, Int}},
        batch_size::Int = 100,
    ) where {N, I <: Integer}
    npaths = ROCVector{Float64}(undef, N * length(C) + 2)
    @allowscalar npaths[end] = 0.0
    src, dst = ROCMatrix{NTuple{2, I}}(undef, N, length(C)), ROCMatrix{NTuple{2, I}}(undef, N, length(C))
    src′, dst′ = reinterpret(reshape, SVector{N, NTuple{2, I}}, src), reinterpret(reshape, SVector{N, NTuple{2, I}}, dst)

    A = ROCArray{Float32}(undef, N, N, length(C) * batch_size)
    det = ROCVector{Float32}(undef, length(C) * batch_size)
    ipiv = ROCMatrix{Cint}(undef, N, length(C) * batch_size)
    info = ROCVector{Cint}(undef, length(C) * batch_size)

    GPUArrays.vectorized_getindex!(src, @view(DV[:, end]), reinterpret(reshape, Int, C))
    dst[:, 1] .= Ref((I(N), I(N)))
    path_matrices!(ROCBackend())(reshape(A, N, N, 1, length(C) * batch_size), src′, dst′; ndrange = (1, length(C)))
    batched_det!(det, view(A, :, :, 1:length(C)), ipiv, info)
    npaths[(N - 1) * length(C) + 1 .+ (1:length(C))] .= log.(view(det, 1:length(C)))

    for k in (N - 1):-1:1
        dst, dst′ = src, src′
        GPUArrays.vectorized_getindex!(src, view(DV, :, k), reinterpret(reshape, Int, C))

        for b in 1:cld(length(C), batch_size)
            batch_start = (b - 1) * batch_size + 1
            batch_end = min(b * batch_size, length(C))
            batch_idx = batch_start:batch_end
            batch_size_actual = length(batch_idx)

            path_matrices!(ROCBackend())(reshape(A, N, N, length(C), batch_size), src′, dst′; ndrange = (length(C), batch_size_actual))
            batched_det!(det, view(A, :, :, 1:(length(C) * batch_size_actual)), ipiv, info)

            det′ = view(reshape(det, length(C), batch_size), :, 1:batch_size_actual)
            det′ .= log.(det′) .+ view(npaths, k * length(C) + 1 .+ batch_idx)'
            logsumexp!(reshape(view(npaths, (k - 1) * length(C) + 1 .+ batch_idx), 1, :), det′)
        end
    end

    dst, dst′ = src, src′
    @allowscalar src[:, 1] .= Ref((I(0), I(0)))
    path_matrices!(ROCBackend())(reshape(A, N, N, length(C), batch_size), src′, dst′; ndrange = (length(C), 1))
    batched_det!(det, view(A, :, :, 1:length(C)), ipiv, info)
    det′ = view(det, 1:length(C))
    det′ .= log.(det′) .+ view(npaths, 1 .+ (1:length(C)))
    @allowscalar npaths[1] = logsumexp(det′)

    return npaths
end

#A = Array{Float32}(undef, 6, 6, 1, length(dst))
#path_matrices!(CPU())(A, src, dst; ndrange = (length(src), length(dst)))

using Combinatorics

N = 6
I = Int8
DV = ROCMatrix{NTuple{2, I}}([
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
])
C = ROCVector(SVector{N}.(with_replacement_combinations(1:(2N + 1), N)))

npaths = compute_npaths(DV, C)

src = ROCVector([SVector(ntuple(_ -> (I(0), I(0)), N))])
dst = reinterpret(reshape, SVector{N, NTuple{2, I}}, view(DV, :, 1)[reinterpret(reshape, Int, C)])
count_paths(src, dst)

npaths = let
    src = reinterpret(reshape, SVector{N, NTuple{2, I}}, view(DV, :, 1)[reinterpret(reshape, Int, C)])
    dst = reinterpret(reshape, SVector{N, NTuple{2, I}}, view(DV, :, 2)[reinterpret(reshape, Int, C)])[1:1000]
    count_paths(src, dst)
end

A = rand(0.0f0:6.0f0, 6, 6, 1000);
A′ = ROCArray(A);
batched_det!(ROCVector{Float32}(undef, 1000), A′, ROCMatrix{Cint}(undef, 6, 1000), ROCVector{Cint}(undef, 1000))
det.(eachslice(A; dims = 3))
