using AMDGPU, KernelAbstractions, LinearAlgebra, StaticArrays, GPUArrays, LogExpFunctions
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

function batched_det!(res::ROCVector{Float32}, A::AnyROCArray{Float32, 3}, info::ROCVector{Cint})
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

Base.@assume_effects :terminates_locally function binom(n::T, k::T) where {T <: Integer}
    (n ≥ 0 && 0 ≤ k ≤ n) || return zero(T)
    (k == 0 || k == n) && return one(T)
    k == 1 && return n
    if k > (n >> 1)
        k = (n - k)
    end
    x = nn = n - k + one(T)
    nn += one(T)
    rr = T(2)
    while rr <= k
        xt = div(widemul(x, nn), rr)
        x = xt % T
        rr += one(T)
        nn += one(T)
    end
    return x
end
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
    if all(x_D .≤ x_A) && all(y_D .≤ y_A)
        inc = SizedVector{N}(I(0):I(N - 1))
        @inbounds @. A[:, :, i, j] = binom(x_A' - x_D + y_A' - y_D, x_A' - x_D + inc' - inc)
    else
        @inbounds @. A[:, :, i, j] .= 0
    end
end

function count_paths(
        src::ROCVector{SVector{N, NTuple{2, I}}},
        dst::ROCVector{SVector{N, NTuple{2, I}}},
    ) where {N, I <: Integer}
    m, n = length(src), length(dst)
    A = ROCArray{Float32}(undef, N, N, n, m)
    path_matrices!(ROCBackend())(A, src, dst; ndrange = (n, m))

    res = ROCVector{Float32}(undef, m * n)
    info = ROCVector{Cint}(undef, m * n)
    batched_det!(res, reshape(A, N, N, :), info)
    AMDGPU.unsafe_free!(A)
    AMDGPU.unsafe_free!(info)

    return reshape(res, n, m)
end

using LogExpFunctions: _logsumexp_onepass_op
function logsumexp2!(out::AbstractArray, X::AbstractArray{<:Number}, xmax_r::AbstractArray{NTuple{2, FT}}) where {FT}
    fill!(xmax_r, (FT(-Inf), zero(FT)))
    GPUArrays.mapreducedim!(identity, _logsumexp_onepass_op, xmax_r, X; init = (FT(-Inf), zero(FT)))
    return @. out = first(xmax_r) + log1p(last(xmax_r))
end

function compute_npaths(
        DV::ROCMatrix{NTuple{2, I}},
        C::ROCVector{SVector{N, Int}},
        batch_size::Int = 100,
    ) where {N, I <: Integer}
    npaths = ROCVector{Float32}(undef, N * length(C) + 2)
    @allowscalar npaths[end] = 0.0
    src, dst = ROCMatrix{NTuple{2, I}}(undef, N, length(C)), ROCMatrix{NTuple{2, I}}(undef, N, length(C))
    src′, dst′ = reinterpret(reshape, SVector{N, NTuple{2, I}}, src), reinterpret(reshape, SVector{N, NTuple{2, I}}, dst)

    A = ROCArray{Float32}(undef, N, N, length(C) * batch_size)
    det = ROCVector{Float32}(undef, length(C) * batch_size)
    info = ROCVector{Cint}(undef, length(C) * batch_size)

    GPUArrays.vectorized_getindex!(src, @view(DV[:, end]), reinterpret(reshape, Int, C))
    dst[:, 1] .= Ref((I(N), I(N)))
    path_matrices!(ROCBackend())(reshape(A, N, N, 1, length(C) * batch_size), src′, dst′; ndrange = (1, length(C)))
    batched_det!(det, view(A, :, :, 1:length(C)), info)
    npaths[(N - 1) * length(C) + 1 .+ (1:length(C))] .= log.(view(det, 1:length(C)))

    xmax_r = ROCVector{NTuple{2, Float32}}(undef, batch_size)
    for k in (N - 1):-1:1
        dst, dst′, src, src′ = src, src′, dst, dst′
        GPUArrays.vectorized_getindex!(src, view(DV, :, k), reinterpret(reshape, Int, C))

        for b in 1:cld(length(C), batch_size)
            batch_start = (b - 1) * batch_size + 1
            batch_end = min(b * batch_size, length(C))
            batch_idx = batch_start:batch_end
            batch_size_actual = length(batch_idx)

            path_matrices!(ROCBackend())(reshape(A, N, N, length(C), batch_size), view(src′, batch_idx), dst′; ndrange = (length(C), batch_size_actual))
            batched_det!(det, view(A, :, :, 1:(length(C) * batch_size_actual)), info)

            det′ = view(reshape(det, length(C), batch_size), :, 1:batch_size_actual)
            det′ .= log.(det′) .+ view(npaths, k * length(C) + 1 .+ (1:length(C)))
            logsumexp2!(reshape(view(npaths, (k - 1) * length(C) + 1 .+ batch_idx), 1, :), det′, reshape(view(xmax_r, 1:batch_size_actual), 1, :))
        end
    end

    dst, dst′, src, src′ = src, src′, dst, dst′
    @allowscalar src[:, 1] .= Ref((I(0), I(0)))
    path_matrices!(ROCBackend())(reshape(A, N, N, length(C), batch_size), src′, dst′; ndrange = (length(C), 1))
    batched_det!(det, view(A, :, :, 1:length(C)), info)
    det′ = view(det, 1:length(C))
    det′ .= log.(det′) .+ view(npaths, 1 .+ (1:length(C)))
    @allowscalar npaths[1] = logsumexp(det′)

    return npaths
end

using Distributions: Categorical

function sample_path(
        npaths::ROCVector{Float32},
        DV::ROCMatrix{NTuple{2, I}},
        C::ROCVector{SVector{N, Int}}
    ) where {N, I <: Integer}
    path = ROCMatrix{NTuple{2, I}}(undef, N, N + 2)
    src, dst = ROCMatrix{NTuple{2, I}}(undef, N, 1), ROCMatrix{NTuple{2, I}}(undef, N, length(C))
    src′, dst′ = reinterpret(reshape, SVector{N, NTuple{2, I}}, src), reinterpret(reshape, SVector{N, NTuple{2, I}}, dst)

    A = ROCArray{Float32}(undef, N, N, length(C))
    det_cpu = Vector{Float32}(undef, length(C))
    GC.@preserve det_cpu begin
        det = unsafe_wrap(ROCVector{Float32}, pointer(det_cpu), size(det_cpu))
        info = ROCVector{Cint}(undef, length(C))

        i = 1
        path[:, 1] .= Ref((I(0), I(0)))
        src[:, 1] .= Ref((I(0), I(0)))
        for j in 1:N
            GPUArrays.vectorized_getindex!(dst, view(DV, :, j), reinterpret(reshape, Int, C))

            path_matrices!(ROCBackend())(reshape(A, N, N, length(C), 1), src′, dst′; ndrange = (length(C), 1))
            batched_det!(det, A, info)

            @allowscalar det .*= exp.(view(npaths, (j - 1) * length(C) + 1 .+ (1:length(C)))) ./ exp(npaths[max(1, (j - 2) * length(C) + 1 + i)])
            synchronize(ROCBackend())
            i = rand(Categorical(det_cpu))
            copyto!(view(path, :, j + 1), view(dst, :, i))
            copyto!(view(src, :, 1), view(dst, :, i))
        end
    end
    path[:, end] .= Ref((I(N), I(N)))
    return path
end

#A = Array{Float32}(undef, 6, 6, 1, length(dst))
#path_matrices!(CPU())(A, src, dst; ndrange = (length(src), length(dst)))

using Combinatorics

N = 6
I = Int8
DV = ROCMatrix{NTuple{2, I}}(
    [
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
)
C = ROCVector(SVector{N}.(with_replacement_combinations(1:(2N + 1), N)))

npaths = compute_npaths(DV, C)
#path = sample_path(npaths, DV, C)
#
#src = ROCVector([SVector(ntuple(_ -> (I(0), I(0)), N))])
#dst = reinterpret(reshape, SVector{N, NTuple{2, I}}, view(DV, :, 1)[reinterpret(reshape, Int, C)])
#count_paths(src, dst)
#
#let
#    src = reinterpret(reshape, SVector{N, NTuple{2, I}}, view(DV, :, 1)[reinterpret(reshape, Int, C)])
#    dst = reinterpret(reshape, SVector{N, NTuple{2, I}}, view(DV, :, 2)[reinterpret(reshape, Int, C)])[1:100]
#    Array(count_paths(src, dst)) ≈ det.(npaths2.(reshape(Array(src), 1, :), Array(dst)))
#end
#
#A = rand(0.0f0:6.0f0, 6, 6, 1000);
#A′ = ROCArray(A);
#batched_det!(ROCVector{Float32}(undef, 1000), A′, ROCVector{Cint}(undef, 1000))
#det.(eachslice(A; dims = 3))
#
