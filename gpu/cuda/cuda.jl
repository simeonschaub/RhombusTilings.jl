using CUDA, KernelAbstractions, LinearAlgebra, StaticArrays, GPUArrays, LogExpFunctions
using CUDA.CUBLAS

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

function batched_det!(res::CuVector{Float32}, A::CuArray{Float32, 3}, ipiv::CuMatrix{Cint}, info::CuVector{Cint})
    m, n = size(A, 1), size(A, 2)
    @assert m == n
    batch_count = size(A, 3)
    @assert length(res) ≥ batch_count
    @assert size(ipiv, 2) ≥ batch_count
    @assert length(info) ≥ batch_count
    # Batched LU factorization
    CUBLAS.getrf_strided_batched!(A, ipiv, info)
    kernel = det_kernel!(CUDABackend())
    kernel(res, A, ipiv, info; ndrange = batch_count)
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
        #x = x * nn ÷ rr
        rr += one(T)
        nn += one(T)
    end
    return x
end
@generated function binom(n::T, k::T, ::Val{N}) where {T <: Integer, N}
    table = SMatrix{N + 1, N + 1}(
        [binom(n, k) for k in 0:N, n in 0:N]
    )
    quote
        if 0 ≤ n ≤ N && 0 ≤ k ≤ n
            return $table[k + 1, n + 1]
        else
            return zero(T)
        end
    end
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
    if all(x_D .≤ x_A) && all(y_D .≤ y_A)
        inc = SizedVector{N}(I(0):I(N - 1))
        @inbounds @. A[:, :, i, j] = binom(x_A' - x_D + y_A' - y_D, x_A' - x_D + inc' - inc, Val(N))
    else
        @inbounds @. A[:, :, i, j] .= 0
    end
end

function count_paths(
        src::CuVector{SVector{N, NTuple{2, I}}},
        dst::CuVector{SVector{N, NTuple{2, I}}},
    ) where {N, I <: Integer}
    m, n = length(src), length(dst)
    A = CuArray{Float32}(undef, N, N, n, m)
    path_matrices!(CUDABackend())(A, src, dst; ndrange = (n, m))

    res = CuVector{Float32}(undef, m * n)
    ipiv = CuMatrix{Cint}(undef, N, m * n)
    info = CuVector{Cint}(undef, m * n)
    batched_det!(res, reshape(A, N, N, :), ipiv, info)
    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(ipiv)
    CUDA.unsafe_free!(info)

    return reshape(res, n, m)
end

using LogExpFunctions: _logsumexp_onepass_op
function logsumexp2!(out::AbstractArray, X::AbstractArray{<:Number}, xmax_r::AbstractArray{NTuple{2, FT}}) where {FT}
    fill!(xmax_r, (FT(-Inf), zero(FT)))
    GPUArrays.mapreducedim!(identity, _logsumexp_onepass_op, xmax_r, X; init = (FT(-Inf), zero(FT)))
    return @. out = first(xmax_r) + log1p(last(xmax_r))
end

function compute_npaths(
        DV::CuMatrix{NTuple{2, I}},
        C::CuVector{SVector{N, Int}},
        batch_size::Int = 100,
    ) where {N, I <: Integer}
    npaths = CuVector{Float32}(undef, N * length(C) + 2)
    @allowscalar npaths[end] = 0.0
    src, dst = CuMatrix{NTuple{2, I}}(undef, N, length(C)), CuMatrix{NTuple{2, I}}(undef, N, length(C))
    src′, dst′ = reinterpret(reshape, SVector{N, NTuple{2, I}}, src), reinterpret(reshape, SVector{N, NTuple{2, I}}, dst)

    A = CuArray{Float32}(undef, N, N, length(C) * batch_size)
    det = CuVector{Float32}(undef, length(C) * batch_size)
    ipiv = CuMatrix{Cint}(undef, N, length(C) * batch_size)
    info = CuVector{Cint}(undef, length(C) * batch_size)

    GPUArrays.vectorized_getindex!(src, @view(DV[:, end]), reinterpret(reshape, Int, C))
    dst[:, 1] .= Ref((I(N), I(N)))
    path_matrices!(CUDABackend())(reshape(A, N, N, 1, length(C) * batch_size), src′, dst′; ndrange = (1, length(C)))
    batched_det!(det, view(A, :, :, 1:length(C)), ipiv, info)
    npaths[(N - 1) * length(C) + 1 .+ (1:length(C))] .= log.(view(det, 1:length(C)))

    xmax_r = CuVector{NTuple{2, Float32}}(undef, batch_size)
    for k in (N - 1):-1:1
        dst, dst′, src, src′ = src, src′, dst, dst′
        GPUArrays.vectorized_getindex!(src, view(DV, :, k), reinterpret(reshape, Int, C))

        for b in 1:cld(length(C), batch_size)
            batch_start = (b - 1) * batch_size + 1
            batch_end = min(b * batch_size, length(C))
            batch_idx = batch_start:batch_end
            batch_size_actual = length(batch_idx)

            path_matrices!(CUDABackend())(reshape(A, N, N, length(C), batch_size), view(src′, batch_idx), dst′; ndrange = (length(C), batch_size_actual))
            batched_det!(det, view(A, :, :, 1:(length(C) * batch_size_actual)), ipiv, info)

            det′ = view(reshape(det, length(C), batch_size), :, 1:batch_size_actual)
            det′ .= log.(det′) .+ view(npaths, k * length(C) + 1 .+ (1:length(C)))
            logsumexp2!(reshape(view(npaths, (k - 1) * length(C) + 1 .+ batch_idx), 1, :), det′, reshape(view(xmax_r, 1:batch_size_actual), 1, :))
        end
    end

    dst, dst′, src, src′ = src, src′, dst, dst′
    @allowscalar src[:, 1] .= Ref((I(0), I(0)))
    path_matrices!(CUDABackend())(reshape(A, N, N, length(C), batch_size), src′, dst′; ndrange = (length(C), 1))
    batched_det!(det, view(A, :, :, 1:length(C)), ipiv, info)
    det′ = view(det, 1:length(C))
    det′ .= log.(det′) .+ view(npaths, 1 .+ (1:length(C)))
    @allowscalar npaths[1] = logsumexp(det′)

    return npaths
end

using Distributions: Categorical

function sample_path(
        npaths::CuVector{Float32},
        DV::CuMatrix{NTuple{2, I}},
        C::CuVector{SVector{N, Int}}
    ) where {N, I <: Integer}
    path = CuMatrix{NTuple{2, I}}(undef, N, N + 2)
    src, dst = CuMatrix{NTuple{2, I}}(undef, N, 1), CuMatrix{NTuple{2, I}}(undef, N, length(C))
    src′, dst′ = reinterpret(reshape, SVector{N, NTuple{2, I}}, src), reinterpret(reshape, SVector{N, NTuple{2, I}}, dst)

    A = CuArray{Float32}(undef, N, N, length(C))
    det_cpu = Vector{Float32}(undef, length(C))
    GC.@preserve det_cpu begin
        det = unsafe_wrap(CuVector{Float32}, pointer(det_cpu), size(det_cpu))
        info = CuVector{Cint}(undef, length(C))

        i = 1
        path[:, 1] .= Ref((I(0), I(0)))
        src[:, 1] .= Ref((I(0), I(0)))
        for j in 1:N
            GPUArrays.vectorized_getindex!(dst, view(DV, :, j), reinterpret(reshape, Int, C))

            path_matrices!(CUDABackend())(reshape(A, N, N, length(C), 1), src′, dst′; ndrange = (length(C), 1))
            batched_det!(det, A, info)

            @allowscalar det .*= exp.(view(npaths, (j - 1) * length(C) + 1 .+ (1:length(C)))) ./ exp(npaths[max(1, (j - 2) * length(C) + 1 + i)])
            synchronize(CUDABackend())
            i = rand(Categorical(det_cpu))
            copyto!(view(path, :, j + 1), view(dst, :, i))
            copyto!(view(src, :, 1), view(dst, :, i))
        end
    end
    path[:, end] .= Ref((I(N), I(N)))
    return path
end

using Combinatorics

N = 6
I = Int8
DV = CuMatrix{NTuple{2, I}}(
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
C = CuVector(SVector{N}.(with_replacement_combinations(1:(2N + 1), N)))

npaths = compute_npaths(DV, C)
#CUDA.@profile compute_npaths(DV, C)
#path = sample_path(npaths, DV, C)
