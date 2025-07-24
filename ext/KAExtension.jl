module KAExtension

using KernelAbstractions, GPUArrays, StaticArrays, RhombusTilings.Slicing
using Atomix: @atomic

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
@generated function binom(n::T, k::T, ::Val{N}) where {T <: Integer, N}
    table = SMatrix{N + 1, N + 1}(
        Float32[binom(n, k) for k in 0:N, n in 0:N]
    )
    quote
        if 0 ≤ n ≤ N && 0 ≤ k ≤ n
            return $table[k + 1, n + 1]
        else
            return 0f0
        end
    end
end

@kernel function path_matrices!(
        A::AbstractArray{Float32, 3},
        count::AbstractArray{Int, 0},
        indices::AbstractVector{Int},
        @Const(src::AbstractVector{SVector{N, NTuple{2, I}}}),
        @Const(dst::AbstractVector{SVector{N, NTuple{2, I}}}),
    ) where {N, I <: Integer}
    idx = @index(Global)
    i, j = @index(Global, NTuple)
    s, d = @inbounds src[j], dst[i]
    x_D, x_A = first.(s), first.(d)
    y_D, y_A = last.(s), last.(d)
    if all(x_D .≤ x_A) && all(y_D .≤ y_A)
        k = @atomic count[] += 1
        inc = SizedVector{N}(I(0):I(N - 1))
        @inbounds @. A[:, :, k] = binom(x_A' - x_D + y_A' - y_D, x_A' - x_D + inc' - inc, Val(N))
        @inbounds indices[k] = idx
    end
end

using .Slicing: batched_det!, requires_pivot, allocate_lu, int_type
Slicing.allocate_lu(backend, args...) = allocate(backend, args...)

function Slicing.count_paths(
        src::AbstractGPUVector{SVector{N, NTuple{2, I}}},
        dst::AbstractGPUVector{SVector{N, NTuple{2, I}}},
    ) where {N, I <: Integer}
    backend = get_backend(src)
    m, n = length(src), length(dst)
    A = allocate_lu(backend, Float32, N, N, m * n)
    _count = allocate(backend, Int)
    fill!(_count, 0)
    idx = allocate(backend, Int, m * n)
    path_matrices!(backend)(A, _count, idx, src, dst; ndrange = (n, m))
    count = @allowscalar _count[]

    _res = allocate_lu(backend, Float32, count)
    ipiv = requires_pivot(backend) ? allocate_lu(backend, int_type(backend), N, count) : nothing
    info = allocate_lu(backend, int_type(backend), count)
    batched_det!(_res, view(A, :, :, 1:count), ipiv, info)
    unsafe_free!(A)
    ipiv !== nothing && unsafe_free!(ipiv)
    unsafe_free!(info)

    res = allocate(backend, Float32, n, m)
    fill!(res, 0f0)
    res[view(idx, 1:count)] .= _res
    unsafe_free!(_res)

    return res
end

using LogExpFunctions: _logsumexp_onepass_op
function logsumexp2!(out::AbstractArray, X::AbstractArray{<:Number}, xmax_r::AbstractArray{NTuple{2, FT}}) where {FT}
    fill!(xmax_r, (FT(-Inf), zero(FT)))
    GPUArrays.mapreducedim!(identity, _logsumexp_onepass_op, xmax_r, X; init = (FT(-Inf), zero(FT)))
    return @. out = first(xmax_r) + log1p(last(xmax_r))
end

function Slicing.compute_npaths(
        DV::AbstractGPUMatrix{NTuple{2, I}},
        C::AbstractGPUVector{SVector{N, Int}},
        batch_size::Int = 100,
    ) where {N, I <: Integer}
    backend = get_backend(DV)
    npaths = allocate(backend, Float32, N * length(C) + 2)
    fill!(npaths, -Inf32)
    @allowscalar npaths[end] = 0f0
    src = allocate(backend, NTuple{2, I}, N, length(C))
    dst = allocate(backend, NTuple{2, I}, N, length(C))
    src′ = reinterpret(reshape, SVector{N, NTuple{2, I}}, src)
    dst′ = reinterpret(reshape, SVector{N, NTuple{2, I}}, dst)

    A = allocate_lu(backend, Float32, N, N, length(C) * batch_size)
    _count = allocate(backend, Int)
    indices = allocate(backend, Int, length(C) * batch_size)
    det = allocate_lu(backend, Float32, length(C) * batch_size)
    ipiv = requires_pivot(backend) ? allocate_lu(backend, int_type(backend), N, length(C) * batch_size) : nothing
    info = allocate_lu(backend, int_type(backend), length(C) * batch_size)
    tmp = allocate_lu(backend, Float32, length(C), batch_size)

    GPUArrays.vectorized_getindex!(src, @view(DV[:, end]), reinterpret(reshape, Int, C))
    dst[:, 1] .= Ref((I(N), I(N)))
    fill!(_count, 0)
    path_matrices!(backend)(A, _count, indices, src′, dst′; ndrange = (1, length(C)))
    count = @allowscalar _count[]
    @show view(indices, 1:count) |> sort
    batched_det!(det, view(A, :, :, 1:count), ipiv, info)
    npaths[(N - 1) * length(C) + 1 .+ view(indices, 1:count)] .= log.(view(det, 1:count))

    xmax_r = allocate(backend, NTuple{2, Float32}, batch_size)
    for k in (N - 1):-1:1
        dst, dst′, src, src′ = src, src′, dst, dst′
        GPUArrays.vectorized_getindex!(src, view(DV, :, k), reinterpret(reshape, Int, C))

        for b in 1:cld(length(C), batch_size)
            batch_start = (b - 1) * batch_size + 1
            batch_end = min(b * batch_size, length(C))
            batch_idx = batch_start:batch_end
            batch_size_actual = length(batch_idx)

            fill!(_count, 0)
            path_matrices!(backend)(A, _count, indices, view(src′, batch_idx), dst′; ndrange = (length(C), batch_size_actual))
            count = @allowscalar _count[]
            batched_det!(det, view(A, :, :, 1:count), ipiv, info)

            tmp′ = view(tmp, :, 1:batch_size_actual)
            tmp′ .= view(npaths, k * length(C) + 1 .+ (1:length(C)))
            tmp′[view(indices, 1:count)] .+= log.(view(det, 1:count))
            logsumexp2!(reshape(view(npaths, (k - 1) * length(C) + 1 .+ batch_idx), 1, :), tmp′, reshape(view(xmax_r, 1:batch_size_actual), 1, :))
        end
    end

    dst, dst′, src, src′ = src, src′, dst, dst′
    src[:, 1] .= Ref((I(0), I(0)))
    fill!(_count, 0)
    path_matrices!(backend)(A, _count, indices, src′, dst′; ndrange = (length(C), 1))
    count = @allowscalar _count[]
    batched_det!(det, view(A, :, :, 1:count), ipiv, info)
    tmp′ = view(tmp, :, 1)
    tmp′ .= view(npaths, 1 .+ (1:length(C)))
    tmp′[view(indices, 1:count)] .+= log.(view(det, 1:count))
    logsumexp2!(view(npaths, 1), tmp′, view(xmax_r, 1))

    unsafe_free!(src)
    unsafe_free!(dst)
    unsafe_free!(A)
    unsafe_free!(det)
    ipiv !== nothing && unsafe_free!(ipiv)
    unsafe_free!(info)

    return npaths
end

using Distributions: Categorical

function Slicing.sample_path(
        npaths::AbstractGPUVector{Float32},
        DV::AbstractGPUMatrix{NTuple{2, I}},
        C::AbstractGPUVector{SVector{N, Int}}
    ) where {N, I <: Integer}
    backend = get_backend(npaths)
    path = allocate(backend, NTuple{2, I}, N, N + 2)
    src = allocate(backend, NTuple{2, I}, N, 1)
    dst = allocate(backend, NTuple{2, I}, N, length(C))
    src′, dst′ = reinterpret(reshape, SVector{N, NTuple{2, I}}, src), reinterpret(reshape, SVector{N, NTuple{2, I}}, dst)

    A = allocate_lu(backend, Float32, N, N, length(C))
    det_cpu = Vector{Float32}(undef, length(C))
    GC.@preserve det_cpu begin
        det = unsafe_wrap(typeof(npaths).name.wrapper, pointer(det_cpu), size(det_cpu))
        ipiv = requires_pivot(backend) ? allocate_lu(backend, int_type(backend), N, length(C)) : nothing
        info = allocate_lu(backend, int_type(backend), length(C))

        i = 1
        path[:, 1] .= Ref((I(0), I(0)))
        src[:, 1] .= Ref((I(0), I(0)))
        for j in 1:N
            GPUArrays.vectorized_getindex!(dst, view(DV, :, j), reinterpret(reshape, Int, C))

            path_matrices!(backend)(reshape(A, N, N, length(C), 1), src′, dst′; ndrange = (length(C), 1))
            batched_det!(det, A, ipiv, info)

            det .*= exp.(view(npaths, (j - 1) * length(C) + 1 .+ (1:length(C)))) ./ exp.(view(npaths, max(1, (j - 2) * length(C) + 1 + i)))
            synchronize(backend)
            i = rand(Categorical(det_cpu))
            copyto!(view(path, :, j + 1), view(dst, :, i))
            copyto!(view(src, :, 1), view(dst, :, i))
        end
        unsafe_free!(det)
        ipiv !== nothing && unsafe_free!(ipiv)
        unsafe_free!(info)
    end
    unsafe_free!(src)
    unsafe_free!(dst)
    unsafe_free!(A)

    path[:, end] .= Ref((I(N), I(N)))
    return path
end

end
