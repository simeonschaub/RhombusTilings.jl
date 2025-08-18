using OpenCL, KernelAbstractions
using LinearAlgebra.BLAS: BlasInt
using RecursiveFactorization: lu!

@kernel function det_kernel_pivot!(res, @Const(A), @Const(ipiv), @Const(info))
    k = @index(Global)
    @inbounds if !iszero(info[k])
        res[k] = 0.0f0
    else
        p = 1.0f0
        s = false
        for i in 1:size(A, 1)
            p *= A[i, i, k]
            s ⊻= ipiv[i, k] != i
        end
        res[k] = s ? -p : p
    end
end

function batched_det!(
        _res::CLVector{Float32, cl.UnifiedSharedMemory},
        _A::CLArray{Float32, 3, cl.UnifiedSharedMemory},
        _ipiv::CLMatrix{BlasInt, cl.UnifiedSharedMemory},
        _info::CLVector{BlasInt, cl.UnifiedSharedMemory},
    )
    GC.@preserve _res _A _ipiv _info begin
        res = unsafe_wrap(Array, _res)
        A = unsafe_wrap(Array, _A)
        ipiv = unsafe_wrap(Array, _ipiv)
        info = unsafe_wrap(Array, _info)

        batch_count = size(A, 3)
        @assert length(res) ≥ batch_count
        @assert size(ipiv, 2) ≥ batch_count
        @assert length(info) ≥ batch_count

        Threads.@threads for k in 1:batch_count
            info[k] = lu!(view(A, :, :, k), view(ipiv, :, k)).info
        end

        kernel = det_kernel_pivot!(OpenCLBackend())
        kernel(_res, _A, _ipiv, _info; ndrange = batch_count)
    end
    return _res
end

requires_pivot(::OpenCLBackend) = true
function allocate_lu(::OpenCLBackend, ::Type{T}, dims::Vararg{Integer, N}) where {T, N}
    return CLArray{T, N, cl.UnifiedSharedMemory}(undef, dims...)
end
int_type(::OpenCLBackend) = BlasInt


using SPIRVIntrinsics
using SPIRVIntrinsics: LLVMPtr, atomic_cmpxchg!, @typed_ccall

reinterpret_llvmptr(::Type{T}, ptr::LLVMPtr{S, A}) where {T, S, A} = reinterpret(LLVMPtr{T, A}, ptr)
function _reinterpret(::Type{UInt64}, (a, b)::NTuple{2, Float32})
    return UInt64(reinterpret(UInt32, a)) | (UInt64(reinterpret(UInt32, b)) << 32)
end
function _reinterpret(::Type{NTuple{2, Float32}}, x::UInt64)
    a, b = x % UInt32, (x >> 32) % UInt32
    return reinterpret(Float32, a), reinterpret(Float32, b)
end
function atomic_arrayset(A::AbstractArray{NTuple{2, Float32}}, I::Integer, op::Function, val::NTuple{2, Float32})
    ptr = reinterpret_llvmptr(UInt64, pointer(A, I))
    old = Base.unsafe_load(ptr, 1)
    while true
        cmp = old
        new = op(_reinterpret(NTuple{2, Float32}, old), val)
        old = atomic_cmpxchg!(ptr, cmp, _reinterpret(UInt64, new))
        (old == cmp) && return new
    end
    return
end

function sub_group_shuffle(x::UInt64, idx::Integer)
    SPIRVIntrinsics.@builtin_ccall("sub_group_shuffle", UInt64, (UInt64, Int32), x, Int32(idx))
end

function reduce_kernel(op, result, X, M, N, init)
    col = get_global_id(1)
    row_thread = get_global_id(2)
    row_stride = get_global_size(2)

    if col > N
        return nothing
    end

    partial = init
    for row in row_thread:row_stride:M
        partial = op(partial, @inbounds X[row, col])
    end

    # Subgroup shuffle-based warp reduction
    lane = get_sub_group_local_id() % Int32 - Int32(1)
    width = get_sub_group_size() % Int32

    offset = Int32(1)
    while offset < width
        other = _reinterpret(NTuple{2, Float32}, sub_group_shuffle(_reinterpret(UInt64, partial), max(lane - offset, 0)))
        partial = op(partial, other)
        offset <<= Int32(1)
    end

    # Only one thread writes result
    if lane == Int32(0)
        atomic_arrayset(result, col, op, partial)
    end
    return nothing
end

using LogExpFunctions: _logsumexp_onepass_op
function logsumexp2!(out::CLMatrix, X::CLMatrix{<:Number}, xmax_r::CLMatrix{NTuple{2, FT}}) where {FT}
    fill!(xmax_r, (FT(-Inf), zero(FT)))
    @assert size(xmax_r, 1) == 1
    @assert size(X, 2) == size(xmax_r, 2)
    local_size, global_size = (1, 64), (size(X, 2), 64)
    @opencl global_size local_size backend = :khronos reduce_kernel(_logsumexp_onepass_op, xmax_r, X, size(X, 1), size(X, 2), (FT(-Inf), zero(FT)))
    return @. out = first(xmax_r) + log1p(last(xmax_r))
end
