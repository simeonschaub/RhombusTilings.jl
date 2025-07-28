
module OpenCLExtension

using OpenCL, KernelAbstractions, RhombusTilings.Slicing
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

function Slicing.batched_det!(
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

Slicing.requires_pivot(::OpenCLBackend) = true
function Slicing.allocate_lu(::OpenCLBackend, ::Type{T}, dims::Vararg{Integer, N}) where {T, N}
    return CLArray{T, N, cl.UnifiedSharedMemory}(undef, dims...)
end
Slicing.int_type(::OpenCLBackend) = BlasInt

end
