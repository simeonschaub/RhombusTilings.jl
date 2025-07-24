using AMDGPU, StaticArrays
using RhombusTilings.Slicing
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
c = let
    src = reinterpret(reshape, SVector{N, NTuple{2, I}}, view(DV, :, 1)[reinterpret(reshape, Int, C)])[1:1000]
    dst = reinterpret(reshape, SVector{N, NTuple{2, I}}, view(DV, :, 2)[reinterpret(reshape, Int, C)])
    Slicing.count_paths(src, dst)
end

count(tuple.(dst, reshape(src, 1, :))) do (d, s)
    x_D, x_A = first.(s), first.(d)
    y_D, y_A = last.(s), last.(d)
    all(x_D .≤ x_A) && all(y_D .≤ y_A)
end / (length(dst) * length(src))
