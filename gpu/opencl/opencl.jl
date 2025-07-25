using OpenCL, pocl_jll, StaticArrays
using RhombusTilings.Slicing
using Combinatorics

using LinearAlgebra.BLAS
BLAS.set_num_threads(1)

N = 6
I = Int8
DV = CLMatrix{NTuple{2, I}}(
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
C = CLVector(SVector{N}.(with_replacement_combinations(1:(2N + 1), N)))

npaths = compute_npaths(DV, C)
#path = sample_path(npaths, DV, C)
