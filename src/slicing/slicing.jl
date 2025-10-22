module Slicing

using ..RhombusTilings
using Adapt, Combinatorics

export compute_npaths, sample_path, count_paths, slice

function distinguished_vertices((; paths, N, T, S)::HahnPaths, I = Int8)
	DV = similar(paths, NTuple{2, I}, T + 1, N)
	for (j, path) in pairs(eachrow(paths))
		x, y = 0, T - S
		DV[1, j] = x, y
		for i in 1:T
			if path[i] == path[i + 1]
				y -= 1
			else
				x += 1
			end
			DV[i + 1, j] = x, y
		end
	end
	return DV
end

function compute_npaths(hahn_paths::HahnPaths, I = Int8; backend = OpenCLBackend(), batch_size = 100)
    DV = distinguished_vertices(hahn_paths, I)
    (; N, T) = hahn_paths
    C = SVector{N}.(with_replacement_combinations(1:T, N))
    return compute_npaths(adapt(backend, DV), adapt(backend, C), batch_size)
end

function slice(hahn_paths::HahnPaths, I = Int8; backend = OpenCLBackend(), batch_size = 100)
    DV = distinguished_vertices(hahn_paths, I)
    (; N, T) = hahn_paths
    C = SVector{N}.(with_replacement_combinations(1:T, N))
    DV′ = adapt(backend, DV)
    C′ = adapt(backend, C)
    npaths = compute_npaths(DV′, C′, batch_size)
    path = sample_path(npaths, DV′, C′)
end

include("ka.jl")
include("opencl.jl")

end
