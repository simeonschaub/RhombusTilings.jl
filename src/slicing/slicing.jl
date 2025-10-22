module Slicing

using ..RhombusTilings
using Adapt, Combinatorics, LinearAlgebra, StaticArrays, Distributions

export compute_npaths, sample_path, count_paths, slicing_paths

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

function npaths(src::SVector{N, NTuple{2, I}}, dst::SVector{N, NTuple{2, I}}) where {N, I}
    inc = SizedVector{N}(I(0):I(N - 1))
	x_D, y_D = first.(src), last.(src)
	x_A, y_A = first.(dst), last.(dst)
	return det(@. binom(x_A' - x_D + y_A' - y_D, x_A' - x_D + inc' - inc))
end

function sample_lattice_paths(src::SVector{N, NTuple{2, I}}, dst::SVector{N, NTuple{2, I}}) where {N, I}
	v = [src]
	n = npaths(src, dst)
	ranges = map(:, first.(src), first.(dst))
	for x in minimum(first, ranges):maximum(last, ranges) - one(I)
		y_ranges = map(ranges, last(v), dst) do r, (_, y_D), (_, y_A)
			x < first(r) && return y_D:y_D
			x >= last(r) && return y_A:y_A
			return y_D:y_A
		end
		y_D′ = SVector.(Iterators.filter(Iterators.product(y_ranges...)) do y
			all(ntuple(i -> y[i] ≥ y[i + 1], N - 1))
		end)
		x_D′ = map(ranges) do r
			x < first(r) ? first(r) : (x ≥ last(r) ? last(r) : x + one(I))
		end
		ns = map(y_D′) do y_D
			npaths(tuple.(x_D′, y_D), dst)
		end
		i = rand(Distributions.Categorical(ns ./ n))
		n = ns[i]
		push!(v, tuple.(first.(v[end]), y_D′[i]))
		push!(v, tuple.(x_D′, y_D′[i]))
	end
	return v
end

function slicing_paths(hahn_paths::HahnPaths, I = Int8; backend = OpenCLBackend(), batch_size = 100)
    DV = distinguished_vertices(hahn_paths, I)
    (; N, T) = hahn_paths
    C = SVector{N}.(with_replacement_combinations(1:T, N))
    DV′ = adapt(backend, DV)
    C′ = adapt(backend, C)
    npaths = compute_npaths(DV′, C′, batch_size)
    p = reinterpret(reshape, SVector{N, NTuple{2, I}}, adapt(Array, sample_path(npaths, DV′, C′)))
	DV_path = SVector{N, NTuple{2, I}}[]
	for i in 1:(length(p) - 1)
		append!(DV_path, sample_lattice_paths(p[i], p[i + 1]))
	end
	push!(DV_path, p[end])
	return reinterpret(reshape, NTuple{2, I}, DV_path)
end

include("ka.jl")
include("opencl.jl")

end
