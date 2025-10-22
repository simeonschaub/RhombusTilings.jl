module Slicing

using ..RhombusTilings
using Adapt, Combinatorics, LinearAlgebra, StaticArrays, Distributions
using Graphs, MetaGraphsNext

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

function slicing_paths(DV::Matrix{NTuple{2, I}}, C::Vector{SVector{N, Int}}; backend = OpenCLBackend(), batch_size = 100) where {N, I}
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

function slicing_paths(hahn_paths::HahnPaths, I = Int8; kwargs...)
    DV = distinguished_vertices(hahn_paths, I)
    (; N, T) = hahn_paths
    C = SVector{N}.(with_replacement_combinations(1:T, N))
	return slicing_paths(DV, C; kwargs...)
end

function slicing_graph((; vert, adj, dims)::RhombusTiling{N, T}) where {N, T}
	g = MetaGraph(SimpleDiGraph{Int}(); label_type = NTuple{N, T}, vertex_data_type = Nothing, edge_data_type = Pair{UInt8, NTuple{2, Int}}, weight_function = first)

	function _add_edge!(loc1, loc2, side, tile, idx)
		g[loc1] = g[loc2] = nothing
		if haskey(g, loc1, loc2)
			_side, tiles = g[loc1, loc2]
			@assert side == _side
			@assert tiles[idx] == 0
		else
			tiles = (0, 0)
		end
		g[loc1, loc2] = side => Base.setindex(tiles, tile, idx)
	end

	for i in vertices(adj)
		i₁, i₂ = extrema(Iterators.filter(!iszero, adj.wts[i]))
		loc = vert[i]
		loc₁ = ntuple(i -> loc[i] + (i == i₁), Val(N))
		loc₂ = ntuple(i -> loc[i] + (i == i₂), Val(N))
		loc₁₂ = ntuple(i -> loc₁[i] + (i == i₂), Val(N))

		_add_edge!(loc, loc₁, i₁, i, 2)
		_add_edge!(loc, loc₂, i₂, i, 1)
		_add_edge!(loc₁, loc₁₂, i₂, i, 2)
		_add_edge!(loc₂, loc₁₂, i₁, i, 1)
	end
	return g
end

#g_sl = slicing_graph(rotr(RhombusTiling(hex)))

function to_slicing_paths(p, g_sl) where {N}
	paths = [[code_for(g_sl, (0, 0, 0))] for _ in 1:N]
	w = weights(g_sl)
	for i in 1:(length(p) - 1)
		v = sample_lattice_paths(p[i], p[i + 1])
		for j in eachindex(v)
			steps = map(.-, (j == length(v) ? p[i + 1] : v[j + 1]), v[j])
			for ((dx, dy), path) in zip(steps, paths)
				c = path[end]
				n = outneighbors(g_sl, c)
				if dx > 0
					k = findfirst(n -> w[c, n] == 3, n)
					push!(path, n[k])
				else
					for _ in 1:dy
						k = findfirst(n -> w[c, n] == 1, n)
						push!(path, n[k])
						c = n[k]
						n = outneighbors(g_sl, c)
					end
				end
			end
		end
		for path in paths
			c = path[end]
			n = outneighbors(g_sl, c)
			isempty(n) && continue
			k = findfirst(n -> w[c, n] == 2, n)
			push!(path, n[k])
		end
	end

	return stack(paths)'
end

function slice!((; adj, vert, dims)::RhombusTiling{N, T}, g, paths) where {N, T}
	starting_tiles = [Set{Int}() for _ in 1:(size(paths, 1) + 1)]
	for i in axes(paths, 1)
		for j in 2:size(paths, 2)
			v, v′ = paths[i, j - 1], paths[i, j]
			side, tiles = g[label_for(g, v), label_for(g, v′)]
			if tiles[1] != 0
				if i == 1 || !(paths[i - 1, j - 1] == v && paths[i - 1, j] == v′)
					push!(starting_tiles[i], tiles[1])
				end
				tiles[2] != 0 || continue
				if has_edge(adj, tiles...)
					adj = rem_edge!(adj, tiles...)
				end
			else
				@assert tiles[2] != 0
			end
			if i == size(paths, 1)
				push!(starting_tiles[i + 1], tiles[2])
			end
		end
	end

	vert′ = similar(vert, NTuple{N + 1, Int})
	fill!(vert′, ntuple(_ -> -1, N + 1))
	for i in 0:size(paths, 1)
		for tile in starting_tiles[i + 1]
			for v in BFSIterator(adj, tile)
				v == 0 && continue
				vert′[v] = (vert[v]..., i)
			end
		end
	end

	for i in axes(paths, 1)
		prev_tile = 0
		for j in 2:size(paths, 2)
			add_vertex!(adj)
			v, v′ = paths[i, j - 1], paths[i, j]
			loc = (label_for(g, v)..., i - 1)
			push!(vert′, loc)

			new_tile = nv(adj)
			side, tiles = g[label_for(g, v), label_for(g, v′)]

			if i > 1 && paths[i - 1, j - 1] == v && paths[i - 1, j] == v′
				adj = add_edge!(adj, new_tile, new_tile - size(paths, 2) + 1, side)
			elseif tiles[1] != 0 && !has_edge(adj, new_tile, tiles[1])
				adj = add_edge!(adj, new_tile, tiles[1], side)
			end
			if i < size(paths, 1) && paths[i + 1, j - 1] == v && paths[i + 1, j] == v′
			elseif tiles[2] != 0 && !has_edge(adj, new_tile, tiles[2])
				adj = add_edge!(adj, new_tile, tiles[2], side)
			end
			if prev_tile != 0
				adj = add_edge!(adj, prev_tile, new_tile, UInt8(N + 1))
			end
			prev_tile = new_tile
		end
	end

	return RhombusTiling(adj, vert′, (dims..., size(paths, 1)))
end

include("ka.jl")
include("opencl.jl")

end
