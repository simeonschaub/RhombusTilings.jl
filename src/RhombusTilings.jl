module RhombusTilings

using Graphs, SimpleWeightedGraphs, SparseArrays
using StaticArrays, Random

export RhombusTiling, shuffled_tiling

include("hybridgraph.jl")

struct RhombusTiling{N}
    adj::HybridGraph{4, UInt8, Int}
    vert::Vector{NTuple{N, UInt8}}
    dims::NTuple{N, Int}
end

function RhombusTiling(dims::NTuple{N, Int}) where {N}
    vert = NTuple{N, UInt8}[]
    sides = Dict{NTuple{N + 1, UInt8}, Vector{Int}}()

    function add_tile!(loc, (i₁, i₂))
        push!(vert, loc)
        j = lastindex(vert)

        push!(get!(Vector{Int}, sides, (loc..., UInt8(i₁))), j)
        push!(get!(Vector{Int}, sides, (loc..., UInt8(i₂))), j)
        push!(get!(Vector{Int}, sides, (ntuple(i -> loc[i] + (i == i₂), Val(N))..., UInt8(i₁))), j)
        push!(get!(Vector{Int}, sides, (ntuple(i -> loc[i] + (i == i₁), Val(N))..., UInt8(i₂))), j)
        return nothing
    end

    for m in (N - 1):-1:1
        origin = ntuple(_ -> 0x00, Val(N))
        for n in 1:m
            for i in 0:(dims[N - m] - 1), j in 0:(dims[n + N - m] - 1)
                loc = let origin = origin
                    ntuple(Val(N)) do k
                        origin[k] + UInt8(i) * (k == N - m) + UInt8(j) * (k == n + N - m)
                    end
                end
                add_tile!(loc, (N - m, n + N - m))
            end
            origin = let origin = origin
                ntuple(Val(N)) do k
                    origin[k] + dims[n + N - m] * (k == n + N - m)
                end
            end
        end
    end

    adj = HybridGraph{4, UInt8}(length(vert))
    for ((_..., dir), edge) in sides
        length(edge) == 2 || continue
        adj = add_edge!(adj, edge[1], edge[2], dir)
    end

    return RhombusTiling(adj, vert, dims)
end

function shuffle!((; adj, vert)::RhombusTiling{N}; rng = Random.default_rng()) where {N}
    j = rand(rng, vertices(adj))
    k = rand(@inbounds adj.adj[j])
    k == 0 && return false

    @inbounds for l in neighbors(adj, j)
        l == k && continue
        s₁ = get_weight(adj, k, l)
        s₁ == 0x00 && continue
        s₂ = get_weight(adj, l, j)
        s₂ == 0x00 && continue
        s₃ = get_weight(adj, j, k)

        _sides = SA[s₁, s₂, s₃]
        jkl = SA[j, k, l]
        π = sort(SA[1, 2, 3]; by = i -> (vert[jkl[i]], -_sides[i]))
        j, k, l = jkl[π]
        loc₁ = vert[j]
        loc₂ = vert[k]
        sides = sort(_sides)
        if loc₁ == loc₂
            vert[j] = ntuple(i -> loc₁[i] + (i == sides[3]), Val(N))
            vert[k] = ntuple(i -> loc₁[i] + (i == sides[1]), Val(N))
            vert[l] = loc₁

            j₁, j₂, k₁, k₂, l₁, l₂ = 0, 0, 0, 0, 0, 0

            for i in neighbors(adj, j)
                (i == k || i == l) && continue
                side = get_weight(adj, i, j)
                side == 0x00 && continue
                if side == sides[1]
                    l₁ = i
                    _replace!(adj, i, j, l, side)
                else
                    k₁ = i
                    _replace!(adj, i, j, k, side)
                end
            end
            for i in neighbors(adj, k)
                (i == j || i == l) && continue
                side = get_weight(adj, i, k)
                side == 0x00 && continue
                if side == sides[2]
                    j₁ = i
                    _replace!(adj, i, k, j, side)
                else
                    l₂ = i
                    _replace!(adj, i, k, l, side)
                end
            end
            for i in neighbors(adj, l)
                (i == j || i == k) && continue
                side = get_weight(adj, i, l)
                side == 0x00 && continue
                if side == sides[1]
                    j₂ = i
                    _replace!(adj, i, l, j, side)
                else
                    k₂ = i
                    _replace!(adj, i, l, k, side)
                end
            end

            adj.adj[j] = SA[k, l, j₁, j₂]
            adj.wts[j] = sides[SA[2, 1, 2, 1]]
            adj.adj[k] = SA[l, j, k₁, k₂]
            adj.wts[k] = sides[SA[3, 2, 2, 3]]
            adj.adj[l] = SA[j, k, l₁, l₂]
            adj.wts[l] = sides[SA[1, 3, 1, 3]]
        else
            vert[j] = ntuple(i -> loc₁[i] + (i == sides[2]), Val(N))
            vert[k] = loc₁
            vert[l] = loc₁

            j₁, j₂, k₁, k₂, l₁, l₂ = 0, 0, 0, 0, 0, 0

            for i in neighbors(adj, j)
                (i == k || i == l) && continue
                side = get_weight(adj, i, j)
                side == 0x00 && continue
                if side == sides[1]
                    k₁ = i
                    _replace!(adj, i, j, k, side)
                else
                    l₁ = i
                    _replace!(adj, i, j, l, side)
                end
            end
            for i in neighbors(adj, k)
                (i == j || i == l) && continue
                side = get_weight(adj, i, k)
                side == 0x00 && continue
                if side == sides[1]
                    j₁ = i
                    _replace!(adj, i, k, j, side)
                else
                    l₂ = i
                    _replace!(adj, i, k, l, side)
                end
            end
            for i in neighbors(adj, l)
                (i == j || i == k) && continue
                side = get_weight(adj, i, l)
                side == 0x00 && continue
                if side == sides[2]
                    k₂ = i
                    _replace!(adj, i, l, k, side)
                else
                    j₂ = i
                    _replace!(adj, i, l, j, side)
                end
            end

            adj.adj[j] = SA[k, l, j₁, j₂]
            adj.wts[j] = sides[SA[1, 3, 1, 3]]
            adj.adj[k] = SA[l, j, k₁, k₂]
            adj.wts[k] = sides[SA[2, 1, 1, 2]]
            adj.adj[l] = SA[j, k, l₁, l₂]
            adj.wts[l] = sides[SA[3, 2, 3, 2]]
        end
        return true
    end
    return false
end

function shuffled_tiling(dims, max_steps; rng = Xoshiro(), nflips = max_steps)
    t = RhombusTiling(dims)
    for _ in 1:max_steps
        nflips -= shuffle!(t; rng)
        nflips == 0 && break
    end
    return t
end
shuffled_tiling(dims; rng = Xoshiro(), nflips) = shuffled_tiling(dims, typemax(Int); rng, nflips)

end
