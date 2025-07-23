module RhombusTilings

using Graphs, SimpleWeightedGraphs, SparseArrays
using StaticArrays, Random
using Dictionaries

export RhombusTiling, shuffled_tiling, shuffled_tiling_minmax, rotr
export HahnPaths, sample_hahn_paths

include("hybridgraph.jl")

struct RhombusTiling{N, T <: Integer}
    adj::HybridGraph{4, UInt8, Int}
    vert::Vector{NTuple{N, T}}
    dims::NTuple{N, Int}
end
Base.copy((; adj, vert, dims)::RhombusTiling) = RhombusTiling(copy(adj), copy(vert), dims)

mutable struct RhombusTilingBuilder{N, T <: Integer}
    adj::HybridGraph{4, UInt8, Int}
    const vert::Vector{NTuple{N, T}}
    const sides::Dictionary{Pair{NTuple{N, T}, UInt8}, Int}
end
function RhombusTilingBuilder{N, T}() where {N, T}
    return RhombusTilingBuilder{N, T}(HybridGraph{4, UInt8}(0), NTuple{N, T}[], Dictionary{Pair{NTuple{N, T}, UInt8}, Int}())
end
function RhombusTiling((; adj, vert)::RhombusTilingBuilder{N, T}, dims::NTuple{N, Int}) where {N, T}
    return RhombusTiling{N, T}(adj, vert, dims)
end

function add_side!(builder::RhombusTilingBuilder{N}, loc::NTuple{N, Integer}, side::Integer, j::Int) where {N}
    (; sides) = builder
    hastoken, token = gettoken!(sides, loc => UInt8(side))
    if hastoken
        i = gettokenvalue(sides, token)
        @assert i != 0
        builder.adj = add_edge!(builder.adj, i, j, UInt8(side))
        settokenvalue!(sides, token, 0)
    else
        settokenvalue!(sides, token, j)
    end
    return builder
end
function add_tile!(builder::RhombusTilingBuilder{N}, loc::NTuple{N, Integer}, (side_1, side_2)::NTuple{2, Integer}) where {N}
    (; adj, vert) = builder
    add_vertex!(adj)
    push!(vert, loc)
    j = lastindex(vert)

    foreach(
        (
            loc => UInt8(side_1),
            loc => UInt8(side_2),
            ntuple(i -> loc[i] + (i == side_2), Val(N)) => UInt8(side_1),
            ntuple(i -> loc[i] + (i == side_1), Val(N)) => UInt8(side_2),
        ),
    ) do (loc, side)
        add_side!(builder, loc, side, j)
    end
    return builder
end


function RhombusTiling(dims::NTuple{N, Int}, init = :MIN; T = UInt8) where {N}
    builder = RhombusTilingBuilder{N, T}()

    origin_0 = ntuple(_ -> 0x00, Val(N))
    for m in (N - 1):-1:1
        side_1 = N - m
        origin = origin_0
        for n in 1:m
            side_2 = init === :MIN ? side_1 + n : N + 1 - n
            for i in 0:(dims[side_1] - 1), j in 0:(dims[side_2] - 1)
                loc = let origin = origin
                    ntuple(Val(N)) do k
                        origin[k] + T(i) * (k == side_1) + T(j) * (k == side_2)
                    end
                end
                add_tile!(builder, loc, (side_1, side_2))
            end
            origin = let origin = origin
                ntuple(Val(N)) do k
                    origin[k] + T(dims[side_2]) * (k == side_2)
                end
            end
        end
        if init !== :MIN
            origin_0 = let origin_0 = origin_0
                ntuple(Val(N)) do k
                    origin_0[k] + T(dims[side_1]) * (k == side_1)
                end
            end
        end
    end

    return RhombusTiling(builder, dims)
end

function Base.:(==)(t1::RhombusTiling, t2::RhombusTiling)
    t1.dims == t2.dims || return false
    d1, d2 = map((t1, t2)) do (; adj, vert)
        Dict(vert[i] => extrema(Iterators.filter(!iszero, adj.wts[i])) for i in vertices(adj))
    end
    return d1 == d2
end

Base.@constprop :aggressive function shuffle!((; adj, vert)::RhombusTiling{N}, j::Int, k::Int, l::Int, up = nothing) where {N}
    l == k && return nothing
    s₁ = @inbounds get_weight(adj, k, l)
    s₁ == 0x00 && return nothing
    s₂ = @inbounds get_weight(adj, l, j)
    s₂ == 0x00 && return nothing
    s₃ = @inbounds get_weight(adj, j, k)
    s₃ == 0x00 && return nothing

    _sides = SA[s₁, s₂, s₃]
    jkl = SA[j, k, l]
    π = sort(SA[1, 2, 3]; by = i -> (vert[jkl[i]], -_sides[i]))
    j, k, l = @inbounds jkl[π]
    loc₁ = @inbounds vert[j]
    loc₂ = @inbounds vert[k]
    sides = sort(_sides)
    @inbounds if loc₁ == loc₂
        up !== false || return nothing
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

        return true
    elseif up !== true
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

        return false
    end

    return nothing
end

function shuffle!(t::RhombusTiling{N}, j::Int, k::Int) where {N}
    @inbounds for l in neighbors(t.adj, j)
        up = shuffle!(t, j, k, l)
        up !== nothing && return l, up
    end
    return nothing
end

function shuffle!(t::RhombusTiling{N}; rng = Random.default_rng()) where {N}
    (; adj) = t
    j = rand(rng, vertices(adj))
    k = rand(rng, @inbounds adj.adj[j])
    k == 0 && return false
    return shuffle!(t, j, k) !== nothing
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

function shuffled_tiling_minmax(dims, max_steps; rng = Xoshiro())
    MIN, MAX = RhombusTiling(dims, :MIN), RhombusTiling(dims, :MAX)
    MIN, MAX = map((MIN, MAX)) do (; adj, vert, dims)
        π = sortperm(
            map(vertices(adj)) do i
                extrema(Iterators.filter(!iszero, adj.wts[i])), vert[i]
            end
        )
        map!(adj.adj, adj.adj) do n
            map(i -> i == 0 ? 0 : π[i], n)
        end
        return RhombusTiling(HybridGraph(adj.adj[π], adj.wts[π], adj.ne), vert[π], dims)
    end
    for _ in 1:(max_steps ÷ (16 * 1024))
        for _ in 1:(16 * 1024)
            j = rand(rng, vertices(MIN.adj))
            if rand(rng, Bool)
                k = rand(rng, @inbounds MIN.adj.adj[j])
                k == 0 && continue
                l_up = shuffle!(MIN, j, k)
                l_up !== nothing && shuffle!(MAX, j, k, l_up...)
            else
                k = rand(rng, @inbounds MAX.adj.adj[j])
                k == 0 && continue
                l_up = shuffle!(MAX, j, k)
                l_up !== nothing && shuffle!(MIN, j, k, l_up...)
            end
        end
        MIN == MAX && break
    end
    return MIN, MAX
end

function rotr((; adj, vert, dims)::RhombusTiling{N, T}) where {N, T}
    wts′ = map(adj.wts) do w
        map(s -> iszero(s) ? s : mod1(s + 0x01, UInt8(N)), w)
    end
    vert′ = map(eachindex(vert), vert) do j, v
        ntuple(i -> i == 1 ? T(dims[mod1(i - 1, N)]) - v[mod1(i - 1, N)] - any(==(0x01), wts′[j]) : v[mod1(i - 1, N)], N)
    end
    dims′ = ntuple(i -> dims[mod1(i - 1, N)], N)
    return RhombusTiling(HybridGraph(adj.adj, wts′, adj.ne), vert′, dims′)
end

include("hahn_paths.jl")
include("slicing.jl")

end
