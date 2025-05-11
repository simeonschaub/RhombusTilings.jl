struct HybridGraph{N, W <: Real, T <: Integer} <: AbstractSimpleWeightedGraph{T, W}
    adj::Vector{SVector{N, T}}
    wts::Vector{SVector{N, W}}
    ne::Int
end

HybridGraph{N, W}(n::T) where {N, W, T} = HybridGraph{N, W, T}(zeros(SVector{N, T}, n), zeros(SVector{N, W}, n), 0)

Graphs.nv(g::HybridGraph) = length(g.adj)
Graphs.vertices(g::HybridGraph) = eachindex(g.adj)
Graphs.edgetype(::HybridGraph{N, W, T}) where {N, W, T} = T
Graphs.ne(g::HybridGraph) = g.ne
Base.@propagate_inbounds Graphs.outneighbors(g::HybridGraph, i) = Iterators.filter(!iszero, g.adj[i])
Graphs.is_directed(::HybridGraph) = false

@noinline function throw_add_edge_error(g::HybridGraph{N}, i, j) where {N}
    throw(ArgumentError("Cannot add edge $i -> $j to graph $g. There are already $N edges for vertex $i"))
end

Base.@propagate_inbounds function Graphs.add_edge!(g::HybridGraph{N, W, T}, i::Integer, j::Integer, w::Real) where {N, W, T}
    (; adj, wts) = g
    adj′, wts′ = reinterpret(reshape, T, adj), reinterpret(reshape, W, wts)

    n = adj[i]
    k = findfirst(iszero, n)
    k === nothing && throw_add_edge_error(g, i, j)
    adj′[k, i] = j
    wts′[k, i] = w

    n = adj[j]
    k = findfirst(iszero, n)
    k === nothing && throw_add_edge_error(g, j, i)
    adj′[k, j] = i
    wts′[k, j] = w

    return HybridGraph{N, W, T}(adj, wts, g.ne + 2)
end

@noinline function throw_replace_error(g::HybridGraph, i, j)
    throw(ArgumentError("Cannot replace edge $i -> $j in graph $g. The edge does not exist"))
end

Base.@propagate_inbounds function _replace!(g::HybridGraph{N, W, T}, i::Integer, j::Integer, j′::Integer, w′::Real) where {N, W, T}
    (; adj, wts) = g
    adj′, wts′ = reinterpret(reshape, T, adj), reinterpret(reshape, W, wts)

    n = adj[i]
    k = findfirst(==(j), n)
    k === nothing && throw_replace_error(g, i, j)
    adj′[k, i] = j′
    wts′[k, i] = w′

    return g
end

Base.@propagate_inbounds function Graphs.rem_edge!(g::HybridGraph{N, W, T}, i::Integer, j::Integer) where {N, W, T}
    _replace!(g, i, j, zero(T), zero(W))
    _replace!(g, j, i, zero(T), zero(W))
    return HybridGraph{N, W, T}(g.adj, g.wts, g.ne - 2)
end

Base.@propagate_inbounds function SimpleWeightedGraphs.get_weight((; adj, wts)::HybridGraph{N, W}, i::Integer, j::Integer) where {N, W}
    isassigned(adj, i) || return zero(W)
    n = adj[i]
    k = findfirst(==(j), n)
    return k === nothing ? zero(W) : wts[i][k]
end

function Graphs.add_vertex!((; adj, wts)::HybridGraph{N, W, T}) where {N, W, T}
    push!(adj, zero(SVector{N, T}))
    push!(wts, zero(SVector{N, W}))
    return true
end

function Graphs.adjacency_matrix(g::HybridGraph{N, W, T}, S::DataType = W; dir = :both) where {N, W, T}
    cols, rows, weights = T[], T[], S[]
    for i in vertices(g)
        for (k, j) in pairs(g.adj[i])
            iszero(j) && continue
            push!(cols, i)
            push!(rows, j)
            push!(weights, g.wts[i][k])
        end
    end
    return sparse(cols, rows, weights, nv(g), nv(g))
end
