using RhombusTilings
using TestItemRunner

@run_package_tests verbose = true

@testitem "Basic Functionality" begin
    using Graphs

    t = shuffled_tiling((5, 5, 5), 10^7)
    @test nv(t.adj) == 75
    @test ne(t.adj) == 270
    @test contains(repr(t), "RhombusTiling{3, UInt8}({75, 270} undirected simple Int64 graph with UInt8 weights, Tuple{UInt8, UInt8, UInt8}[")
end

@testitem "Test Paths" begin
    using Graphs, SimpleWeightedGraphs

    function verify_tiling((; adj, vert, dims)::RhombusTiling{N}) where {N}
        starting_points = findall(vertices(adj)) do i
            all(iszero, vert[i][2:end]) && 0x01 ∈ get_weight.(Ref(adj), i, neighbors(adj, i))
        end
        @test length(starting_points) == dims[1]
        sort!(starting_points; by = i -> vert[i])
        @testset "Path $n" for (n, i) in pairs(starting_points)
            i_prev = 0
            for _ in 1:1000
                _i = findfirst(adj.adj[i]) do j
                    j ∉ (0, i_prev) && get_weight(adj, i, j) == 0x01
                end
                _i === nothing && break
                i_prev, i = i, adj.adj[i][_i[]]
                step = adj.wts[i_prev][findfirst(>(0x01), adj.wts[i_prev])]
                @test vert[i] == ntuple(j -> vert[i_prev][j] + (j == step), N)
            end
            last_step = adj.wts[i][findfirst(>(0x01), adj.wts[i])]
            end_pos = ntuple(j -> vert[i][j] + (j == last_step), N)
            @test end_pos == ntuple(j -> j == 1 ? n - 1 : dims[j], N)
        end
    end

    @testset "shuffled_tiling" begin
        t = shuffled_tiling(ntuple(_ -> 16, 5); nflips = 10^8)
        (; adj) = t
        @test nv(adj) == 2560
        @test ne(adj) == 10080
        verify_tiling(t)
    end

    @testset "RhombusTiling(sample_hahn_paths(...))" begin
        t = RhombusTiling(sample_hahn_paths(16, 300, 10))
        (; adj) = t
        @test nv(adj) == 7700
        @test ne(adj) == 30168
        verify_tiling(t)
    end
end

@testitem "HybridGraph" begin
    using RhombusTilings: HybridGraph
    using Graphs, SimpleWeightedGraphs, SparseArrays

    g = HybridGraph{2, Float64}(10)
    @test nv(g) == 10
    @test vertices(g) == 1:10
    @test edgetype(g) == Int
    @test ne(g) == 0
    @test isempty(outneighbors(g, 1))
    @test !is_directed(g)
    @test iszero(adjacency_matrix(g))

    @test_throws ArgumentError rem_edge!(g, 1, 2)
    g = add_edge!(g, 1, 2, 1.2)
    g = add_edge!(g, 1, 3, 3.4)
    @test_throws ArgumentError add_edge!(g, 1, 4, 5.6)
    @test ne(g) == 4
    @test collect(outneighbors(g, 1)) == [2, 3]
    @test collect(outneighbors(g, 2)) == [1]
    @test collect(outneighbors(g, 3)) == [1]
    @test get_weight(g, 1, 2) == 1.2
    @test adjacency_matrix(g) == sparse([1, 1, 2, 3], [2, 3, 1, 1], [1.2, 3.4, 1.2, 3.4], 10, 10)

    g = rem_edge!(g, 1, 3)
    @test ne(g) == 2
    @test collect(outneighbors(g, 1)) == [2]
    @test collect(outneighbors(g, 2)) == [1]
    @test collect(outneighbors(g, 3)) == Int64[]
    @test adjacency_matrix(g) == sparse([1, 2], [2, 1], [1.2, 1.2], 10, 10)
end

@testitem "MakieExtension" begin
    using CairoMakie

    t = RhombusTiling((1, 2, 3, 4, 5, 6))
    p, c = Base.get_extension(RhombusTilings, :MakieExtension).polys(t)
    @test c == mapreduce(vcat, Iterators.product(1:6, 1:6)) do (n, m)
        m < n ? fill(sum((7 - m):5) + (n - m), m * n) : Int[]
    end
    @test length(p) == length(c)

    # TODO: add some better tests
    @test plot(t) isa Makie.FigureAxisPlot
    @test plot(sample_hahn_paths(2, 4, 2)) isa Makie.FigureAxisPlot
end

@testitem "JET" begin
    using JET, SIMD

    test_package("RhombusTilings"; ignored_modules = [SIMD, RhombusTilings.StaticArrays, JET.AnyFrameModule(RhombusTilings.Dictionaries)])
    test_call(shuffled_tiling, Tuple{NTuple{16, Int}, Int})
    test_call(sample_hahn_paths, NTuple{3, Int})
end

@testitem "AllocCheck" begin
    using AllocCheck

    @test isempty(check_allocs(RhombusTilings.shuffle!, Tuple{RhombusTiling{16, UInt8}}))
end
