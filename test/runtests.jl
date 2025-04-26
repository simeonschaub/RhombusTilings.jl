using RhombusTilings
using TestItemRunner

@run_package_tests verbose = true

@testitem "Test Paths" begin
    using Graphs, SimpleWeightedGraphs

    t = shuffled_nflips(ntuple(_ -> 16, 5), 10^8)
    (; adj, vert) = t
    @test nv(adj) == 2560
    @test ne(adj) == 10080
    starting_points = findall(vertices(adj)) do i
        all(iszero, vert[i][2:end]) && 0x01 ∈ get_weight.(Ref(adj), i, neighbors(adj, i))
    end
    @test length(starting_points) == 16
    sort!(starting_points; by = i -> vert[i])
    @testset "Path $n" for (n, i) in pairs(starting_points)
        i_prev = 0
        for _ in 1:1000
            _i = findfirst(neighbors(adj, i)) do j
                j ∉ (0, i_prev) && get_weight(adj, i, j) == 0x01
            end
            _i === nothing && break
            i_prev, i = i, neighbors(adj, i)[_i[]]
            step = adj.wts[i_prev][findfirst(>(0x01), adj.wts[i_prev])]
            @test vert[i] == ntuple(j -> vert[i_prev][j] + (j == step), 5)
        end
        last_step = adj.wts[i][findfirst(>(0x01), adj.wts[i])]
        end_pos = ntuple(j -> vert[i][j] + (j == last_step), 5)
        @test end_pos == ntuple(j -> j == 1 ? n - 1 : 16, 5)
    end

    @test shuffled_tiling((5, 5, 5), 10^7) isa RhombusTiling
end

@testitem "HybridGraph" begin
    using RhombusTilings: HybridGraph
    using Graphs, SimpleWeightedGraphs

    g = HybridGraph{2, Float64}(10)
    @test nv(g) == 10
    @test vertices(g) == 1:10
    @test edgetype(g) == Int
    @test ne(g) == 0
    @test outneighbors(g, 1) == [0, 0]
    @test !is_directed(g)

    @test_throws ArgumentError rem_edge!(g, 1, 2)
    g = add_edge!(g, 1, 2, 1.2)
    g = add_edge!(g, 1, 3, 3.4)
    @test_throws ArgumentError add_edge!(g, 1, 4, 5.6)
    @test ne(g) == 4
    @test outneighbors(g, 1) == [2, 3]
    @test get_weight(g, 1, 2) == 1.2
end

@testitem "MakieExtension" begin
    using CairoMakie

    t = base_tiling(1, 2, 3, 4, 5, 6)
    p, c = Base.get_extension(RhombusTilings, :MakieExtension).polys(t)
    @test c == mapreduce(vcat, Iterators.product(1:6, 1:6)) do (n, m)
        m < n ? fill(sum((7 - m):5) + (n - m), m * n) : Int[]
    end
    @test length(p) == length(c)

    # TODO: add some better tests
    @test plot(t) isa Makie.FigureAxisPlot
end

@testitem "JET" begin
    using JET

    test_package("RhombusTilings")
    test_call(shuffled_tiling, Tuple{NTuple{5, Int}, Int})
end

@testitem "AllocCheck" begin
    using AllocCheck

    @test isempty(check_allocs(RhombusTilings.shuffle!, Tuple{RhombusTiling{16}}))
end
