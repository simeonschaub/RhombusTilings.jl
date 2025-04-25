using RhombusTilings
using TestItemRunner

@run_package_tests verbose = true

@testitem "Test Paths" begin
    using Graphs, SimpleWeightedGraphs

    t = shuffled_nflips(ntuple(_ -> 16, 5), 10^8)
    (; adj, vert) = t
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
