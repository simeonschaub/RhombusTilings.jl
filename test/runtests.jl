using RhombusTilings
using TestItemRunner

@run_package_tests verbose = true

@testitem "JET" begin
    using JET

    test_package("RhombusTilings")
    test_call(shuffled_tiling, Tuple{NTuple{5, Int}, Int})
end

@testitem "AllocCheck" begin
    using AllocCheck

    @test isempty(check_allocs(RhombusTilings.shuffle!, Tuple{RhombusTiling{16}}))
end
