using AMDGPU, RhombusTilings, RhombusTilings.Slicing

function generate_tiling(k; kwargs...)
    hex = sample_hahn_paths(k, 2k, k)
    paths, t, g = slicing_paths(hex; kwargs...)
    return slice!(t, g, paths)
end

using BenchmarkTools

suite = BenchmarkGroup()
for k in 1:7
    if k ≤ 5
        suite["OpenCLBackend, k=$k"] = @benchmarkable generate_tiling($k)
    end
    suite["ROCBackend, k=$k"] = @benchmarkable generate_tiling($k; backend = ROCBackend())
end

tune!(suite)
results = run(suite; verbose = true)
