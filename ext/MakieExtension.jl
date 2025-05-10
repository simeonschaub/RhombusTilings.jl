module MakieExtension

using RhombusTilings
using GeometryBasics
using Makie

function polys((; adj, vert)::RhombusTiling{N}) where {N}
    res = Polygon{2, Float32}[]
    basis = Point2f.(reim.(cispi.((0:(N - 1)) ./ N)))
    color = Int[]
    for (i, loc) in pairs(vert)
        origin = sum(loc .* basis)
        i₁, i₂ = extrema(filter(!iszero, adj.wts[i]))
        a, b = basis[i₁], basis[i₂]
        push!(res, Polygon([origin, origin + a, origin + a + b, origin + b]))

        c = sum((N + 1 - i₁):(N - 1)) + (i₂ - i₁)
        push!(color, c)
    end
    return res, color
end

@recipe(RhombusTilingPlot, t) do scene
    Attributes()
end

Makie.plottype(::RhombusTiling) = RhombusTilingPlot

function Makie.plot!(x::RhombusTilingPlot{<:Tuple{RhombusTiling}})
    p = map(polys, x[:t])
    return poly!(x, Makie.shared_attributes(x, Poly), map(first, p); color = map(last, p))
end

Makie.plottype(::HahnPaths) = Series
Makie.convert_arguments(p::Type{<:Series}, (; paths)::HahnPaths) = Makie.convert_arguments(p, paths)

end
