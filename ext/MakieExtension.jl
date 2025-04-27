module MakieExtension

using RhombusTilings
using GeometryBasics
using Makie

function polys((; adj, vert)::RhombusTiling{N}, swap_xy = false) where {N}
    res = Polygon{2, Float32}[]
    basis = Point2f.(reim.(cispi.((0:(N - 1)) ./ N)))
    if swap_xy
        basis .= Point2f.(last.(basis), first.(basis))
    end
    color = Int[]
    pts, text = Point2f[], Makie.RichText[]
    for (i, loc) in pairs(vert)
        origin = sum(loc .* basis)
        i₁, i₂ = extrema(filter(!iszero, adj.wts[i]))
        a, b = basis[i₁], basis[i₂]
        push!(res, Polygon([origin, origin + a, origin + a + b, origin + b]))

        c = sum((N + 1 - i₁):(N - 1)) + (i₂ - i₁)
        push!(color, c)

        push!(pts, origin + (a + b) / 2)
        push!(text, rich("$i", subscript("$i₁,$i₂")))
    end
    return res, pts, text, color
end

@recipe(RhombusTilingPlot, t) do scene
    Attributes(;
        swap_xy = false,
    )
end

Makie.plottype(::RhombusTiling) = RhombusTilingPlot

function Makie.plot!(x::RhombusTilingPlot{<:Tuple{RhombusTiling}})
    p = map(polys, x[:t], x[:swap_xy])
    p = map(polys, x[:t])
    poly!(x, Makie.shared_attributes(x, Poly), map(first, p); color = map(last, p))
    text!(x, map(p -> p[2], p); text = map(p -> p[3], p), align = (:center, :center), color = :white)
    return x
end

Makie.plottype(::HahnPaths) = Series
Makie.convert_arguments(p::Type{<:Series}, (; paths)::HahnPaths) = Makie.convert_arguments(p, paths)

end
