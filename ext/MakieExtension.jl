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

@recipe RhombusTilingPlot (t,) begin
    swap_xy = false
    indices = false
    Makie.documented_attributes(Poly)...
end

Makie.plottype(::RhombusTiling) = RhombusTilingPlot

function Makie.plot!(x::RhombusTilingPlot{<:Tuple{RhombusTiling}})
    map!(polys, x.attributes, [:t, :swap_xy], [:res, :pts, :text, :_color])
    poly!(x, Makie.shared_attributes(x, Poly), x.res; color = x._color)
    text!(x, x.pts; x.text, align = (:center, :center), color = :white, visible = x.indices)
    return x
end

Makie.plottype(::HahnPaths) = Series
Makie.convert_arguments(p::Type{<:Series}, (; paths)::HahnPaths) = Makie.convert_arguments(p, paths)

end
