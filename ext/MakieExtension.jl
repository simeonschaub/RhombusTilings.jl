module MakieExtension

using RhombusTilings
using GeometryBasics
using GeometryBasics: GLTriangleFace
using Makie
using Makie: Mesh

function polys((; adj, vert)::RhombusTiling{N}, swap_xy = false, basis = nothing) where {N}
    if basis === nothing
        basis = Point2f.(reim.(cispi.((0:(N - 1)) ./ N)))
    end
    if swap_xy
        basis .= Point2f.(last.(basis), first.(basis))
    end
    res = Polygon{length(eltype(basis)), Float32}[]
    color = Int[]
    pts, text = similar(basis, 0), Makie.RichText[]
    for (i, loc) in pairs(vert)
        origin = sum(loc .* basis)
        sides = filter(!iszero, adj.wts[i])
        length(sides) < 2 && continue
        i₁, i₂ = extrema(sides)
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
    basis = nothing
    indices = false
    Makie.documented_attributes(Poly)...
end

Makie.plottype(::RhombusTiling) = RhombusTilingPlot

function Makie.plot!(x::RhombusTilingPlot{<:Tuple{RhombusTiling}})
    map!(polys, x.attributes, [:t, :swap_xy, :basis], [:res, :pts, :text, :_color])
    poly!(x, Makie.shared_attributes(x, Poly), x.res; color = x._color)
    text!(x, x.pts; x.text, align = (:center, :center), color = :white, visible = x.indices)
    return x
end

function polys_mesh((; adj, vert)::RhombusTiling{N}, swap_xy = false, basis = nothing) where {N}
    if basis === nothing
        basis = Point2f.(reim.(cispi.((0:(N - 1)) ./ N)))
    end
    if swap_xy
        basis .= Point2f.(last.(basis), first.(basis))
    end

    positions = Point2f[]
    faces = GLTriangleFace[]
    colors = Int[]

    for (i, loc) in pairs(vert)
        origin = sum(loc .* basis)
        sides = filter(!iszero, adj.wts[i])
        length(sides) < 2 && continue
        i₁, i₂ = extrema(sides)
        a, b = basis[i₁], basis[i₂]

        v1 = origin
        v2 = origin + a
        v3 = origin + a + b
        v4 = origin + b

        start_idx = length(positions) + 1

        push!(positions, v1, v2, v3, v4)
        push!(faces,
              GLTriangleFace(start_idx, start_idx + 1, start_idx + 2),
              GLTriangleFace(start_idx, start_idx + 2, start_idx + 3))

        c = sum((N + 1 - i₁):(N - 1)) + (i₂ - i₁)
        push!(colors, c, c, c, c)
    end

    # Create the mesh
    return GeometryBasics.mesh(positions, faces; color = colors)
end

Makie.convert_arguments(::Type{<:Mesh}, t::RhombusTiling) = (polys_mesh(t),)

Makie.plottype(::HahnPaths) = Series
Makie.convert_arguments(p::Type{<:Series}, (; paths)::HahnPaths) = Makie.convert_arguments(p, paths)

end
