using RhombusTilings
using Documenter

DocMeta.setdocmeta!(RhombusTilings, :DocTestSetup, :(using RhombusTilings); recursive = true)

makedocs(;
    modules = [RhombusTilings],
    authors = "Simeon David Schaub <simeon@schaub.rocks> and contributors",
    sitename = "RhombusTilings.jl",
    format = Documenter.HTML(;
        canonical = "https://simeonschaub.github.io/RhombusTilings.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/simeonschaub/RhombusTilings.jl",
    devbranch = "main",
)
