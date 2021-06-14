using LazyLieManifolds
using Documenter

DocMeta.setdocmeta!(LazyLieManifolds, :DocTestSetup, :(using LazyLieManifolds); recursive=true)

makedocs(;
    modules=[LazyLieManifolds],
    authors="Johannes Terblanche <Affie@users.noreply.github.com> and contributors",
    repo="https://github.com/Affie/LazyLieManifolds.jl/blob/{commit}{path}#{line}",
    sitename="LazyLieManifolds.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Affie.github.io/LazyLieManifolds.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Affie/LazyLieManifolds.jl",
)
