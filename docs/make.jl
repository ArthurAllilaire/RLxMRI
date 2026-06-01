using Documenter
using QalibreMDPhantom

makedocs(
    sitename = "QalibreMDPhantom.jl",
    authors  = "Arthur Allilaire",
    modules  = [QalibreMDPhantom],
    format   = Documenter.HTML(
        prettyurls       = get(ENV, "CI", nothing) == "true",
        canonical        = "https://arthurallilaire.github.io/QalibreMDPhantom.jl",
        edit_link        = "main",
        assets           = String[],
    ),
    pages = [
        "Home"            => "index.md",
        "Getting Started" => "getting_started.md",
        "Guides" => [
            "Phantom Construction" => "phantom.md",
            "Pulse Sequences"      => "sequences.md",
            "Parameter Fitting"    => "fitting.md",
        ],
        "API Reference"   => "api.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(
    repo   = "github.com/arthuraa/QalibreMDPhantom.jl.git",
    branch = "gh-pages",
    push_preview = true,
)
