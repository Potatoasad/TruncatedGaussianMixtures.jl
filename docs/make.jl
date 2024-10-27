using Documenter, TruncatedGaussianMixtures

makedocs(
    sitename = "TruncatedGaussianMixtures",
    modules  = [TruncatedGaussianMixtures],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    pages    = [
        "index.md",
        "QuickExample/QuickExample.md",
        "abstract_transformation.md",
    ]
)