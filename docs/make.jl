using Documenter
using HighOrderMRI

DocMeta.setdocmeta!(
    HighOrderMRI,
    :DocTestSetup,
    :(using HighOrderMRI);
    recursive=true,
)

const DEVBRANCH = get(ENV, "DOCUMENTER_DEVBRANCH", "main")

makedocs(
    modules=[HighOrderMRI],
    authors="HighOrderMRI.jl contributors",
    sitename="HighOrderMRI.jl",
    checkdocs=:exports,
    format=Documenter.HTML(
        canonical="https://bennyzhang-codes.github.io/HighOrderMRI.jl/stable/",
        edit_link=DEVBRANCH,
        prettyurls=get(ENV, "CI", "false") == "true",
        repolink="https://github.com/BennyZhang-Codes/HighOrderMRI.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Theory" => [
            "Expanded encoding model" => "theory/encoding-model.md",
            "Low-rank shared subspace" => "theory/low-rank.md",
        ],
        "User guide" => [
            "Choose an operator" => "guide/operators.md",
            "Reconstruction workflow" => "guide/reconstruction.md",
            "Multi-GPU execution" => "guide/multi-gpu.md",
            "Field preprocessing and synchronization" => "guide/field-preprocessing.md",
            "Reconstruction protocol" => "guide/reconstruction-protocol.md",
        ],
        "API reference" => "api.md",
        "References and source notes" => "references.md",
    ],
)

deploydocs(
    repo="github.com/BennyZhang-Codes/HighOrderMRI.jl.git",
    devbranch=DEVBRANCH,
)
