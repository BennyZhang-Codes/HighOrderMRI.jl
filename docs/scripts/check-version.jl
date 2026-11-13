# Keep each code-side test script in its own module while sharing package loading.
# Plotting retains the existing child-process boundary in check_examples.jl.
length(ARGS) == 2 || error("Expected code root and website input root")
const CHECK_CODE_ROOT = realpath(ARGS[1])
const CHECK_TOOLS_ROOT = ispath(joinpath(CHECK_CODE_ROOT, "docs", "scripts", "export_api.jl")) ?
    joinpath(CHECK_CODE_ROOT, "docs", "scripts") : joinpath(CHECK_CODE_ROOT, "scripts", "docs")
for script in ("export_api.jl", "test_contract_check.jl", "test_export_api.jl", "check_examples.jl")
    isfile(joinpath(CHECK_TOOLS_ROOT, script)) || error("Missing documentation tool in selected code checkout: $script")
end
ENV["DOCS_API_MAP"] = joinpath(realpath(ARGS[2]), "api-map.toml")
ENV["DOCS_SITE_PATH"] = realpath(ARGS[2])
empty!(ARGS)

module ContractChecks
include(joinpath(Main.CHECK_TOOLS_ROOT, "test_contract_check.jl"))
end

module ExportChecks
include(joinpath(Main.CHECK_TOOLS_ROOT, "test_export_api.jl"))
end

module ExampleChecks
include(joinpath(Main.CHECK_TOOLS_ROOT, "check_examples.jl"))
end
