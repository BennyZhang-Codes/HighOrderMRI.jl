using Test, Markdown, SHA, TOML
include(joinpath(@__DIR__, "export_api.jl"))
using .DocAPIExport

const EXPORT_CODE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const EXPORT_MAP = get(ENV, "DOCS_API_MAP", "")
isfile(EXPORT_MAP) || error("Set DOCS_API_MAP to the documentation checkout api-map.toml")
const EXPORT_COMMIT = "d141e7dc48e8d18380050aaa970ee4ae364d916d"
const API = DocAPIExport
println("EXPORT_TEST_JULIA=", VERSION)
println("EXPORT_TEST_PACKAGE=", pathof(API.HighOrderMRI))
println("EXPORT_TEST_CUDA_VISIBLE_DEVICES=", repr(get(ENV, "CUDA_VISIBLE_DEVICES", "unset")))
println("EXPORT_TEST_DEPOT=", DEPOT_PATH)

function api_entry(id, binding; module_name="HighOrderMRI")
    Dict{String,Any}("id"=>id, "binding"=>binding, "module"=>module_name,
        "page"=>"reference/example.md", "anchor"=>"api-$id", "status"=>"candidate",
        "fragment"=>"$id.md")
end

function write_mapping(path, mapping)
    open(path, "w") do io
        TOML.print(io, mapping; sorted=true)
    end
    return path
end

# This fixture models a re-export with documentation attached in both modules.
# It neither changes HighOrderMRI methods nor executes an MRI reconstruction.
module DocsOwnerFixture
export f
"""    f(x::Int)

Owner integer method.
"""
f(x::Int) = x
"""    f(x::String)

Owner string method.
"""
f(x::String) = x
end
module DocsFacadeFixture
import ..DocsOwnerFixture: f
export f
"""    f(x::Float64)

Facade floating method.
"""
f(x::Float64) = x
end

module ExportsFixture
export defined, unimplemented_api
const defined = 1
end

@testset "API export behavior" begin
    @testset "Markdown links, mathematics and stable headings" begin
        refs = Dict("f"=>"/reference/f#api-f", "Theory"=>"/theory/encoding-model")
        text = "# Arguments\nSee [`f`](@ref) and [the model](@ref Theory).\n\n# Arguments\nDone."
        rendered = API.render_markdown(text, "api-f"; references=refs)
        @test occursin("[`f`](/reference/f#api-f)", rendered)
        @test occursin("[the model](/theory/encoding-model)", rendered)
        @test occursin("### Arguments {#api-f-arguments}", rendered)
        @test occursin("### Arguments {#api-f-arguments-2}", rendered)
        @test API.render_markdown(text, "api-f"; references=refs) == rendered
        @test occursin("{#api-f-参数}", API.render_markdown("# 参数\n说明", "api-f"))
        # Test both Markdown.jl's actual serialization and the accepted raw fence form.
        markdown_math = Markdown.parse(raw"Inline ``x^2`` and a formula.")
        serialized = sprint(show, MIME("text/markdown"), markdown_math)
        @test occursin(raw"$x^2$", API.render_markdown(serialized, "api-f"))
        math = API.render_markdown("```math\nA^H W^H W A\n```", "api-f")
        @test occursin("\$\$\nA^H W^H W A\n\$\$", math)
        @test !occursin("```math", math)
        display_math = sprint(show, MIME("text/markdown"), Markdown.parse("```math\nx+y\n```"))
        @test occursin("\$\$\nx+y\n\$\$", API.render_markdown(display_math, "api-f"))
        ordinary = "[paper](https://doi.org/10.1002/mrm.29460) and `plain_code`"
        @test strip(API.render_markdown(ordinary, "api-f")) == ordinary
        literal_ref = "Write `[x](@ref Unknown)` when discussing reference syntax."
        @test strip(API.render_markdown(literal_ref, "api-f")) == literal_ref
        mixed = "Literal `[x](@ref Unknown)` and real [`f`](@ref)."
        @test strip(API.render_markdown(mixed, "api-f"; references=refs)) ==
              "Literal `[x](@ref Unknown)` and real [`f`](/reference/f#api-f)."
        double_ticks = "Literal ``a ` b [x](@ref Unknown)`` remains code."
        @test strip(API.render_markdown(double_ticks, "api-f")) == double_ticks
    end

    @testset "Code fence boundaries and rejected unsupported syntax" begin
        # A four-backtick fence can contain triples, Documenter syntax and math.
        literal = "````julia\n[x](@ref Unknown)\n```math\n!!! note\n<!-- @include: hidden.md -->\n```\n````"
        @test strip(API.render_markdown(literal, "api-f")) == literal
        tilde = "~~~julia\n@docs Hidden\n[x](@ref Unknown)\n~~~"
        @test strip(API.render_markdown(tilde, "api-f")) == tilde
        for unsupported in (
            "[missing](@ref Unknown)", "# [Title](@id custom)",
            "```@docs\nf\n```", "```@mystery\nf\n```", "```@eval\nx\n```",
            "@autodocs Module=Foo", "@meta CurrentModule=Foo", "!!! note\n    Details",
            "<!-- @include: hidden.md -->", "```julia\nunterminated", "# !!!",
        )
            @test_throws ErrorException API.render_markdown(unsupported, "api-f")
        end
        # A closed code example must not suppress validation of following prose.
        @test_throws ErrorException API.render_markdown("```julia\nx\n```\n[x](@ref Missing)", "api-f")
    end

    @testset "Mapping identity, path boundaries and export dispositions" begin
        base = Dict{String,Any}("schema_version"=>1, "api"=>[api_entry("f", "f"), api_entry("g", "g")])
        @test length(API.validate_mapping(base)) == 2
        for key in ("id", "anchor", "fragment", "binding")
            duplicate = deepcopy(base)
            duplicate["api"][2][key] = duplicate["api"][1][key]
            @test_throws ErrorException API.validate_mapping(duplicate)
        end
        # A same-named binding in a different module is a different mapping identity.
        distinct_modules = deepcopy(base)
        distinct_modules["api"][2]["binding"] = "f"
        distinct_modules["api"][2]["module"] = "OtherModule"
        @test length(API.validate_mapping(distinct_modules)) == 2
        for (key, value) in (("page","../outside.md"), ("page","/absolute.md"),
                             ("fragment","../outside.md"), ("id","BAD ID"),
                             ("anchor","bad anchor"), ("page","reference/f.html"))
            unsafe = deepcopy(base)
            unsafe["api"][1][key] = value
            @test_throws ErrorException API.validate_mapping(unsafe)
        end
        @test_throws ErrorException API.validate_mapping(Dict("schema_version"=>2, "api"=>base["api"]))
        @test_throws ErrorException API.validate_mapping(Dict("schema_version"=>1, "api"=>[]))
        exports = Dict{String,Any}("api"=>[api_entry("module", "ExportsFixture"), api_entry("defined", "defined")],
            "exports"=>[Dict("module"=>"HighOrderMRI", "binding"=>"unimplemented_api", "disposition"=>"known_issue")])
        @test API.check_exports(exports, ExportsFixture) == 3
        unresolved = deepcopy(exports)
        unresolved["exports"][1]["disposition"] = "approved"
        @test_throws ErrorException API.check_exports(unresolved, ExportsFixture)
        duplicated = deepcopy(exports)
        push!(duplicated["exports"], Dict("module"=>"HighOrderMRI", "binding"=>"defined", "disposition"=>"reexport"))
        @test_throws ErrorException API.check_exports(duplicated, ExportsFixture)
        @test API.check_exports(TOML.parsefile(EXPORT_MAP), API.HighOrderMRI) == length(names(API.HighOrderMRI))
    end

    @testset "Source metadata includes both re-export owners" begin
        docs = API.source_docs(DocsFacadeFixture, "f")
        text = join(sprint(show, MIME("text/markdown"), Base.Docs.parsedoc(doc)) for doc in docs)
        @test length(docs) == 3
        @test occursin("Owner integer method", text)
        @test occursin("Owner string method", text)
        @test occursin("Facade floating method", text)
        @test [(doc.data[:path], doc.data[:linenumber], doc.data[:typesig]) for doc in docs] ==
              [(doc.data[:path], doc.data[:linenumber], doc.data[:typesig]) for doc in API.source_docs(DocsFacadeFixture, "f")]
        @test_throws ErrorException API.source_docs(DocsFacadeFixture, "not_defined")
    end

    @testset "Real FindDelay preview and failure before output replacement" begin
        mktempdir() do temporary
            output = joinpath(temporary, "generated")
            common = (; map_path=EXPORT_MAP, output, code_root=EXPORT_CODE_ROOT, code_sha=EXPORT_COMMIT)
            index = API.export_api(; common..., mode="preview", only=["find-delay"])
            fragment_path = joinpath(output, "find-delay.md")
            text = read(fragment_path, String)
            index_path = joinpath(output, "index.toml")
            @test index["partial"]
            @test index["mode"] == "preview"
            @test length(index["api"]) == 1
            record = only(index["api"])
            @test record["id"] == "find-delay"
            @test record["sha256"] == bytes2hex(sha256(read(fragment_path)))
            @test only(record["contract_checks"])["methods"] == 2
            @test only(record["contract_checks"])["signatures"] == 2
            @test only(record["sources"])["sha256"] == bytes2hex(sha256(read(joinpath(EXPORT_CODE_ROOT, "src", "Synchronization", "FindDelay.jl"))))
            @test occursin("<a id=\"api-find-delay\"></a>", text)
            @test occursin("candidate docstring is uncommitted", text)
            @test occursin("/blob/$EXPORT_COMMIT/src/Synchronization/FindDelay.jl)", text)
            @test occursin("same time unit as", text)
            @test occursin("10.1002/mrm.29460", text)
            @test TOML.parsefile(index_path)["api"] == index["api"]
            saved_fragment, saved_index = read(fragment_path), read(index_path)
            # Determinism covers doc metadata order, generated headings and output hashes.
            API.export_api(; common..., mode="preview", only=["find-delay"])
            @test read(fragment_path) == saved_fragment
            @test read(index_path) == saved_index
            @test_throws ErrorException API.export_api(; common..., mode="build", only=["find-delay"])
            pendingmap = TOML.parsefile(EXPORT_MAP)
            only(filter(e -> e["id"] == "grid", pendingmap["api"]))["status"] = "pending"
            pendingpath = write_mapping(joinpath(temporary, "pending.toml"), pendingmap)
            @test_throws ErrorException API.export_api(; common..., map_path=pendingpath, mode="build")
            @test_throws ErrorException API.export_api(; common..., mode="preview", only=["does-not-exist"])
            @test_throws ErrorException API.export_api(; common..., map_path=pendingpath, mode="preview", only=["grid"])
            wrong_checkout = joinpath(temporary, "wrong-checkout")
            mkpath(joinpath(wrong_checkout, "src"))
            write(joinpath(wrong_checkout, "src", "HighOrderMRI.jl"), "# wrong code checkout")
            @test_throws ErrorException API.export_api(; common..., mode="preview", code_root=wrong_checkout, only=["find-delay"])
            badmap = TOML.parsefile(EXPORT_MAP)
            copy_entry = deepcopy(only(filter(e -> e["id"] == "find-delay", badmap["api"])))
            copy_entry["id"] = "second-delay"
            copy_entry["anchor"] = "api-second-delay"
            copy_entry["fragment"] = "second-delay.md"
            push!(badmap["api"], copy_entry)
            badmap_path = write_mapping(joinpath(temporary, "duplicate.toml"), badmap)
            @test_throws ErrorException API.export_api(; common..., map_path=badmap_path, mode="preview", only=["find-delay"])
            @test read(fragment_path) == saved_fragment
            @test read(index_path) == saved_index
        end
    end
end
@testset "Explicit LowRank type and constructor selection" begin
    mapping = TOML.parsefile(EXPORT_MAP)
    entry = only(filter(e -> e["id"] == "highorderlowrankop", mapping["api"]))
    docs = API.source_docs(API.HighOrderMRI, "HighOrderLowRankOp")
    parts = API.documented_parts(entry, docs)
    @test length(parts) == 3
    @test parts[1].doc.data[:typesig] === Union{}
    @test [p.anchor for p in parts] == [p["anchor"] for p in entry["parts"]]
    @test [p.doc for p in API.documented_parts(entry, reverse(docs))] == [p.doc for p in parts]
    omitted = deepcopy(entry)
    pop!(omitted["parts"])
    @test_throws ErrorException API.documented_parts(omitted, docs)
    duplicated = deepcopy(entry)
    push!(duplicated["parts"], deepcopy(duplicated["parts"][1]))
    @test_throws ErrorException API.documented_parts(duplicated, docs)
    wrong = deepcopy(entry)
    wrong["parts"][2]["argument_types"][2] = "AbstractArray{T,7}"
    @test_throws ErrorException API.documented_parts(wrong, docs)
    implicit = deepcopy(entry)
    delete!(implicit, "parts")
    @test_throws ErrorException API.documented_parts(implicit, docs)
    conflicting = deepcopy(mapping)
    lowrank = only(filter(e -> e["id"] == "highorderlowrankop", conflicting["api"]))
    lowrank["parts"][1]["anchor"] = "api-find-delay"
    @test_throws ErrorException API.validate_mapping(conflicting)
    @test occursin("#### Arguments", API.render_markdown("# Arguments", "api-f"; heading_offset=3))
    mktempdir() do output
        common = (; map_path=EXPORT_MAP, output, code_root=EXPORT_CODE_ROOT,
                   code_sha=EXPORT_COMMIT, mode="preview", only=["find-delay", "highorderlowrankop"])
        index = API.export_api(; common...)
        record = only(filter(r -> r["id"] == "highorderlowrankop", index["api"]))
        @test record["docstring_count"] == 3
        @test only(record["contract_checks"])["methods"] == 2
        @test only(record["contract_checks"])["signatures"] == 2
        text = read(joinpath(output, "highorderlowrankop.md"), String)
        for part in entry["parts"]
            @test count("{#" * part["anchor"] * "}", text) == 1
        end
        @test occursin("[Low-rank theory](/theory/low-rank)", text)
        @test !occursin("(@ref", text)
        @test !occursin(r"/blob/[^)]+#L", text)
        saved = Dict(f=>read(joinpath(output,f)) for f in readdir(output))
        API.export_api(; common...)
        @test all(read(joinpath(output,f)) == data for (f,data) in saved)
    end
end

module UntypedDocsFixture
"""    g(x)

Unannotated argument.
"""
g(x) = x
"""    g(x::Int)

Integer argument.
"""
g(x::Int) = x
end

@testset "Unannotated parts and explicit dependency exclusion" begin
    entry = api_entry("g", "g")
    entry["parts"] = [
        Dict("kind"=>"method", "title"=>"General", "anchor"=>"g-general", "argument_types"=>["<untyped>"]),
        Dict("kind"=>"method", "title"=>"Integer", "anchor"=>"g-integer", "argument_types"=>["Int"]),
    ]
    docs = API.source_docs(UntypedDocsFixture, "g")
    parts = API.documented_parts(entry, docs)
    @test length(parts) == 2
    @test occursin("Unannotated", sprint(show, MIME("text/markdown"), Base.Docs.parsedoc(parts[1].doc)))
    wrong = deepcopy(entry)
    wrong["parts"][1]["argument_types"] = ["Any"]
    @test_throws ErrorException API.documented_parts(wrong, docs)
    @test length(first(API.repository_docs(docs, entry, EXPORT_CODE_ROOT))) == 2
    @test isempty(last(API.repository_docs(docs, entry, EXPORT_CODE_ROOT)))
    mktempdir() do outside
        @test_throws ErrorException API.repository_docs(docs, entry, outside)
        excluded = deepcopy(entry)
        excluded["dependency_docs"] = "exclude"
        @test_throws ErrorException API.repository_docs(docs, excluded, outside)
    end
end

println("AUDIT_EXPORT_TESTS_COMPLETED")
