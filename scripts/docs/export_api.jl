module DocAPIExport

using Markdown, SHA, TOML
import HighOrderMRI
include("contract_check.jl")
using .DocContractCheck

const REPO_URL = "https://github.com/BennyZhang-Codes/HighOrderMRI.jl"

function inline_code_spans(line::AbstractString)
    spans = RegexMatch[]
    for span in eachmatch(r"(?<!`)(`+)(?!`)(.*?)(?<!`)\1(?!`)", line)
        # Escaped opening backticks are literal prose characters, not delimiters.
        before = prevind(line, span.offset)
        slashes = 0
        while before >= firstindex(line) && line[before] == '\\'
            slashes += 1
            before = prevind(line, before)
        end
        isodd(slashes) && continue
        push!(spans, span)
    end
    spans
end

"Apply a prose conversion outside matching inline-code backtick runs."
function outside_inline_code(convert, line::AbstractString)
    out = IOBuffer()
    cursor = firstindex(line)
    for span in inline_code_spans(line)
        print(out, convert(SubString(line, cursor, prevind(line, span.offset))))
        print(out, span.match)
        cursor = span.offset + ncodeunits(span.match)
    end
    print(out, convert(SubString(line, cursor)))
    String(take!(out))
end

"Resolve only explicitly mapped Documenter references; never guess a destination."
function resolve_references(line::AbstractString, references::AbstractDict)
    spans = inline_code_spans(line)
    out = IOBuffer()
    cursor = firstindex(line)
    for m in eachmatch(r"\[([^\]]+)\]\(@ref(?:\s+([^\)]+))?\)", line)
        # A whole literal reference inside inline code is text. Code-formatted
        # labels inside real links, e.g. [`f`](@ref), still resolve normally.
        any(span -> span.offset <= m.offset &&
            m.offset + ncodeunits(m.match) <= span.offset + ncodeunits(span.match), spans) && continue
        label = m.captures[1]
        key = strip(something(m.captures[2], replace(label, "`" => "")))
        haskey(references, key) || error("Unresolved docstring reference: $key")
        print(out, SubString(line, cursor, prevind(line, m.offset)))
        print(out, "[$label]($(references[key]))")
        cursor = m.offset + ncodeunits(m.match)
    end
    print(out, SubString(line, cursor))
    String(take!(out))
end

"Convert the project's documented Markdown subset, preserving code fences."
function render_markdown(markdown::AbstractString, anchor::AbstractString;
                         references=Dict{String,String}(), heading_offset=2)
    out = IOBuffer()
    fence = nothing
    math_fence = false
    headings = Dict{String,Int}()
    for line in split(markdown, '\n')
        marker = match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if fence !== nothing
            if marker !== nothing && startswith(marker.captures[1], fence) &&
               isempty(strip(marker.captures[2]))
                println(out, math_fence ? "\$\$" : line)
                fence = nothing
                math_fence = false
            else
                println(out, line)
            end
            continue
        end
        if marker !== nothing
            language = strip(marker.captures[2])
            startswith(language, "@") && error("Unconverted Documenter fence: $language")
            fence = marker.captures[1]
            math_fence = language == "math"
            println(out, math_fence ? "\$\$" : line)
            continue
        end
        occursin(r"^\s*!!!", line) && error("Admonition needs an explicit VitePress conversion")
        occursin(r"<!--\s*@include:", line) && error("Docstrings must not introduce file includes")
        line = resolve_references(line, references)
        line = outside_inline_code(line) do prose
            occursin(r"\]\(@(?:ref|id)|^\s*@(?:docs|autodocs|meta)\b", prose) &&
                error("Unconverted Documenter directive: $prose")
            prose
        end
        heading = match(r"^(#{1,6})\s+(.+)$", line)
        if heading !== nothing
            title = heading.captures[2]
            slug = strip(replace(lowercase(title), r"[^\p{L}\p{N}]+" => "-"), '-')
            isempty(slug) && error("Cannot form stable heading anchor: $title")
            count = get(headings, slug, 0) + 1
            headings[slug] = count
            suffix = count == 1 ? "" : "-$count"
            level = min(length(heading.captures[1]) + heading_offset, 6)
            println(out, repeat("#", level), " ", title, " {#", anchor, "-", slug, suffix, "}")
        else
            println(out, line)
        end
    end
    fence === nothing || error("Unterminated code fence in docstring")
    String(take!(out))
end

function validate_mapping(mapping)
    get(mapping, "schema_version", 0) == 1 || error("Unsupported API mapping schema")
    entries = get(mapping, "api", [])
    isempty(entries) && error("Empty API mapping")
    for key in ("id", "anchor")
        values = [entry[key] for entry in entries]
        length(unique(values)) == length(values) || error("Duplicate API $key")
    end
    fragments = String[]
    anchors = String[entry["anchor"] for entry in entries]
    for entry in entries
        for key in ("id", "binding", "module", "page", "anchor", "status")
            haskey(entry, key) || error("Missing API mapping field $key")
        end
        occursin(r"^[a-z0-9-]+$", entry["id"]) || error("Unsafe API id")
        occursin(r"^[a-z0-9-]+$", entry["anchor"]) || error("Unsafe API anchor")
        page = entry["page"]
        (isabspath(page) || ".." in splitpath(page) || !endswith(page, ".md")) &&
            error("Invalid API page: $page")
        fragment = get(entry, "fragment", entry["id"] * ".md")
        basename(fragment) == fragment && endswith(fragment, ".md") ||
            error("Invalid fragment name: $fragment")
        push!(fragments, fragment)
        for part in get(entry, "parts", [])
            for key in ("kind", "title", "anchor")
                haskey(part, key) || error("Missing API part field $key")
            end
            part["kind"] in ("type", "method") || error("Unsupported API part kind")
            occursin(r"^[a-z0-9-]+$", part["anchor"]) || error("Unsafe API part anchor")
            occursin(r"^[A-Za-z0-9 ()-]+$", part["title"]) || error("Unsafe API part title")
            part["kind"] == "method" &&
                !(get(part, "argument_types", nothing) isa Vector{String}) &&
                error("Method part requires argument_types")
            push!(anchors, part["anchor"])
        end
    end
    bindings = [(entry["module"], entry["binding"]) for entry in entries]
    length(unique(bindings)) == length(bindings) || error("Duplicate API module/binding")
    length(unique(fragments)) == length(fragments) || error("Duplicate API fragment")
    length(unique(anchors)) == length(anchors) || error("Duplicate API part anchor")
    entries
end

function check_exports(mapping, package_module)
    disposition = Dict{String,String}()
    for entry in mapping["api"]
        entry["module"] == "HighOrderMRI" || continue
        disposition[entry["binding"]] = "api"
    end
    for entry in get(mapping, "exports", [])
        entry["module"] == "HighOrderMRI" || continue
        haskey(disposition, entry["binding"]) && error("Duplicate export disposition")
        disposition[entry["binding"]] = entry["disposition"]
    end
    visible = Set(string.(names(package_module)))
    Set(keys(disposition)) == visible || error(
        "Export mapping differs: missing=$(setdiff(visible, keys(disposition))); " *
        "obsolete=$(setdiff(keys(disposition), visible))")
    for name in visible
        defined = isdefined(package_module, Symbol(name))
        defined || disposition[name] == "known_issue" ||
            error("Undefined export without a recorded issue: $name")
    end
    length(visible)
end

function source_docs(package_module, name::String)
    symbol = Symbol(name)
    isdefined(package_module, symbol) || error("Undefined binding: $name")
    binding = Base.Docs.Binding(package_module, symbol)
    # The binding's owning module matters for explicitly re-exported functions.
    owners = unique([package_module, binding.mod])
    docs = Base.Docs.DocStr[]
    seen = Set{Tuple{String,Int,String}}()
    for owner in owners
        metadata = Base.Docs.meta(owner)
        haskey(metadata, binding) || continue
        multi = metadata[binding]
        for sig in multi.order
            doc = multi.docs[sig]
            key = (string(doc.data[:path]), Int(doc.data[:linenumber]), string(sig))
            key in seen && continue
            push!(seen, key)
            push!(docs, doc)
        end
    end
    isempty(docs) && error("No source docstring for $name")
    docs
end

"Select each source docstring exactly once by kind and positional type syntax."
function documented_parts(entry, docs)
    parts = get(entry, "parts", nothing)
    if parts === nothing
        length(docs) == 1 || error("Multiple docstrings require explicit API parts for " * entry["binding"])
        return [(doc=only(docs), anchor=entry["anchor"], title=nothing)]
    end
    selected = NamedTuple[]
    used = Set{Int}()
    for part in parts
        matches = Int[]
        for (i, doc) in enumerate(docs)
            is_type = doc.data[:typesig] === Union{}
            if part["kind"] == "type"
                is_type && push!(matches, i)
            elseif !is_type
                text = sprint(show, MIME("text/markdown"), Base.Docs.parsedoc(doc))
                methods = DocContractCheck.documented_contracts(text, string(doc.data[:path]), Int(doc.data[:linenumber]))
                expected = [t == "<untyped>" ? nothing : Meta.parse(t; raise=true)
                            for t in part["argument_types"]]
                any(method -> split(method.name, '.')[end] == entry["binding"] &&
                    length(method.positional) == length(expected) &&
                    all(DocContractCheck.same_expr(arg.type_expr, type) for (arg, type) in zip(method.positional, expected)), methods) &&
                    push!(matches, i)
            end
        end
        length(matches) == 1 || error("API part " * part["anchor"] * " must match exactly one docstring; found $(length(matches))")
        index = only(matches)
        index in used && error("A source docstring is selected by multiple API parts")
        push!(used, index)
        push!(selected, (doc=docs[index], anchor=part["anchor"], title=part["title"]))
    end
    length(used) == length(docs) || error("API parts omit source docstrings for " * entry["binding"])
    selected
end

"Keep repository-owned documentation; dependency methods require an explicit exclusion."
function repository_docs(docs, entry, code_root)
    root = realpath(code_root)
    local_docs = Base.Docs.DocStr[]
    excluded = String[]
    for doc in docs
        path = realpath(doc.data[:path])
        if ".." in splitpath(relpath(path, root))
            push!(excluded, path)
        else
            push!(local_docs, doc)
        end
    end
    if !isempty(excluded)
        get(entry, "dependency_docs", "error") == "exclude" ||
            error("Dependency docstrings require an explicit exclusion for " * entry["binding"])
    end
    isempty(local_docs) && error("No repository-owned documentation for " * entry["binding"])
    return local_docs, sort!(unique(excluded))
end

function export_api(; map_path, output, code_root, code_sha, mode="build", only=String[])
    mode in ("preview", "build") || error("Mode must be preview or build")
    occursin(r"^[0-9a-f]{40}$", code_sha) || error("code_sha must be a full Git SHA")
    mode == "build" && !isempty(only) && error("Partial export is preview-only")
    mapping = TOML.parsefile(map_path)
    entries = validate_mapping(mapping)
    selected = isempty(only) ? entries : filter(e -> e["id"] in only, entries)
    Set(e["id"] for e in selected) == Set(isempty(only) ? [e["id"] for e in entries] : only) ||
        error("Unknown --only API id")
    for entry in selected
        allowed = mode == "build" ? ("approved",) : ("candidate", "approved")
        entry["status"] in allowed || error("API $(entry["id"]) is $(entry["status"]), not approved for this mode")
    end
    if mode == "build"
        any(e -> get(e, "disposition", "pending") == "pending", get(mapping, "exports", [])) &&
            error("Additional exports still have pending dispositions")
    end
    # Loading occurs in the explicitly selected Julia project, never the site's old code.
    package_module = HighOrderMRI
    expected = realpath(joinpath(code_root, "src", "HighOrderMRI.jl"))
    realpath(pathof(package_module)) == expected || error("Loaded the wrong code checkout")
    exports_count = check_exports(mapping, package_module)
    references = Dict{String,String}()
    for entry in entries
        entry["status"] in ("candidate", "approved") || continue
        references[entry["binding"]] = "/" * chop(entry["page"]; tail=3) * "#" * entry["anchor"]
    end
    for entry in get(mapping, "references", [])
        references[entry["label"]] = entry["target"]
    end
    rendered = Dict{String,String}()
    records = Dict{String,Any}[]
    for entry in selected
        entry["module"] == "HighOrderMRI" || error("Unconfigured module $(entry["module"])")
        docs, excluded_docs = repository_docs(source_docs(package_module, entry["binding"]), entry, code_root)
        parts = documented_parts(entry, docs)
        contract_checks = Dict{String,Any}[]
        for file in unique(realpath(doc.data[:path]) for doc in docs)
            # Qualified extensions retain their exact source spelling in the
            # signature check. Types/modules/macros have no named methods here;
            # record that distinction rather than claiming a method check.
            methods = method_contracts(file)
            names = unique(m.name for m in methods if split(m.name, '.')[end] == entry["binding"])
            if isempty(names)
                value = getfield(package_module, Symbol(entry["binding"]))
                (value isa Type || value isa Module || startswith(entry["binding"], "@")) ||
                    error("No inspectable method definitions for " * entry["binding"])
                push!(contract_checks, Dict("path"=>relpath(file, code_root),
                                            "kind"=>"non-method documentation",
                                            "methods"=>0, "signatures"=>0))
                continue
            end
            check = check_doc_contracts(file; names=collect(names))
            check.ok || error("Source/docstring contract mismatch for $(entry["binding"]): " * join(check.errors, "; "))
            push!(contract_checks, Dict("path"=>relpath(file, code_root),
                                        "methods"=>check.checked_methods,
                                        "signatures"=>check.documented_signatures))
        end
        io = IOBuffer()
        println(io, "<!-- Generated from source docstrings; do not edit this fragment. -->")
        println(io, "<a id=\"", entry["anchor"], "\"></a>\n")
        source_records = Dict{String,Any}[]
        for part in parts
            doc = part.doc
            file = realpath(doc.data[:path])
            relative = relpath(file, code_root)
            ".." in splitpath(relative) && error("Docstring comes from outside the selected source: $file")
            line = doc.data[:linenumber]
            text = sprint(show, MIME("text/markdown"), Base.Docs.parsedoc(doc))
            part.title === nothing || println(io, "### ", part.title, " {#", part.anchor, "}\n")
            println(io, render_markdown(text, part.anchor; references,
                                        heading_offset=part.title === nothing ? 2 : 3))
            label = mode == "preview" ? "Source baseline (candidate docstring is uncommitted)" : "Source"
            # Local prose changes shift line numbers relative to code_sha. A
            # preview links to the baseline file, never an invented commit line.
            location = mode == "build" ? "#L$line" : ""
            println(io, "[", label, ": `", replace(relative, '\\'=>'/'), "`](", REPO_URL,
                    "/blob/", code_sha, "/", replace(relative, '\\'=>'/'), location, ")\n")
            push!(source_records, Dict("path"=>relative, "line"=>line, "anchor"=>part.anchor,
                                      "sha256"=>bytes2hex(sha256(read(file)))))
        end
        fragment = get(entry, "fragment", entry["id"] * ".md")
        rendered[fragment] = String(take!(io))
        push!(records, Dict("id"=>entry["id"], "fragment"=>fragment,
                            "docstring_count"=>length(docs), "sources"=>source_records,
                            "excluded_dependency_docs"=>excluded_docs,
                            "contract_checks"=>contract_checks,
                            "sha256"=>bytes2hex(sha256(rendered[fragment]))))
    end
    # Validate all selected inputs before replacing any generated content.
    mkpath(output)
    for (name, text) in rendered
        write(joinpath(output, name), text)
    end
    index = Dict("schema_version"=>1, "mode"=>mode, "partial"=>!isempty(only),
                 "code_sha"=>code_sha, "julia_version"=>string(VERSION),
                 "export_count"=>exports_count, "mapped_api_count"=>length(entries),
                 "api"=>records)
    open(joinpath(output, "index.toml"), "w") do io
        TOML.print(io, index; sorted=true)
    end
    println("Generated ", length(records), "/", length(entries), " API bindings; mode=", mode)
    index
end

function main(args)
    options = Dict{String,String}()
    iseven(length(args)) || error("Expected --option value pairs")
    for i in 1:2:length(args)
        args[i] in ("--map", "--output", "--code-root", "--code-sha", "--mode", "--only") ||
            error("Unknown option $(args[i])")
        haskey(options, args[i]) && error("Duplicate option $(args[i])")
        options[args[i]] = args[i+1]
    end
    export_api(; map_path=options["--map"], output=options["--output"],
               code_root=options["--code-root"], code_sha=options["--code-sha"],
               mode=get(options, "--mode", "build"),
               only=haskey(options, "--only") ? split(options["--only"], ',') : String[])
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    DocAPIExport.main(ARGS)
end
