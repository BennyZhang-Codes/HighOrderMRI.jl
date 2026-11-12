function plan_main(args)
    options = command_options(args, ["--output"])
    output = required_option(options, "--output")
    event, ref = get(ENV, "GITHUB_EVENT_NAME", ""), get(ENV, "GITHUB_REF", "")
    payload = read_json(ENV["GITHUB_EVENT_PATH"])
    preview = event == "pull_request"
    preview || validate_context(get(ENV, "GITHUB_REPOSITORY", ""), event, ref, payload)
    release_tag = get(ENV, "RELEASE_TAG", "")
    require(isempty(release_tag) || event == "workflow_dispatch" && ref in last.(SOURCE_REFS),
            "Manual release backfills must run from a code branch")
    source = git(pwd(), "rev-parse", "HEAD")
    saved = git_status(pwd(), "show", "refs/remotes/origin/gh-pages:versions.json")
    archive = saved.code == 0 ? parse_json(saved.output) : Dict()
    plan = resolve_versions(pwd(), published_catalog(), archive, source, event, ref;
                            base=get(ENV, "GITHUB_BASE_REF", nothing), release_tag)
    write_json(output, plan)
    workflow_output(["source_sha" => source, "publish" => !preview, "preview" => preview])
    open(ENV["GITHUB_ENV"], "a") do environment
        println(environment, "DOCS_BUILD_PLAN=", abspath(output))
    end
    println(json_text(plan))
    return nothing
end

function verify_main(args)
    options = command_options(args, ["--docs", "--plan"], ["--allow-unpublished"])
    docs = abspath(get(options, "--docs", "docs"))
    config = read_json(get(options, "--plan", joinpath(docs, "versions.json")))
    verify_releases(dirname(docs), config; allow_unpublished=haskey(options, "--allow-unpublished"))
    return nothing
end

function archive_main(args)
    options = command_options(args, ["--artifact-tar", "--site", "--source-sha"])
    artifact, site, source = (required_option(options, key) for key in ("--artifact-tar", "--site", "--source-sha"))
    payload = read_json(ENV["GITHUB_EVENT_PATH"])
    validate_context(get(ENV, "GITHUB_REPOSITORY", ""), get(ENV, "GITHUB_EVENT_NAME", ""), get(ENV, "GITHUB_REF", ""), payload)
    require(0 < filesize(artifact) <= MAX_BYTES, "Artifact exceeds the supported size")
    extract_tar(read(artifact), site)
    plan = read_json(joinpath(site, "publication-plan.json"))
    require(get(plan, "source_sha", nothing) == source, "Build plan does not match the checked source")
    targets = Dict(row["version"] => row for row in plan["targets"])
    code = Dict("schema_version" => 1, "code_sha" => targets["dev"]["code_sha"], "main_sha" => targets["main"]["code_sha"],
                "dev_docs_sha" => targets["dev"]["docs_sha"], "main_docs_sha" => targets["main"]["docs_sha"])
    releases = Dict("schema_version" => 1, "stable" => plan["stable"], "releases" => plan["releases"])
    manifest = validate_site(site, code, releases)
    result = publish(pwd(), abspath(site), source, manifest; source_ref=ENV["GITHUB_REF"])
    values = Pair{String,Any}["publish" => result["publish"]]
    result["publish"] && push!(values, "sha" => result["sha"])
    workflow_output(values)
    println(json_text(result))
    return nothing
end

function live_main(args)
    options = command_options(args, ["--pages-url", "--manifest", "--config", "--attempts", "--interval", "--max-wait"], ["--allow-preview"])
    root = site_root(get(options, "--pages-url", get(ENV, "PAGES_URL", "")))
    if haskey(options, "--manifest")
        manifest = read_json(options["--manifest"])
        check(get(manifest, "schema_version", nothing) === 1 && get(manifest, "preview", nothing) === false,
              "expected a checked formal artifact manifest")
        stable, versions = manifest["stable"], manifest["versions"]
    else
        stable, versions = expected_versions(get(options, "--config", joinpath(@__DIR__, "..", "code-version.json")))
    end
    attempts = parse(Int, get(options, "--attempts", "12"))
    interval = parse(Float64, get(options, "--interval", "10"))
    max_wait = parse(Float64, get(options, "--max-wait", "120"))
    for row in versions
        verify_site(root, row["code_sha"]; attempts, interval, max_wait, prefix=row["version"]) || return false
    end
    response = fetch_response(root * "versions.json?check=" * string(uuid4()), "GET", 10)
    check_version_index(response, stable, versions; allow_preview=haskey(options, "--allow-preview"))
    println("Published version index verified: stable $stable, $(length(versions)) sites")
    return true
end

function command_main(args)
    require(!isempty(args), "Expected plan, verify, archive, check-site, check-live, preview or dark-svg")
    command, rest = first(args), args[2:end]
    if command == "plan"
        plan_main(rest)
    elseif command == "verify"
        verify_main(rest)
    elseif command == "archive"
        archive_main(rest)
    elseif command == "check-live"
        return live_main(rest)
    elseif command == "check-site"
        require(length(rest) == 4, "Expected DOCS INPUT_HTML BASE VERSION")
        report = check_site(rest[1], rest[2]; base=rest[3], version=rest[4])
        println(json_text(report))
        return isempty(report["errors"])
    elseif command == "preview"
        options = command_options(rest, ["--port", "--directory"])
        serve_preview(get(options, "--directory", normpath(joinpath(@__DIR__, "..", ".pages"))), parse(Int, get(options, "--port", "8775")))
    elseif command == "dark-svg"
        options = command_options(rest, ["--source", "--output"])
        figures = normpath(joinpath(@__DIR__, "..", "src", "public", "figures", "adaptive-shared-basis"))
        build_dark_svg(get(options, "--source", joinpath(figures, "Fig1_adaptive_shared_basis.svg")),
                       get(options, "--output", joinpath(figures, "Fig1_adaptive_shared_basis_dark.svg")))
    else
        throw(ArgumentError("Unknown publication command $command"))
    end
    return true
end
