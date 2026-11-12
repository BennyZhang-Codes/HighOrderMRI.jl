function validate_release_metadata(metadata, version)
    require(get(metadata, "tag_name", nothing) == version, "Release tag does not match $version")
    published_at = get(metadata, "published_at", nothing)
    require(get(metadata, "draft", nothing) === false && get(metadata, "prerelease", nothing) === false &&
            published_at isa AbstractString && !isempty(published_at),
            "Stable documentation requires published, non-prerelease $version")
    return nothing
end

formal_releases(published) = filter(published) do release
    published_at = get(release, "published_at", nothing)
    is_version(get(release, "tag_name", nothing)) && get(release, "draft", nothing) === false &&
    get(release, "prerelease", nothing) === false && published_at isa AbstractString && !isempty(published_at)
end

function validate_release_catalog(config, published)
    expected = Set(release["tag_name"] for release in formal_releases(published))
    require(issubset(expected, Set(row["version"] for row in config["releases"])),
            "Every published formal release must remain in the version plan")
    return nothing
end

function validate_context(repository, event, ref, payload=Dict())
    require(repository == REPOSITORY, "Unexpected documentation repository")
    if event in ("push", "workflow_dispatch")
        require(ref in last.(SOURCE_REFS), "Only code branches may publish documentation")
    elseif event == "release"
        release = get(payload, "release", Dict())
        require(get(payload, "action", nothing) == "published" && is_version(get(release, "tag_name", nothing)),
                "Only published formal releases may publish documentation")
        require(ref == "refs/tags/" * release["tag_name"], "Release ref mismatch")
        validate_release_metadata(release, release["tag_name"])
    else
        throw(ArgumentError("Pull requests and unsupported events cannot publish documentation"))
    end
    return nothing
end

function documentation_source(repo, code; legacy=LEGACY_DOCS_SOURCE)
    # Pre-migration code uses the saved documentation ancestor, never a branch.
    return git_status(repo, "cat-file", "-e", "$code:docs/src/index.md").code == 0 ? code : legacy
end

function published_catalog()
    command = Cmd(["gh", "api", "--paginate", "--slurp", "repos/$REPOSITORY/releases?per_page=100"])
    pages = parse_json(read(command, String))
    return reduce(vcat, pages; init=Any[])
end

function resolve_versions(repo, published, archive, source, event, ref;
                          base=nothing, release_tag="", legacy=LEGACY_DOCS_SOURCE)
    require(is_sha(source), "Invalid build source")
    saved = read_json(joinpath(repo, "docs", "versions.json"))
    rows = Dict(row["version"] => copy(row) for row in saved["releases"])
    for row in get(archive, "versions", [])
        if is_version(row["version"])
            frozen = Dict(key => row[key] for key in ("version", "code_sha", "docs_sha"))
            require(!haskey(rows, row["version"]) || rows[row["version"]] == frozen, "Frozen release identity mismatch")
            rows[row["version"]] = frozen
        end
    end
    for release in formal_releases(published)
        version = release["tag_name"]
        code = git(repo, "rev-parse", "--verify", "refs/tags/$version^{commit}")
        if haskey(rows, version)
            require(rows[version]["code_sha"] == code, "Published release tag cannot move")
        else
            require(documentation_source(repo, code; legacy) == code, "Release $version has no in-tree documentation")
            rows[version] = Dict("version" => version, "code_sha" => code, "docs_sha" => code)
        end
    end
    if !isempty(release_tag)
        require(is_version(release_tag), "Expected a formal release tag")
        require(release_tag in [release["tag_name"] for release in formal_releases(published)],
                "Backfill requires an existing published formal release")
    end
    versions = sort!(collect(keys(rows)); by=version_number, rev=true)
    require(!isempty(versions), "At least one published formal version is required")
    targets = Any[]
    for (version, branch_ref) in SOURCE_REFS
        branch = chop(branch_ref; head=length("refs/heads/"), tail=0)
        code = git(repo, "rev-parse", "refs/remotes/origin/$branch")
        if event == "pull_request" && base == branch || event in ("push", "workflow_dispatch") && ref == branch_ref
            code = source
        end
        push!(targets, Dict("version" => version, "code_sha" => code, "docs_sha" => documentation_source(repo, code; legacy)))
    end
    releases = [rows[version] for version in versions]
    stable = first(releases)
    append!(targets, releases)
    push!(targets, merge(stable, Dict("version" => "stable", "release" => stable["version"])))
    return Dict("schema_version" => 1, "source_sha" => source, "stable" => stable["version"],
                "releases" => releases, "targets" => targets)
end

function verify_releases(repo, config; allow_unpublished=false, published=nothing)
    allow_unpublished || validate_release_catalog(config, published === nothing ? published_catalog() : published)
    for release in config["releases"]
        version, code, source = (release[key] for key in ("version", "code_sha", "docs_sha"))
        require(is_version(version) && is_sha(code) && is_sha(source), "Invalid release identity")
        project = TOML.parse(git(repo, "show", "$code:Project.toml"))
        require(get(project, "version", nothing) == version[2:end], "Project.toml version does not match $version")
        git(repo, "cat-file", "-e", "$source:docs/api-map.toml")
        git(repo, "cat-file", "-e", "$source:docs/src/index.md")
        git(repo, "merge-base", "--is-ancestor", source, "HEAD")
        git(repo, "merge-base", "--is-ancestor", code, "origin/main")
        tag = git_status(repo, "rev-parse", "--verify", "refs/tags/$version^{commit}")
        tag.code == 0 && require(tag.output == code, "Tag $version points to a different commit")
        if !allow_unpublished
            require(tag.code == 0, "Missing formal tag $version")
            metadata = parse_json(read(`gh api repos/$REPOSITORY/releases/tags/$version`, String))
            validate_release_metadata(metadata, version)
        end
        println(allow_unpublished ? "Candidate " : "Published release verified: ", version, ": code ", code, ", documentation ", source)
    end
    return nothing
end
