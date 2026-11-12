function extract_tar(raw, output; code_sha=nothing)
    require(code_sha === nothing || is_sha(code_sha), "Invalid configured code SHA")
    require(0 < length(raw) <= MAX_BYTES, "TAR exceeds the supported size")
    members = Dict{String,Tar.Header}()
    folded = Set{String}()
    total = 0
    for header in Tar.list(IOBuffer(raw))
        name = header.path
        require(!startswith(name, "/") && !occursin('\\', name) && !occursin(':', name) &&
                !any(character -> Int(character) < 32, name) && !(".." in split(name, '/')), "Unsafe TAR path")
        path = join(filter(part -> !isempty(part) && part != ".", split(name, '/')), "/")
        isempty(path) && (path = ".")
        require(header.type in (:file, :directory), "TAR links and special files are not allowed")
        require(path != "." || header.type == :directory, "Invalid TAR root")
        key = Unicode.normalize(path; casefold=true)
        require(!(key in folded), "TAR path collision")
        push!(folded, key)
        members[path] = header
        require(0 <= header.size <= MAX_BYTES - total, "Extracted site exceeds the supported size")
        total += header.size
    end
    for path in keys(members)
        parent = dirname(path)
        while !isempty(parent) && parent != "."
            require(!haskey(members, parent) || members[parent].type == :directory, "TAR file/directory collision")
            parent = dirname(parent)
        end
    end
    required = ("index.html", "dev/index.html", "dev/reference/highorderlowrankop.html")
    require(all(path -> haskey(members, path) && members[path].type == :file, required), "Required site pages are missing")
    if code_sha !== nothing
        mktempdir() do temporary
            Tar.extract(header -> header.type == :directory || replace(normpath(header.path), '\\' => '/') == required[end],
                        IOBuffer(raw), temporary; set_permissions=false)
            require(occursin(code_sha, read(joinpath(temporary, required[end]), String)), "Artifact does not document the configured code SHA")
        end
    end
    require(!islink(output) && (!ispath(output) || isdir(output) && isempty(readdir(output))),
            "Output must be an empty directory, not a symlink")
    mkpath(output)
    Tar.extract(IOBuffer(raw), output; set_permissions=false)
    for (directory, directories, files) in walkdir(output)
        chmod(directory, 0o755)
        foreach(name -> chmod(joinpath(directory, name), 0o755), directories)
        foreach(name -> chmod(joinpath(directory, name), 0o644), files)
    end
    return count(header -> header.type == :file, values(members))
end

function validate_site(site, code, releases)
    manifest = read_json(joinpath(site, "versions.json"))
    require(get(manifest, "schema_version", nothing) === 1 && get(manifest, "preview", nothing) === false,
            "Only a complete formal site may be archived")
    require(get(code, "schema_version", nothing) === 1 && get(releases, "schema_version", nothing) === 1,
            "Invalid source configuration")
    expected = Dict{String,Tuple{Any,Any}}(row["version"] => (row["code_sha"], row["docs_sha"]) for row in releases["releases"])
    require(length(expected) == length(releases["releases"]) && haskey(expected, releases["stable"]), "Invalid release registry")
    require(all(is_version, keys(expected)) && all(pair -> all(is_sha, pair), values(expected)), "Invalid release identity")
    require(is_sha(code["code_sha"]), "Invalid dev code SHA")
    expected["dev"] = (code["code_sha"], get(code, "dev_docs_sha", nothing))
    expected["stable"] = expected[releases["stable"]]
    require(!haskey(code, "main_sha"), "The main documentation channel is not published")
    rows = get(manifest, "versions", [])
    require(length(rows) == length(expected) && Set(get(row, "version", nothing) for row in rows) == Set(keys(expected)),
            "The archive must contain every configured version exactly once")
    require(get(manifest, "stable", nothing) == releases["stable"], "Stable release mismatch")
    for row in rows
        version = row["version"]
        require((get(row, "code_sha", nothing), get(row, "docs_sha", nothing)) == expected[version] &&
                get(row, "preview", nothing) === false, "Version source or preview mismatch")
        require(isfile(joinpath(site, version, "index.html")), "Version index is missing")
        require(read_json(joinpath(site, version, "version.json")) == row, "Version metadata does not match the complete-site manifest")
    end
    require(isfile(joinpath(site, "index.html")), "Site index is missing")
    require(!ispath(joinpath(site, "main")), "The main documentation directory is not published")
    for (directory, directories, files) in walkdir(site; follow_symlinks=false)
        require(!any(name -> islink(joinpath(directory, name)), vcat(directories, files)), "Archive links are not allowed")
    end
    require(!any(name -> ispath(joinpath(site, name)), (".git", ".github", "README.md", "Project.toml")),
            "The archive must contain generated pages, not repository source")
    return manifest
end

function preserve_releases(previous, current)
    old = Dict(row["version"] => row for row in previous["versions"] if is_version(row["version"]))
    new = Dict(row["version"] => row for row in current["versions"] if is_version(row["version"]))
    require(issubset(Set(keys(old)), Set(keys(new))), "Published release versions cannot be removed")
    for (version, row) in old
        require(all(key -> get(row, key, nothing) == get(new[version], key, nothing), ("code_sha", "docs_sha")),
                "Published release source identity cannot be changed")
    end
    require(version_number(current["stable"]) >= version_number(previous["stable"]), "Stable cannot move backwards")
    return nothing
end

function current_sources(repo, manifest; source_sha=nothing, source_ref=nothing, legacy=LEGACY_DOCS_SOURCE)
    if source_ref !== nothing
        controller = startswith(source_ref, "refs/tags/") ? "refs/heads/main" : source_ref
        require(controller in CODE_REFS, "Unexpected publication source ref")
        remote_ref(repo, controller) == source_sha || return false
    end
    rows = Dict(row["version"] => row for row in manifest["versions"])
    for (version, ref) in SOURCE_REFS
        require(haskey(rows, version), "The development version is required")
        row = rows[version]
        remote_ref(repo, ref) == row["code_sha"] || return false
        require(row["docs_sha"] == documentation_source(repo, row["code_sha"]; legacy), "Branch documentation source mismatch")
    end
    tags = remote_tags(repo)
    for (version, row) in rows
        if is_version(version)
            ref = "refs/tags/$version"
            require(get(tags, ref * "^{}", get(tags, ref, nothing)) == row["code_sha"],
                    "Release tag changed since the build")
        end
    end
    return true
end

function publish(repo, site, source_sha, manifest; source_ref=nothing, legacy=LEGACY_DOCS_SOURCE)
    require(is_sha(source_sha) && git(repo, "rev-parse", "HEAD") == source_sha,
            "Source checkout does not match the checked workflow commit")
    require(isempty(git(repo, "status", "--porcelain")), "Source checkout must be clean")
    current_sources(repo, manifest; source_sha, source_ref, legacy) || return Dict("publish" => false, "reason" => "superseded_source")
    parent = remote_ref(repo, ARCHIVE_REF)
    if parent !== nothing
        git(repo, "fetch", "--no-tags", "origin", ARCHIVE_REF)
        parent = git(repo, "rev-parse", "FETCH_HEAD")
        preserve_releases(parse_json(git(repo, "show", "$parent:versions.json")), manifest)
    end
    touch(joinpath(site, ".nojekyll"))
    rows = Dict(row["version"] => row for row in manifest["versions"])
    info = Dict("schema_version" => 1, "source_ref" => source_ref, "source_sha" => source_sha,
                "dev_code_sha" => rows["dev"]["code_sha"],
                "stable" => manifest["stable"])
    write_json(joinpath(site, "archive-info.json"), info)
    tree = mktempdir() do temporary
        env = Dict("GIT_INDEX_FILE" => joinpath(temporary, "index"))
        git(repo, "read-tree", "--empty"; env)
        git_dir = git(repo, "rev-parse", "--absolute-git-dir")
        command = Cmd(Cmd(["git", "--git-dir", git_dir, "--work-tree", abspath(site), "add", "--all", "--force", "--", "."]); dir=site)
        run(addenv(command, env))
        git(repo, "write-tree"; env)
    end
    if parent !== nothing && git(repo, "rev-parse", "$parent^{tree}") == tree
        return Dict("publish" => true, "sha" => parent, "updated" => false)
    end
    env = Dict("GIT_AUTHOR_NAME" => COMMIT_NAME, "GIT_AUTHOR_EMAIL" => COMMIT_EMAIL,
               "GIT_COMMITTER_NAME" => COMMIT_NAME, "GIT_COMMITTER_EMAIL" => COMMIT_EMAIL)
    args = String["commit-tree", tree]
    parent === nothing || append!(args, ["-p", parent])
    append!(args, ["-m", "Deploy checked documentation from $source_sha"])
    commit = git(repo, args...; env)
    current_sources(repo, manifest; source_sha, source_ref, legacy) || return Dict("publish" => false, "reason" => "superseded_source")
    git(repo, "push", "origin", "$commit:$ARCHIVE_REF")
    return Dict("publish" => true, "sha" => commit, "updated" => true)
end
