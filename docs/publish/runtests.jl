using Test
using Sockets
using Tar
include("PublicationTools.jl")
const PT = PublicationTools
const SHA_A = repeat("a", 40)
const SHA_B = repeat("b", 40)
const SHA_C = repeat("c", 40)

published(version="v0.1.0") = Dict("tag_name" => version, "draft" => false, "prerelease" => false, "published_at" => "2026-11-12T00:00:00Z")

function with_repository(f)
    mktempdir() do root
        repo, remote = joinpath(root, "source"), joinpath(root, "remote.git")
        run(pipeline(`git init --bare $remote`; stdout=devnull, stderr=devnull))
        run(pipeline(`git init -b main $repo`; stdout=devnull, stderr=devnull))
        PT.git(repo, "config", "user.name", "Fixture")
        PT.git(repo, "config", "user.email", "fixture@example.invalid")
        write(joinpath(repo, "Project.toml"), "version = \"0.1.0\"\n")
        write(joinpath(repo, "source.md"), "Preserve source")
        PT.git(repo, "add", ".")
        PT.git(repo, "commit", "-m", "legacy code")
        code = PT.git(repo, "rev-parse", "HEAD")
        PT.git(repo, "tag", "v0.1.0")
        PT.git(repo, "branch", "dev_jinyuan")
        mkpath(joinpath(repo, "docs", "src"))
        write(joinpath(repo, "docs", "src", "index.md"), "Preserve documentation")
        write(joinpath(repo, "docs", "api-map.toml"), "schema_version = 1\n")
        PT.git(repo, "add", ".")
        PT.git(repo, "commit", "-m", "saved documentation")
        legacy = PT.git(repo, "rev-parse", "HEAD")
        release = Dict("version" => "v0.1.0", "code_sha" => code, "docs_sha" => legacy)
        PT.write_json(joinpath(repo, "docs", "versions.json"), Dict("schema_version" => 1, "stable" => "v0.1.0", "releases" => [release]))
        PT.git(repo, "add", ".")
        PT.git(repo, "commit", "-m", "in-tree migration")
        source = PT.git(repo, "rev-parse", "HEAD")
        PT.git(repo, "remote", "add", "origin", remote)
        PT.git(repo, "push", "origin", "main", "dev_jinyuan", "--tags")
        return f((; root, repo, remote, code, legacy, release, source, dev=code))
    end
end

function site_fixture(root; code=SHA_A, source=nothing, main=nothing, main_source=nothing,
                      release=Dict("version" => "v0.1.0", "code_sha" => SHA_B, "docs_sha" => SHA_C))
    rows = Any[Dict("version" => "dev", "code_sha" => code, "docs_sha" => source, "preview" => false),
               merge(release, Dict("preview" => false)), merge(release, Dict("version" => "stable", "release" => release["version"], "preview" => false))]
    main === nothing || push!(rows, Dict("version" => "main", "code_sha" => main, "docs_sha" => main_source, "preview" => false))
    manifest = Dict("schema_version" => 1, "stable" => release["version"], "preview" => false, "versions" => rows)
    mkpath(root)
    write(joinpath(root, "index.html"), "stable redirect")
    PT.write_json(joinpath(root, "versions.json"), manifest)
    for row in rows
        directory = joinpath(root, row["version"])
        mkpath(joinpath(directory, "reference"))
        write(joinpath(directory, "index.html"), row["version"])
        write(joinpath(directory, "reference", "highorderlowrankop.html"), row["code_sha"])
        PT.write_json(joinpath(directory, "version.json"), row)
    end
    return manifest
end

function fixture_plan(f; published=[published()], source=f.source, event="workflow_dispatch", ref="refs/heads/main", kwargs...)
    return PT.resolve_versions(f.repo, published, Dict(), source, event, ref; legacy=f.legacy, kwargs...)
end

@testset "Publication contexts and formal release catalog" begin
    flush(stdout)
    for event in ("push", "workflow_dispatch"), ref in PT.CODE_REFS
        @test PT.validate_context(PT.REPOSITORY, event, ref) === nothing
    end
    for (repository, event, ref) in [("other/repo", "push", "refs/heads/main"),
                                   (PT.REPOSITORY, "pull_request", "refs/heads/main"),
                                   (PT.REPOSITORY, "push", "refs/heads/docs"),
                                   (PT.REPOSITORY, "push", PT.ARCHIVE_REF)]
        @test_throws ArgumentError PT.validate_context(repository, event, ref)
    end
    payload = Dict("action" => "published", "release" => published("v0.2.0"))
    @test PT.validate_context(PT.REPOSITORY, "release", "refs/tags/v0.2.0", payload) === nothing
    @test_throws ArgumentError PT.validate_context(PT.REPOSITORY, "release", "refs/tags/v0.1.0", payload)
    for (key, value) in [("draft", true), ("prerelease", true), ("published_at", nothing), ("published_at", ""), ("tag_name", "v9.9.9")]
        @test_throws ArgumentError PT.validate_release_metadata(merge(published(), Dict(key => value)), "v0.1.0")
    end
    config = Dict("releases" => [Dict("version" => "v0.1.0")])
    @test PT.validate_release_catalog(config, [published(), merge(published("v0.2.0"), Dict("prerelease" => true))]) === nothing
    @test_throws ArgumentError PT.validate_release_catalog(config, [published(), published("v0.2.0")])
    @test_throws ArgumentError PT.version_number("../outside")
    @test PT.version_number("v0.10.0") > PT.version_number("v0.9.0")
    @test PT.is_preview("push", "refs/heads/main")
    @test PT.is_preview("pull_request", "refs/pull/1/merge")
    @test !PT.is_preview("push", "refs/heads/dev_jinyuan")
    @test !PT.is_preview("workflow_dispatch", "refs/heads/main")
    @test !PT.is_preview("release", "refs/tags/v0.1.0")
end

@testset "Resolved source histories and annotated release tags" begin
    flush(stdout)
    with_repository() do f
        plan = fixture_plan(f)
        rows = Dict(row["version"] => row for row in plan["targets"])
        @test Set(keys(rows)) == Set(["dev", "stable", "v0.1.0"])
        @test rows["dev"]["code_sha"] == f.code
        @test rows["dev"]["docs_sha"] == rows["v0.1.0"]["docs_sha"] == f.legacy
        @test isempty(PT.git(f.repo, "ls-remote", "origin", "refs/heads/docs"))
        preview = fixture_plan(f; event="pull_request", ref="refs/pull/1/merge", base="dev_jinyuan")
        @test first(preview["targets"])["code_sha"] == first(preview["targets"])["docs_sha"] == f.source
        main_preview = fixture_plan(f; event="pull_request", ref="refs/pull/1/merge", base="main")
        @test first(main_preview["targets"])["code_sha"] == f.source
        main_push = fixture_plan(f; event="push")
        @test first(main_push["targets"])["code_sha"] == first(main_push["targets"])["docs_sha"] == f.source
        @test_throws ArgumentError fixture_plan(f; event="pull_request", base="other")
        @test fixture_plan(f; event="workflow_dispatch", release_tag="v0.1.0")["stable"] == "v0.1.0"
        for tag in ("v9.9.9", "../outside")
            @test_throws ArgumentError fixture_plan(f; event="workflow_dispatch", release_tag=tag)
        end
        @test PT.verify_releases(f.repo, plan; allow_unpublished=true) === nothing
        PT.git(f.repo, "tag", "v0.3.0", f.code)
        @test_throws ArgumentError fixture_plan(f; published=[published(), published("v0.3.0")])
        write(joinpath(f.repo, "Project.toml"), "version = \"0.2.0\"\n")
        PT.git(f.repo, "add", ".")
        PT.git(f.repo, "commit", "-m", "release with source docs")
        code = PT.git(f.repo, "rev-parse", "HEAD")
        PT.git(f.repo, "tag", "-a", "v0.2.0", "-m", "annotated release")
        PT.git(f.repo, "push", "origin", "main", "--tags")
        next_plan = fixture_plan(f; published=[published(), published("v0.2.0")], source=code, event="release", ref="refs/tags/v0.2.0")
        @test next_plan["stable"] == "v0.2.0"
        @test first(next_plan["releases"]) == Dict("version" => "v0.2.0", "code_sha" => code, "docs_sha" => code)
        @test last(next_plan["releases"])["docs_sha"] == f.legacy
        @test fixture_plan(f; published=[published(), merge(published("v0.2.0"), Dict("prerelease" => true))])["stable"] == "v0.1.0"
        archive = Dict("versions" => [merge(f.release, Dict("docs_sha" => f.source))])
        @test_throws ArgumentError PT.resolve_versions(f.repo, [published()], archive, code, "push", "refs/heads/main"; legacy=f.legacy)
        PT.git(f.repo, "update-ref", "refs/tags/v0.1.0", code)
        @test_throws ArgumentError fixture_plan(f)
    end
end

@testset "Complete artifacts, previews and preserved release identities" begin
    flush(stdout)
    mktempdir() do temporary
        site = joinpath(temporary, "site")
        manifest = site_fixture(site)
        code = Dict("schema_version" => 1, "code_sha" => SHA_A)
        releases = Dict("schema_version" => 1, "stable" => "v0.1.0", "releases" => [Dict("version" => "v0.1.0", "code_sha" => SHA_B, "docs_sha" => SHA_C)])
        @test PT.validate_site(site, code, releases) == manifest
        for changed in [merge(manifest, Dict("preview" => true)), merge(manifest, Dict("stable" => "v0.2.0")),
                        merge(manifest, Dict("versions" => manifest["versions"][1:1]))]
            PT.write_json(joinpath(site, "versions.json"), changed)
            @test_throws ArgumentError PT.validate_site(site, code, releases)
        end
        PT.write_json(joinpath(site, "versions.json"), manifest)
        PT.write_json(joinpath(site, "dev", "version.json"), Dict())
        @test_throws ArgumentError PT.validate_site(site, code, releases)
        PT.write_json(joinpath(site, "dev", "version.json"), first(manifest["versions"]))
        symlink(temporary, joinpath(site, "external"))
        @test_throws ArgumentError PT.validate_site(site, code, releases)
        rm(joinpath(site, "external"))
        write(joinpath(site, "README.md"), "source")
        @test_throws ArgumentError PT.validate_site(site, code, releases)
        rm(joinpath(site, "README.md"))
        @test_throws ArgumentError PT.validate_site(site, merge(code, Dict("main_sha" => SHA_C)), releases)
        mkpath(joinpath(site, "main"))
        @test_throws ArgumentError PT.validate_site(site, code, releases)
        rm(joinpath(site, "main"))
        @test PT.preserve_releases(manifest, manifest) === nothing
        removed = deepcopy(manifest)
        filter!(row -> row["version"] != "v0.1.0", removed["versions"])
        @test_throws ArgumentError PT.preserve_releases(manifest, removed)
        for key in ("code_sha", "docs_sha")
            moved = deepcopy(manifest)
            only(filter(row -> row["version"] == "v0.1.0", moved["versions"]))[key] = SHA_A
            @test_throws ArgumentError PT.preserve_releases(manifest, moved)
        end
        @test_throws ArgumentError PT.preserve_releases(manifest, merge(manifest, Dict("stable" => "v0.0.9")))
    end
end

@testset "Real Git archive update, idempotence, author and source ownership" begin
    flush(stdout)
    with_repository() do f
        site = joinpath(f.root, "site")
        manifest = site_fixture(site; code=f.dev, source=f.legacy, release=f.release)
        first = PT.publish(f.repo, site, f.source, manifest; legacy=f.legacy)
        @test first["publish"] && first["updated"]
        @test PT.git(f.repo, "show", "-s", "--format=%an|%ae|%cn|%ce", first["sha"]) == join([PT.COMMIT_NAME, PT.COMMIT_EMAIL, PT.COMMIT_NAME, PT.COMMIT_EMAIL], "|")
        files = Set(split(PT.git(f.repo, "ls-tree", "--name-only", first["sha"]), '\n'))
        @test files == Set([".nojekyll", "archive-info.json", "index.html", "versions.json", "dev", "stable", "v0.1.0"])
        previous = site_fixture(joinpath(f.root, "previous"); code=f.dev, source=f.legacy,
                                main=f.source, main_source=f.source, release=f.release)
        @test PT.preserve_releases(previous, manifest) === nothing
        again = PT.publish(f.repo, site, f.source, manifest; legacy=f.legacy)
        @test !again["updated"] && again["sha"] == first["sha"]
        write(joinpath(site, "dev", "index.html"), "updated page")
        next = PT.publish(f.repo, site, f.source, manifest; legacy=f.legacy)
        @test PT.git(f.repo, "rev-parse", "$(next["sha"])^") == first["sha"]
        @test PT.git(f.repo, "rev-parse", "HEAD") == f.source
        @test isempty(PT.git(f.repo, "status", "--porcelain"))
        @test read(joinpath(f.repo, "source.md"), String) == "Preserve source"
        changed = deepcopy(manifest)
        only(filter(row -> row["version"] == "v0.1.0", changed["versions"]))["docs_sha"] = SHA_C
        @test_throws ArgumentError PT.publish(f.repo, site, f.source, changed; legacy=f.legacy)
        @test PT.remote_ref(f.repo, PT.ARCHIVE_REF) == next["sha"]
        @test_throws ArgumentError PT.publish(f.repo, site, SHA_A, manifest; legacy=f.legacy)
        write(joinpath(f.repo, "source.md"), "dirty")
        @test_throws ArgumentError PT.publish(f.repo, site, f.source, manifest; legacy=f.legacy)
    end
    with_repository() do f
        site = joinpath(f.root, "site")
        manifest = site_fixture(site; code=f.dev, source=f.legacy, release=f.release)
        @test PT.current_sources(f.repo, manifest; source_sha=f.source, source_ref="refs/tags/v0.1.0", legacy=f.legacy)
        PT.git(f.repo, "commit", "--allow-empty", "-m", "superseding code")
        PT.git(f.repo, "push", "origin", "main")
        PT.git(f.repo, "checkout", "--detach", f.source)
        @test !PT.publish(f.repo, site, f.source, manifest; source_ref="refs/heads/main", legacy=f.legacy)["publish"]
        @test !PT.current_sources(f.repo, manifest; source_sha=f.source, source_ref="refs/tags/v0.1.0", legacy=f.legacy)
        @test PT.remote_ref(f.repo, PT.ARCHIVE_REF) === nothing
    end
    with_repository() do f
        site = joinpath(f.root, "site")
        manifest = site_fixture(site; code=f.dev, source=f.legacy, release=f.release)
        PT.git(f.repo, "update-ref", "refs/heads/dev_jinyuan", f.source)
        PT.git(f.repo, "push", "origin", "dev_jinyuan")
        @test !PT.publish(f.repo, site, f.source, manifest; source_ref="refs/heads/dev_jinyuan", legacy=f.legacy)["publish"]
        @test PT.remote_ref(f.repo, PT.ARCHIVE_REF) === nothing
    end
    with_repository() do f
        site = joinpath(f.root, "site")
        manifest = site_fixture(site; code=f.dev, source=f.legacy, release=f.release)
        # The local tag still matches. A moved remote tag must reject publication.
        PT.git(f.remote, "update-ref", "refs/tags/v0.1.0", f.source)
        @test PT.git(f.repo, "rev-parse", "refs/tags/v0.1.0") == f.code
        @test_throws ArgumentError PT.publish(f.repo, site, f.source, manifest; legacy=f.legacy)
        @test PT.remote_ref(f.repo, PT.ARCHIVE_REF) === nothing
    end
end

@testset "TAR safety and failure without partial output" begin
    flush(stdout)
    mktempdir() do temporary
        input = joinpath(temporary, "input")
        site_fixture(input)
        raw = read(`tar --format=ustar -cf - -C $input .`)
        output = joinpath(temporary, "output")
        @test PT.extract_tar(raw, output; code_sha=SHA_A) > 3
        @test isfile(joinpath(output, "dev", "index.html"))
        wrong = joinpath(temporary, "wrong")
        @test_throws ArgumentError PT.extract_tar(raw, wrong; code_sha=SHA_B)
        @test !ispath(wrong)
        traversal = read(`tar --format=ustar --transform=s,^,../, -cf - -C $input .`)
        @test_throws Exception PT.extract_tar(traversal, joinpath(temporary, "traversal"))
        duplicate = read(`tar --format=ustar -cf - -C $input index.html index.html dev`)
        @test_throws Exception PT.extract_tar(duplicate, joinpath(temporary, "duplicate"))
        symlink("../outside", joinpath(input, "link"))
        linked = read(`tar --format=ustar -cf - -C $input .`)
        @test_throws Exception PT.extract_tar(linked, joinpath(temporary, "linked"))
        rm(joinpath(input, "link"))
        rm(joinpath(input, "dev", "reference", "highorderlowrankop.html"))
        missing = read(`tar --format=ustar -cf - -C $input .`)
        @test_throws ArgumentError PT.extract_tar(missing, joinpath(temporary, "missing"))
        @test_throws ArgumentError PT.extract_tar(raw, output)
    end
end

function good_response(url, method, timeout)
    path = PT.URIs.URI(url).path
    if method == "HEAD"
        return PT.PageResponse(200, url, "image/svg+xml")
    end
    body = endswith(path, "symbols") ? "<html><p>Reconstruction mask</p></html>" :
           endswith(path, "highorderlowrankop") ? "<html><a href=\"https://github.com/owner/repository/blob/$SHA_A/src/op.jl\">Source</a></html>" :
           "<html><p>The common signal model is:</p></html>"
    return PT.PageResponse(200, url, "text/html; charset=utf-8", Vector{UInt8}(codeunits(body)))
end

@testset "Live-site semantics, bounded retries and independent version identity" begin
    flush(stdout)
    root = "https://example.test/package/"
    requests = []
    fetch(url, method, timeout) = (push!(requests, (url, method, timeout)); good_response(url, method, timeout))
    @test PT.verify_site(root, SHA_A; attempts=1, fetch, emit=identity)
    @test getindex.(requests, 2) == ["GET", "GET", "GET", "HEAD", "HEAD"]
    @test length(Set(PT.URIs.URI(row[1]).query for row in requests)) == 5
    @test all(row -> startswith(PT.URIs.URI(row[1]).path, "/package/dev/") && 0 < row[3] <= 10, requests)
    page = PT.LIVE_PAGES[3]
    stale = "<html>$SHA_A<a href=\"https://github.com/owner/repository/blob/$SHA_B/src/op.jl\">Source</a></html>"
    @test_throws PT.CheckError PT.check_response(page, PT.PageResponse(200, root, "text/html", Vector{UInt8}(codeunits(stale))), root, SHA_A)
    for (page, response) in [(PT.LIVE_PAGES[1], PT.PageResponse(200, root, "text/html", collect(codeunits("<html>Old home</html>")))),
                              (PT.LIVE_PAGES[1], PT.PageResponse(404, root, "text/html")),
                              (PT.LIVE_PAGES[1], PT.PageResponse(200, root, "application/json")),
                              (PT.LIVE_PAGES[1], PT.PageResponse(200, root, "text/html")),
                              (PT.LIVE_PAGES[1], PT.PageResponse(200, root, "text/html", collect(codeunits("The common signal model is:")))),
                              (PT.LIVE_PAGES[1], PT.PageResponse(200, root, "text/html", collect(codeunits("<html><script>The common signal model is:</script></html>")))),
                              (PT.LIVE_PAGES[4], PT.PageResponse(200, root, "text/html"))]
        @test_throws PT.CheckError PT.check_response(page, response, root, SHA_A)
    end
    @test_throws PT.CheckError PT.check_response(PT.LIVE_PAGES[4], PT.PageResponse(200, "http://example.test/package/", "image/svg+xml"), root, SHA_A)
    @test_throws PT.CheckError PT.origin("https://user:secret@example.test/")
    @test_throws PT.CheckError PT.origin("file:///tmp/site")
    @test_throws PT.CheckError PT.site_root(root * "?query=1")
    @test_throws PT.CheckError PT.origin("https://example.test:99999/")
    waits, messages = [], String[]
    count = Ref(0)
    function retry_fetch(url, method, timeout)
        count[] += 1
        count[] == 1 && return PT.PageResponse(200, url, "text/html", collect(codeunits("<html>Old homepage</html>")))
        return good_response(url, method, timeout)
    end
    @test PT.verify_site(root, SHA_A; attempts=2, fetch=retry_fetch, sleep=value -> push!(waits, value), emit=value -> push!(messages, value))
    @test count[] == 10 && waits == [10]
    @test occursin("dev/: HTTP 200", first(messages))
    failures = Ref(0)
    bad_fetch(url, method, timeout) = (failures[] += 1; PT.PageResponse(503, url, "text/html"))
    @test !PT.verify_site(root, SHA_A; attempts=2, interval=0, fetch=bad_fetch, emit=identity)
    @test failures[] == 10
    ticks = [0.0, 121.0, 121.0]
    @test !PT.verify_site(root, SHA_A; fetch=bad_fetch, clock=() -> popfirst!(ticks), emit=identity)
    @test failures[] == 10
    records = [Dict("version" => "dev", "code_sha" => SHA_A, "docs_sha" => nothing),
               Dict("version" => "v0.1.0", "code_sha" => SHA_B, "docs_sha" => SHA_C)]
    manifest = Dict("schema_version" => 1, "stable" => "v0.1.0", "preview" => false,
                    "versions" => [merge(row, Dict("preview" => false)) for row in records])
    response(value) = PT.PageResponse(200, root, "application/json", collect(codeunits(PT.json_text(value))))
    @test PT.check_version_index(response(manifest), "v0.1.0", records) === nothing
    @test_throws PT.CheckError PT.check_version_index(response(merge(manifest, Dict("stable" => "v0.0.9"))), "v0.1.0", records)
    @test_throws PT.CheckError PT.check_version_index(response(merge(manifest, Dict("versions" => manifest["versions"][1:1]))), "v0.1.0", records)
    preview = merge(manifest, Dict("preview" => true, "versions" => [merge(row, Dict("preview" => true)) for row in records]))
    @test_throws PT.CheckError PT.check_version_index(response(preview), "v0.1.0", records)
    @test PT.check_version_index(response(preview), "v0.1.0", records; allow_preview=true) === nothing
end

@testset "Preview HTTP, clean URLs, HEAD and redirect origin enforcement" begin
    flush(stdout)
    mktempdir() do root
        PT.write_json(joinpath(root, "versions.json"), Dict())
        mkpath(joinpath(root, "dev"))
        write(joinpath(root, "dev", "index.html"), "<html>index</html>")
        write(joinpath(root, "dev", "guide.html"), "<html>guide</html>")
        write(joinpath(root, "dev", "figure.svg"), "<svg/>")
        symlink(dirname(root), joinpath(root, "escape"))
        function handler(request)
            request.target == "/redirect" && return PT.HTTP.Response(302, ["Location" => "https://outside.invalid/"])
            request.target == "/chunked" && return PT.HTTP.Response(200, ["Transfer-Encoding" => "chunked", "Content-Type" => "text/html"], "<html>chunked response</html>")
            return PT.preview_response(request, root)
        end
        listener = Sockets.listen(ip"127.0.0.1", 0)
        port = Sockets.getsockname(listener)[2]
        server = PT.HTTP.serve!(handler, "127.0.0.1", port; server=listener, verbose=-1)
        try
            base = "http://127.0.0.1:$port"
            @test PT.fetch_response(base * "/HighOrderMRI.jl/dev/", "GET", 3).body == collect(codeunits("<html>index</html>"))
            @test PT.fetch_response(base * "/HighOrderMRI.jl/dev/guide", "GET", 3.5).status == 200
            @test PT.fetch_response(base * "/chunked", "GET", 3.5).body == collect(codeunits("<html>chunked response</html>"))
            head = PT.fetch_response(base * "/HighOrderMRI.jl/dev/figure.svg", "HEAD", 3)
            @test head.content_type == "image/svg+xml" && isempty(head.body)
            @test_throws PT.CheckError PT.fetch_response(base * "/redirect", "GET", 3)
            @test PT.preview_response(PT.HTTP.Request("GET", "/../../outside"), root).status == 404
            @test PT.preview_response(PT.HTTP.Request("GET", "/escape/"), root).status == 404
        finally
            close(server)
            isopen(listener) && close(listener)
        end
    end
end

@testset "SVG byte preservation and explicit palette failure" begin
    flush(stdout)
    figures = normpath(joinpath(@__DIR__, "..", "src", "public", "figures", "adaptive-shared-basis"))
    original = read(joinpath(figures, "Fig1_adaptive_shared_basis.svg"), String)
    @test PT.dark_svg(original) == read(joinpath(figures, "Fig1_adaptive_shared_basis_dark.svg"), String)
    @test_throws ArgumentError PT.recolor_svg("fill:#000000;")
    @test_throws ArgumentError PT.recolor_svg("fill:rgb(1%,2%,3%);")
    @test PT.recolor_svg("stroke:none;") == "stroke:none;"
end

@testset "Command arguments reject ambiguous and incomplete calls" begin
    flush(stdout)
    @test_throws ArgumentError PT.command_options(["--unknown"], ["--output"])
    @test_throws ArgumentError PT.command_options(["--output"], ["--output"])
    @test_throws ArgumentError PT.command_options(["--output", "a", "--output", "b"], ["--output"])
    @test_throws ArgumentError PT.required_option(Dict(), "--output")
    @test_throws ArgumentError PT.command_main(["unknown"])
end

@testset "Built-site API, anchor, math, edit-link and export coverage" begin
    flush(stdout)
    mktempdir() do root
        docs, dist = joinpath(root, "docs"), joinpath(root, "site")
        mkpath(joinpath(docs, "src", "reference"))
        mkpath(joinpath(docs, ".generated", "api"))
        mkpath(joinpath(dist, "reference"))
        for name in ("index.md", "reference/op.md", "reference/exports.md")
            write(joinpath(docs, "src", name), "source")
        end
        mapping = Dict("api" => [Dict("id" => "op", "page" => "reference/op.md", "anchor" => "op")],
                       "exports" => [Dict("binding" => "exported_value", "page" => "reference/exports.md", "anchor" => "exported-value")])
        index = Dict("partial" => false, "api" => [Dict("id" => "op", "sources" => [Dict("path" => "src/op.jl", "line" => 12)])])
        open(io -> PT.TOML.print(io, mapping), joinpath(docs, "api-map.toml"), "w")
        open(io -> PT.TOML.print(io, index), joinpath(docs, ".generated", "api", "index.toml"), "w")
        PT.write_json(joinpath(docs, ".generated", "api", "build-meta.json"), Dict("codeRef" => SHA_A, "preview" => false))
        home = "<html><a href=\"/HighOrderMRI.jl/dev/reference/op#op\">op</a><mjx-container><math><mi>x</mi></math></mjx-container></html>"
        op = "<html><h1 id=\"op\">op</h1><a href=\"https://github.com/$(PT.REPOSITORY)/blob/$SHA_A/src/op.jl#L12\">Source</a><a href=\"https://github.com/$(PT.REPOSITORY)/edit/dev_jinyuan/docs/src/reference/op.md\">Edit</a></html>"
        write(joinpath(dist, "index.html"), home)
        write(joinpath(dist, "reference", "op.html"), op)
        write(joinpath(dist, "reference", "exports.html"), "<html><h1 id=\"exported-value\">exported_value</h1></html>")
        write(joinpath(dist, "404.html"), "<html>Not found</html>")
        report = PT.check_site(docs, dist)
        @test isempty(report["errors"])
        @test report["content_pages"] == 3 && report["html_pages"] == 4
        @test report["api_bindings"] == 1 && report["additional_export_dispositions"] == 1
        @test report["math_expressions"] == 1
        for bad in [replace(op, "#L12" => "#L99"), replace(op, "<h1" => "<p id=\"op\">duplicate</p><h1"),
                    replace(op, "/edit/dev_jinyuan/" => "/edit/main/")]
            write(joinpath(dist, "reference", "op.html"), bad)
            @test !isempty(PT.check_site(docs, dist)["errors"])
        end
        write(joinpath(dist, "reference", "op.html"), op)
        write(joinpath(dist, "index.html"), replace(home, "</html>" => "<merror>bad</merror><a href=\"/outside\">bad</a></html>"))
        errors = PT.check_site(docs, dist)["errors"]
        @test any(error -> error["reason"] == "math rendering error", errors)
        @test any(error -> error["reason"] == "outside configured base", errors)
        @test !isempty(PT.check_site(docs, dist; version="v0.1.0")["errors"])
    end
end

@testset "Local version config excludes main and rejects invalid JSON" begin
    flush(stdout)
    mktempdir() do root
        config = joinpath(root, "code-version.json")
        PT.write_json(config, Dict("schema_version" => 1, "code_sha" => SHA_A))
        PT.write_json(joinpath(root, "versions.json"), Dict("stable" => "v0.1.0", "releases" => [Dict("version" => "v0.1.0", "code_sha" => SHA_B, "docs_sha" => SHA_C)]))
        stable, rows = PT.expected_versions(config)
        @test stable == "v0.1.0" && [row["version"] for row in rows] == ["dev", "v0.1.0", "stable"]
        for value in ["not JSON", "[]", "{}", "{\"schema_version\":true,\"code_sha\":\"$SHA_A\"}",
                      PT.json_text(Dict("schema_version" => 1, "code_sha" => "short")),
                      PT.json_text(Dict("schema_version" => 1, "code_sha" => SHA_A, "main_sha" => "invalid")),
                      PT.json_text(Dict("schema_version" => 1, "code_sha" => SHA_A, "main_sha" => SHA_B))]
            write(config, value)
            @test_throws Exception PT.expected_versions(config)
        end
    end
end

@testset "Independent archive CLI process and exact workflow outputs" begin
    flush(stdout)
    with_repository() do f
        PT.git(f.repo, "update-ref", "refs/heads/dev_jinyuan", f.source)
        PT.git(f.repo, "push", "origin", "dev_jinyuan")
        site = joinpath(f.root, "site")
        manifest = site_fixture(site; code=f.source, source=f.source, release=f.release)
        plan = PT.resolve_versions(f.repo, [published()], Dict(), f.source, "workflow_dispatch", "refs/heads/main")
        PT.write_json(joinpath(site, "publication-plan.json"), plan)
        artifact = joinpath(f.root, "artifact.tar")
        Tar.create(site, artifact)
        event = joinpath(f.root, "event.json")
        PT.write_json(event, Dict())
        output = joinpath(f.root, "workflow-output")
        env = Dict("GITHUB_REPOSITORY" => PT.REPOSITORY, "GITHUB_EVENT_NAME" => "workflow_dispatch", "GITHUB_REF" => "refs/heads/main",
                   "GITHUB_EVENT_PATH" => event, "GITHUB_OUTPUT" => output)
        script = joinpath(@__DIR__, "run.jl")
        project = @__DIR__
        command = Cmd(`$(Base.julia_cmd()) --startup-file=no --project=$project $script archive --artifact-tar $artifact --site $(joinpath(f.root,"extracted")) --source-sha $(f.source)`; dir=f.repo)
        result = PT.parse_json(read(addenv(command, env), String))
        @test result["publish"] && result["updated"]
        @test read(output, String) == "publish=true\nsha=$(result["sha"])\n"
        @test PT.remote_ref(f.repo, PT.ARCHIVE_REF) == result["sha"]
        @test isempty(PT.git(f.repo, "status", "--porcelain"))
        @test PT.parse_json(PT.git(f.repo, "show", "$(result["sha"]):archive-info.json"))["source_ref"] == "refs/heads/main"
        @test PT.git(f.repo, "show", "-s", "--format=%an|%cn", result["sha"]) == "$(PT.COMMIT_NAME)|$(PT.COMMIT_NAME)"
        again = Cmd(`$(Base.julia_cmd()) --startup-file=no --project=$project $script archive --artifact-tar $artifact --site $(joinpath(f.root,"extracted-again")) --source-sha $(f.source)`; dir=f.repo)
        repeated = PT.parse_json(read(addenv(again, env), String))
        @test repeated["publish"] && !repeated["updated"] && repeated["sha"] == result["sha"]
        check_only = joinpath(f.root, "check-only-output")
        withenv(merge(env, Dict("GITHUB_EVENT_NAME" => "push"))...) do
            @test_throws ArgumentError PT.archive_main(["--artifact-tar", artifact, "--site", check_only, "--source-sha", f.source])
        end
        @test !ispath(check_only)
    end
end
