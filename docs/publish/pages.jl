mutable struct PageData
    ids::Dict{String,Int}
    links::Vector{String}
    assets::Vector{String}
    math_errors::Vector{String}
    math::Int
    text::Vector{String}
    has_html::Bool
end

function walk_html!(page, node)
    if EzXML.iselement(node)
        name = lowercase(EzXML.nodename(node))
        if haskey(node, "id")
            id = node["id"]
            page.ids[id] = get(page.ids, id, 0) + 1
        end
        name == "a" && haskey(node, "href") && push!(page.links, node["href"])
        name in ("img", "script") && haskey(node, "src") && push!(page.assets, node["src"])
        name == "mjx-container" && (page.math += 1)
        marker = haskey(node, "data-mml-node") ? node["data-mml-node"] : ""
        classes = haskey(node, "class") ? node["class"] : ""
        if name == "merror" || marker == "merror" || occursin("katex-error", classes)
            push!(page.math_errors, name)
        end
        name in ("script", "style") && return nothing
    elseif EzXML.istext(node) || EzXML.iscdata(node)
        push!(page.text, EzXML.nodecontent(node))
    end
    for child in EzXML.eachnode(node)
        walk_html!(page, child)
    end
    return nothing
end

function parse_page(text)
    visible_markup = replace(text, r"<!--.*?-->|<script\b[^>]*>.*?</script>|<style\b[^>]*>.*?</style>"is => "")
    page = PageData(Dict{String,Int}(), String[], String[], String[], 0, String[],
                    occursin(r"<html(?:\s|>)"i, visible_markup))
    isempty(strip(text)) && return page
    document = EzXML.parsehtml(text; noerror=true, nowarning=true)
    walk_html!(page, EzXML.root(document))
    return page
end

function check_site(docs, dist; base="/HighOrderMRI.jl/dev/", version="dev")
    mapping = TOML.parsefile(joinpath(docs, "api-map.toml"))
    index = TOML.parsefile(joinpath(docs, ".generated", "api", "index.toml"))
    meta = read_json(joinpath(docs, ".generated", "api", "build-meta.json"))
    pages = Dict{String,PageData}()
    for (directory, _, files) in walkdir(dist), file in files
        endswith(file, ".html") || continue
        path = replace(relpath(joinpath(directory, file), dist), '\\' => '/')
        pages[path] = parse_page(read(joinpath(directory, file), String))
    end
    errors = Any[]
    fail(page, reason, value="") = push!(errors, Dict("page" => page, "reason" => reason, "value" => value))
    for (path, page) in pages
        isempty(page.math_errors) || fail(path, "math rendering error")
        content = read(joinpath(dist, path), String)
        (occursin("<!--@include:", content) || occursin("(@ref ", content)) && fail(path, "unexpanded source directive")
        for value in vcat(page.links, page.assets)
            url = URIs.URI(value)
            isempty(url.scheme) && isempty(url.host) || continue
            request = URIs.unescapeuri(url.path)
            if startswith(request, base)
                request = chopprefix(request, base)
                isempty(request) && (request = "index.html")
            elseif startswith(request, "/")
                fail(path, "outside configured base", value)
                continue
            elseif isempty(request)
                request = path
            else
                request = replace(normpath(joinpath(dirname(path), request)), '\\' => '/')
            end
            choices = [request, request * ".html", rstrip(request, '/') * "/index.html"]
            found = findfirst(candidate -> isfile(joinpath(dist, candidate)), choices)
            if found === nothing
                fail(path, "missing target", value)
            elseif !isempty(url.fragment)
                target = choices[found]
                if haskey(pages, target) && !haskey(pages[target].ids, URIs.unescapeuri(url.fragment))
                    fail(path, "missing anchor", value)
                end
            end
        end
    end
    entries = Dict(entry["id"] => entry for entry in mapping["api"])
    generated = Dict(entry["id"] => entry for entry in index["api"])
    !index["partial"] && Set(keys(entries)) != Set(keys(generated)) && fail("api", "incomplete generated coverage")
    for (id, record) in generated
        entry = entries[id]
        target = chopsuffix(entry["page"], ".md") * ".html"
        if !haskey(pages, target)
            fail(target, "missing API page", id)
            continue
        end
        page = pages[target]
        anchors = [entry["anchor"]; [part["anchor"] for part in get(entry, "parts", [])]]
        for anchor in anchors
            get(page.ids, anchor, 0) == 1 || fail(target, "API anchor must occur once", anchor)
        end
        for source in record["sources"]
            prefix = "https://github.com/$REPOSITORY/blob/$(meta["codeRef"])/$(source["path"])"
            link = meta["preview"] ? prefix : prefix * "#L$(source["line"])"
            link in page.links || fail(target, "source link does not match generated metadata", link)
        end
        branch = version == "dev" ? "dev_jinyuan" : "main"
        edit = "https://github.com/$REPOSITORY/edit/$branch/docs/src/$(entry["page"])"
        version in ("dev", "main") && !(edit in page.links) && fail(target, "edit link does not target its code branch")
    end
    for entry in get(mapping, "exports", [])
        target = chopsuffix(entry["page"], ".md") * ".html"
        if !haskey(pages, target)
            fail(target, "missing export disposition page", entry["binding"])
        elseif haskey(entry, "anchor") && get(pages[target].ids, entry["anchor"], 0) != 1
            fail(target, "missing export disposition anchor", entry["anchor"])
        elseif !occursin(entry["binding"], read(joinpath(dist, target), String))
            fail(target, "export disposition has no visible binding name", entry["binding"])
        end
    end
    expected = Set{String}()
    for (directory, _, files) in walkdir(joinpath(docs, "src")), file in files
        endswith(file, ".md") && !(".vitepress" in splitpath(directory)) || continue
        push!(expected, replace(chopsuffix(relpath(joinpath(directory, file), joinpath(docs, "src")), ".md"), '\\' => '/') * ".html")
    end
    Set(keys(pages)) == union(expected, Set(["404.html"])) || fail("site", "unexpected or missing content pages")
    if !(version in ("dev", "main")) && any(link -> occursin("/edit/", link), (link for page in values(pages) for link in page.links))
        fail("site", "frozen release exposes an edit-current-docs link")
    end
    return Dict("content_pages" => length(expected), "html_pages" => length(pages), "api_bindings" => length(generated),
                "original_bindings" => count(entry -> get(entry, "original_documenter", true), values(entries)),
                "additional_export_dispositions" => length(get(mapping, "exports", [])),
                "links" => sum(length(page.links) for page in values(pages); init=0),
                "math_expressions" => sum(page.math for page in values(pages); init=0), "errors" => errors)
end

struct CheckError <: Exception
    message::String
end
Base.showerror(io::IO, error::CheckError) = print(io, error.message)
check(condition, message) = condition || throw(CheckError(message))

struct PageResponse
    status::Int
    url::String
    content_type::String
    body::Vector{UInt8}
end
PageResponse(status, url, content_type) = PageResponse(status, url, content_type, UInt8[])

struct PageSpec
    path::String
    phrase::String
    source_link::Bool
    svg::Bool
end
PageSpec(path; phrase="", source_link=false, svg=false) = PageSpec(path, phrase, source_link, svg)

const LIVE_PAGES = (
    PageSpec("dev/"; phrase="The common signal model is:"),
    PageSpec("dev/theory/symbols"; phrase="Reconstruction mask"),
    PageSpec("dev/reference/highorderlowrankop"; source_link=true),
    PageSpec("dev/figures/adaptive-shared-basis/Fig1_adaptive_shared_basis.svg"; svg=true),
    PageSpec("dev/figures/adaptive-shared-basis/Fig1_adaptive_shared_basis_dark.svg"; svg=true),
)

function origin(url)
    uri = URIs.URI(url)
    scheme = lowercase(uri.scheme)
    check(scheme in ("http", "https") && !isempty(uri.host) && isempty(uri.userinfo),
          "site URL must be an HTTP(S) URL without credentials")
    port = isempty(uri.port) ? (scheme == "https" ? 443 : 80) : tryparse(Int, uri.port)
    check(port !== nothing && 1 <= port <= 65535, "site URL has an invalid port")
    return (scheme, lowercase(uri.host), port)
end

function site_root(url)
    origin(url)
    uri = URIs.URI(url)
    check(isempty(uri.query) && isempty(uri.fragment), "site root URL must not contain a query or fragment")
    return rstrip(url, '/') * "/"
end

function fetch_response(url, method, timeout)
    allowed = origin(url)
    current = String(url)
    seconds = max(1, ceil(Int, timeout))
    for _ in 1:11
        response = Ref{PageResponse}()
        location = Ref("")
        try
            HTTP.open(method, current, ["Cache-Control" => "no-cache", "User-Agent" => "HighOrderMRI-Pages-Check",
                                       "Content-Length" => "0", "Accept-Encoding" => "identity"];
                      redirect=false, retry=false, cookies=false, status_exception=false,
                      readtimeout=seconds, connect_timeout=seconds) do stream
                HTTP.closewrite(stream)
                metadata = HTTP.startread(stream)
                location[] = HTTP.header(metadata, "Location", "")
                body = UInt8[]
                if method != "HEAD"
                    # HTTP.Stream reads available chunk data, not necessarily the
                    # complete message. Keep reading until the response EOF.
                    while !eof(stream)
                        append!(body, read(stream, min(64 * 1024, 8_000_001 - length(body))))
                        check(length(body) <= 8_000_000, "HTML response exceeds the size limit")
                    end
                end
                response[] = PageResponse(metadata.status, current, HTTP.header(metadata, "Content-Type", ""), body)
                return nothing
            end
        catch error
            error isa HTTP.RequestError && error.error isa CheckError && throw(error.error)
            error isa Union{HTTP.Exceptions.HTTPError, Base.IOError, EOFError} || rethrow()
            throw(CheckError("network error ($(sprint(showerror, error)))"))
        end
        result = response[]
        if result.status in (301, 302, 303, 307, 308) && !isempty(location[])
            target = string(URIs.resolvereference(current, location[]))
            check(origin(target) == allowed, "redirect leaves the site origin")
            current = target
            continue
        end
        check(result.status < 400, "HTTP $(result.status)")
        return result
    end
    throw(CheckError("too many redirects"))
end

function check_response(page, response, root, code_sha)
    check(response.status == 200, "HTTP $(response.status)")
    check(origin(response.url) == origin(root), "response leaves the site origin")
    mime = lowercase(strip(first(split(response.content_type, ';'))))
    if page.svg
        check(mime == "image/svg+xml", "HTTP 200 but expected an SVG content type")
        return nothing
    end
    check(mime in ("text/html", "application/xhtml+xml"), "HTTP 200 but expected an HTML content type")
    text = String(copy(response.body))
    check(isvalid(text), "HTML is not valid UTF-8")
    html = parse_page(text)
    check(html.has_html, "HTTP 200 but response is not an HTML document")
    visible = join(split(join(html.text, " ")), " ")
    isempty(page.phrase) || check(occursin(page.phrase, visible), "HTTP 200 but expected page text is missing")
    if page.source_link
        check(any(html.links) do link
            uri = URIs.URI(link)
            uri.scheme in ("http", "https") && occursin("/blob/$code_sha/", uri.path)
        end, "HTTP 200 but source links do not match the pinned code SHA")
    end
    return nothing
end

function verify_site(root, code_sha; attempts=12, interval=10.0, max_wait=120.0, fetch=fetch_response,
                     sleep=Base.sleep, clock=() -> time_ns() / 1e9, emit=println, prefix="dev")
    root = site_root(root)
    check(attempts >= 1 && isfinite(interval) && interval >= 0 && isfinite(max_wait) && max_wait > 0,
          "attempts and max-wait must be positive; interval must be nonnegative")
    deadline = clock() + max_wait
    for attempt in 1:attempts
        failures = String[]
        for original in LIVE_PAGES
            page = PageSpec(replace(original.path, "dev/" => prefix * "/"; count=1), original.phrase, original.source_link, original.svg)
            remaining = deadline - clock()
            if remaining <= 0
                push!(failures, "$(page.path): verification deadline reached")
                break
            end
            url = string(URIs.resolvereference(root, page.path)) * "?pages_check=" * string(uuid4())
            try
                response = fetch(url, page.svg ? "HEAD" : "GET", min(10, remaining))
                check_response(page, response, root, code_sha)
            catch error
                error isa CheckError || rethrow()
                push!(failures, "$(page.path): $(error.message)")
            end
        end
        if isempty(failures)
            emit("Verified 3 $prefix pages and 2 Fig.1 SVGs for code $code_sha.")
            return true
        end
        foreach(failure -> emit("Attempt $attempt/$attempts: $failure"), failures)
        remaining = deadline - clock()
        (attempt == attempts || remaining <= 0) && break
        sleep(min(interval, remaining))
    end
    emit("Published-site verification failed within the configured retry limit.")
    return false
end

function expected_versions(config_path)
    branches = read_json(config_path)
    check(get(branches, "schema_version", nothing) === 1 && is_sha(get(branches, "code_sha", nothing)), "Invalid code-version.json")
    config = read_json(joinpath(dirname(config_path), "versions.json"))
    records = Any[Dict("version" => "dev", "code_sha" => branches["code_sha"], "docs_sha" => nothing)]
    check(!haskey(branches, "main_sha"), "The main documentation channel is not published")
    append!(records, config["releases"])
    stable = only(filter(row -> row["version"] == config["stable"], config["releases"]))
    push!(records, merge(stable, Dict("version" => "stable", "release" => stable["version"])))
    return config["stable"], records
end

function check_version_index(response, stable, expected; allow_preview=false)
    check(response.status == 200 && lowercase(strip(first(split(response.content_type, ';')))) == "application/json",
          "published version index is missing or not JSON")
    actual = try
        parse_json(response.body)
    catch error
        error isa ArgumentError || rethrow()
        throw(CheckError("invalid published version index"))
    end
    check(actual isa AbstractDict && get(actual, "schema_version", nothing) === 1 && get(actual, "stable", nothing) == stable,
          "published stable version differs from configuration")
    preview = get(actual, "preview", nothing)
    check(preview isa Bool && (!preview || allow_preview), "candidate artifact cannot pass formal publication checks")
    records = get(actual, "versions", [])
    check(records isa AbstractVector && all(row -> row isa AbstractDict, records), "invalid published version records")
    check(length(records) == length(expected) && length(Set(get(row, "version", nothing) for row in records)) == length(expected),
          "published version set differs from configuration")
    for target in expected
        found = findfirst(row -> get(row, "version", nothing) == target["version"], records)
        check(found !== nothing && all(pair -> get(records[found], first(pair), nothing) == last(pair), target),
              "published version identity differs from configuration")
        check(get(records[found], "preview", nothing) === preview, "published version candidate status is inconsistent")
    end
    return nothing
end
