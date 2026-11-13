function within_root(root, path)
    relative = relpath(path, root)
    return !isabspath(relative) && !(".." in splitpath(relative))
end

function preview_response(request, directory)
    headers = ["Cache-Control" => "no-store"]
    request.method in ("GET", "HEAD") || return HTTP.Response(501, headers, "Unsupported method")
    root = realpath(directory)
    uri = URIs.URI(request.target)
    path = URIs.unescapeuri(uri.path)
    base = "/HighOrderMRI.jl/"
    relative = startswith(path, base) ? chopprefix(path, base) : lstrip(path, '/')
    target = normpath(joinpath(root, relative))
    within_root(root, target) || return HTTP.Response(404, headers, "Not found")
    isdir(target) && (target = joinpath(target, "index.html"))
    !ispath(target) && isempty(splitext(target)[2]) && (target *= ".html")
    isfile(target) && within_root(root, realpath(target)) || return HTTP.Response(404, headers, "Not found")
    extension = lowercase(splitext(target)[2])
    mime = get(Dict(".html" => "text/html; charset=utf-8", ".css" => "text/css", ".js" => "text/javascript",
                    ".json" => "application/json", ".svg" => "image/svg+xml", ".png" => "image/png",
                    ".woff2" => "font/woff2", ".ico" => "image/x-icon"), extension, "application/octet-stream")
    push!(headers, "Content-Type" => mime, "Content-Length" => string(filesize(target)))
    body = request.method == "HEAD" ? UInt8[] : read(target)
    return HTTP.Response(200, headers, body)
end

function serve_preview(directory, port)
    require(isfile(joinpath(directory, "versions.json")), "Build the complete website before starting its preview")
    require(0 <= port <= 65535, "Invalid preview port")
    println("Preview http://localhost:$port/HighOrderMRI.jl/")
    flush(stdout)
    HTTP.serve(request -> preview_response(request, directory), "127.0.0.1", port)
    return nothing
end
