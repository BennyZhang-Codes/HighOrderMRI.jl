module PublicationTools

using Dates
using SHA
using Sockets
using Tar
using TOML
using UUIDs
using Unicode
import EzXML
import HTTP
import JSON
import URIs

const REPOSITORY = "BennyZhang-Codes/HighOrderMRI.jl"
const COMMIT_NAME = "BennyZhang-Codes"
const COMMIT_EMAIL = "57568342+BennyZhang-Codes@users.noreply.github.com"
const SOURCE_REFS = ("dev" => "refs/heads/dev_jinyuan", "main" => "refs/heads/main")
const ARCHIVE_REF = "refs/heads/gh-pages"
const LEGACY_DOCS_SOURCE = "4bf8a2e45bfa703db13fc0796e2c4b205de4429c"
const SHA_PATTERN = r"\A[0-9a-f]{40}\z"
const VERSION_PATTERN = r"\Av(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\z"
const MAX_BYTES = 1024^3

require(condition, message) = condition || throw(ArgumentError(message))
is_sha(value) = value isa AbstractString && occursin(SHA_PATTERN, value)
is_version(value) = value isa AbstractString && occursin(VERSION_PATTERN, value)
version_number(value) = (require(is_version(value), "Invalid formal version"); VersionNumber(value[2:end]))
read_json(path) = JSON.parsefile(path; dicttype=Dict{String,Any})
parse_json(data) = JSON.parse(data; dicttype=Dict{String,Any})
json_text(value) = JSON.json(value; pretty=2, sort_keys=true) * "\n"
write_json(path, value) = write(path, json_text(value))

function git(repo, args...; env=nothing)
    command = Cmd(["git", "-C", String(repo), String.(args)...])
    return strip(read(env === nothing ? command : addenv(command, env), String))
end

function git_status(repo, args...)
    command = Cmd(["git", "-C", String(repo), String.(args)...])
    output = IOBuffer()
    process = run(pipeline(ignorestatus(command); stdout=output, stderr=devnull))
    return (code=process.exitcode, output=strip(String(take!(output))))
end

function remote_ref(repo, ref)
    result = git_status(repo, "ls-remote", "--exit-code", "origin", ref)
    result.code == 2 && return nothing
    require(result.code == 0, "Cannot read remote reference $ref")
    parts = split(result.output)
    require(length(parts) == 2 && is_sha(parts[1]) && parts[2] == ref, "Invalid remote reference")
    return String(parts[1])
end

function remote_tags(repo)
    result = git_status(repo, "ls-remote", "--exit-code", "--tags", "origin", "refs/tags/v*")
    result.code == 2 && return Dict{String,String}()
    require(result.code == 0, "Cannot read remote release tags")
    refs = Dict{String,String}()
    for line in split(result.output, '\n'; keepempty=false)
        parts = split(line)
        require(length(parts) == 2 && is_sha(parts[1]) && startswith(parts[2], "refs/tags/"), "Invalid remote tag")
        refs[parts[2]] = parts[1]
    end
    return refs
end

function command_options(args, values, flags=String[])
    result = Dict{String,String}()
    index = 1
    while index <= length(args)
        option = args[index]
        require(!haskey(result, option), "Duplicate option $option")
        if option in flags
            result[option] = "true"
        else
            require(option in values && index < length(args), "Unknown or incomplete option $option")
            index += 1
            result[option] = args[index]
        end
        index += 1
    end
    return result
end

function required_option(options, name)
    require(haskey(options, name), "Missing $name")
    return options[name]
end

function workflow_output(values)
    open(ENV["GITHUB_OUTPUT"], "a") do output
        for (key, value) in values
            println(output, key, "=", value)
        end
    end
    return nothing
end

include("sources.jl")
include("archive.jl")
include("pages.jl")
include("preview.jl")
include("dark_svg.jl")
include("cli.jl")

end
