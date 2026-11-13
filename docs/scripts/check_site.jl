include(joinpath(@__DIR__, "..", "publish", "PublicationTools.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    try
        PublicationTools.require(1 <= length(ARGS) <= 4, "Expected DOCS [HTML_DIR BASE VERSION]")
        docs = ARGS[1]
        dist = length(ARGS) >= 2 ? ARGS[2] : joinpath(docs, ".pages", "dev")
        base = length(ARGS) >= 3 ? ARGS[3] : "/HighOrderMRI.jl/dev/"
        version = length(ARGS) >= 4 ? ARGS[4] : "dev"
        report = PublicationTools.check_site(docs, dist; base, version)
        println(PublicationTools.json_text(report))
        isempty(report["errors"]) || exit(1)
    catch error
        showerror(stderr, error, catch_backtrace())
        println(stderr)
        exit(1)
    end
end
