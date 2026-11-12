include("PublicationTools.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    try
        PublicationTools.command_main(ARGS) || exit(1)
    catch error
        error isa InterruptException && rethrow()
        showerror(stderr, error, catch_backtrace())
        println(stderr)
        exit(1)
    end
end
