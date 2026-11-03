using Test
include(joinpath(@__DIR__, "contract_check.jl"))
using .DocContractCheck

const CONTRACT_CODE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
println("CONTRACT_CHECK_JULIA=", VERSION)
println("CONTRACT_CHECK_GPU_ACCESS=none; standard libraries only")

function fixture(signature, body="x"; prose="Explain the method.")
    doc = "    " * replace(signature, "\n" => "\n    ") * "\n\n" * prose
    return repr(doc) * "\nfunction " * signature * "\n    " * body * "\nend\n"
end

@testset "Static contract preservation" begin
    mktempdir() do directory
        source = joinpath(directory, "candidate.jl")
        baseline = joinpath(directory, "baseline.jl")
        # A live default would create a marker. Parsing must never do so, even
        # when it encounters this expression in both code and documentation.
        marker = joinpath(directory, "default_was_executed")
        signature = "f(x::T; delta::T=0.005, payload=(write($(repr(marker)), \"executed\"); (x + 1)^2))::T where {T<:Real}"
        original = fixture(signature)
        write(baseline, original)
        write(source, fixture(signature; prose="Revised explanation with an example.\n\n```julia\nf(99)\n```"))
        result = check_file(source; baseline, names=[:f])
        @test result.ok
        @test result.docs.checked_methods == 1
        @test result.docs.documented_signatures == 1
        @test !ispath(marker)
        @test result.docs.methods[1].keywords[1].default_expr === 0.005
        @test result.docs.methods[1].keywords[2].default_expr isa Expr

        # Documentation can agree with an unauthorized default mutation. The
        # independent full-source comparison must still reject that change.
        write(source, replace(original, "delta::T=0.005" => "delta::T=T(0.005)"))
        @test check_doc_contracts(source; names=[:f]).ok
        @test !compare_code_ast(source, baseline).ok
        @test !check_file(source; baseline, names=[:f]).ok

        # Change only the documented expression, preserving executable source.
        stale = replace(original, "delta::T=0.005" => "delta::T=T(0.005)"; count=1)
        write(source, stale)
        mismatch = check_doc_contracts(source; names=[:f])
        @test !mismatch.ok
        @test any(error -> occursin("default_expr", error), mismatch.errors)
        @test compare_code_ast(source, baseline).ok
        @test !ispath(marker)

        # Expressions are compared structurally rather than executed/simplified.
        write(source, replace(original, "(x + 1)^2" => "x + 1^2"; count=1))
        @test !check_doc_contracts(source; names=[:f]).ok
        write(source, replace(original, "T<:Real" => "T<:AbstractFloat"; count=1))
        @test !check_doc_contracts(source; names=[:f]).ok
        write(source, replace(original, "delta::T" => "delta::Float64"; count=1))
        @test !check_doc_contracts(source; names=[:f]).ok

        # Braced and unbraced single where variables are ordinary equivalent syntax.
        write(source, repr("    g(x::T) where T\n\nA method.") * "\ng(x::T) where {T} = x\n")
        @test check_doc_contracts(source; names=[:g]).ok

        two_signatures = "    h(x::T; scale::Int=2)::T where {T<:Real}\n\n    h(x::T, y::T; scale::Int=2)::T where {T<:Real}\n\nTwo overloads."
        two_bodies = "function h(x::T; scale::Int=2)::T where {T<:Real}\n x\nend\nfunction h(x::T, y::T; scale::Int=2)::T where {T<:Real}\n x+y\nend\n"
        write(baseline, repr(two_signatures) * "\n" * two_bodies)
        write(source, read(baseline, String))
        @test check_file(source; baseline, names=[:h]).ok
        @test length(method_contracts(source; names=[:h])) == 2
        # Omitted documentation and omitted implementation are distinct failures.
        only_one_doc = "    h(x::T; scale::Int=2)::T where {T<:Real}\n\nOne overload."
        write(source, repr(only_one_doc) * "\n" * two_bodies)
        @test !check_doc_contracts(source; names=[:h]).ok
        @test compare_code_ast(source, baseline).ok
        write(source, repr(two_signatures) * "\nfunction h(x::T; scale::Int=2)::T where {T<:Real}\n x\nend\n")
        @test !check_doc_contracts(source; names=[:h]).ok
        @test !compare_code_ast(source, baseline).ok

        # Body literals and non-documentation macros must survive normalization.
        macro_source = fixture("m(x)", "@inbounds x[1]")
        write(baseline, macro_source)
        write(source, "\n# shifted source locations\n" * replace(macro_source, "Explain the method." => "New prose."))
        @test compare_code_ast(source, baseline).ok
        write(source, replace(macro_source, "@inbounds" => "@boundscheck"))
        @test !compare_code_ast(source, baseline).ok
        write(baseline, fixture("m(x)", "\"runtime literal\""))
        write(source, replace(read(baseline, String), "runtime literal" => "changed literal"))
        @test !compare_code_ast(source, baseline).ok
        quoted = "const template = quote\n    \"quoted doc as executable data\"\n    q(x) = x\nend\n"
        write(baseline, quoted)
        write(source, replace(quoted, "quoted doc as executable data" => "different data"))
        @test !compare_code_ast(source, baseline).ok

        # Dynamic docs are rejected, never evaluated or silently discarded.
        dynamic = "\"Doc \$(write($(repr(marker)), \"bad\"))\"\nf(x) = x\n"
        write(source, dynamic)
        @test_throws ArgumentError method_contracts(source; names=[:f])
        @test_throws ArgumentError compare_code_ast(source, baseline)
        @test !ispath(marker)

        # Even invalid Julia bytes in an excluded joint file are hash-only.
        joint = joinpath(directory, "joint_example.jl")
        write(joint, "this intentionally cannot be parsed as Julia: (((")
        write(baseline, read(joint))
        hash_result = check_file(joint; baseline)
        @test hash_result.ok
        @test hash_result.code.mode === :hash_only
        @test hash_result.docs === nothing
        @test_throws ArgumentError method_contracts(joint)
        write(joint, read(joint, String) * " changed")
        @test !compare_code_ast(joint, baseline).ok
    end
end

@testset "Selected checkout FindDelay contract" begin
    source = joinpath(CONTRACT_CODE_ROOT, "src", "Synchronization", "FindDelay.jl")
    @test isfile(source)
    result = check_doc_contracts(source; names=[:FindDelay])
    @test result.ok
    @test result.checked_methods == 2
    @test result.documented_signatures == 2
    for method in result.methods
        delta = only(filter(arg -> arg.name === :Δτ_min, method.keywords))
        @test delta.default_expr === 0.005
        @test delta.type_expr === :T
    end
end
