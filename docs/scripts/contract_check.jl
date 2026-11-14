"""
Static documentation-contract checks. Uses only Julia standard libraries and never
loads the inspected module, expands its macros, or evaluates default expressions.

This deliberately checks a stricter, complete signature format than Julia's general
documentation style: the first Markdown code block must list the selected methods
with positional/keyword names, annotations, defaults, return type and `where` clauses.
Later example blocks are not signatures. One docstring may cover several overloads.

AST preservation means equality of parsed syntax after removing static docstrings
and line metadata; it is not a proof of arbitrary macro-generated runtime behavior.
Joint files are excluded from parsing and compared only by SHA-256.
"""
module DocContractCheck

using Markdown
using SHA

export ArgumentContract, MethodContract, method_contracts, check_doc_contracts,
       compare_code_ast, check_file, render_signature, main

struct ArgumentContract
    name::Any
    type_expr::Any
    default_expr::Any
    vararg::Bool
end

struct MethodContract
    name::String
    positional::Vector{ArgumentContract}
    keywords::Vector{ArgumentContract}
    return_type::Any
    where_parameters::Vector{Any}
    signature::Any
    line::Int
    source::String
    docstring::Union{Nothing,String}
end

isjoint(path) = occursin("joint", lowercase(basename(path)))
selected(name, names) = names === nothing || name in string.(names)
exprtext(x) = x === nothing ? "<absent>" : sprint(Base.show_unquoted, x)

function macro_name(x)
    x isa GlobalRef && return string(x.mod, '.', x.name)
    x isa Symbol && return string(x)
    x isa Expr && x.head === :. && return exprtext(x)
    return ""
end

function isdoc(x)
    x isa Expr && x.head === :macrocall || return false
    return macro_name(x.args[1]) in ("Core.@doc", "Base.@doc", "Base.Docs.@doc", "@doc")
end

function literal_doc(x)
    x isa String && return x
    if x isa Expr && x.head === :macrocall &&
       macro_name(x.args[1]) in ("@raw_str", "Base.@raw_str")
        payload = filter(a -> !(a isa LineNumberNode), x.args[2:end])
        length(payload) == 1 && payload[1] isa String && return payload[1]
    end
    throw(ArgumentError("dynamic/nonliteral docstring is unsupported; it is not evaluated or silently discarded"))
end

function docparts(x)
    payload = filter(a -> !(a isa LineNumberNode), x.args[2:end])
    length(payload) == 2 || throw(ArgumentError("unsupported @doc form; expected a static docstring and a target"))
    return literal_doc(payload[1]), payload[2]
end

# Preserve all non-doc macro names/arguments, constants, bodies and quoted ASTs.
# Do not macroexpand: expansion could execute arbitrary code from the inspected file.
function normalized_ast(x; strip_docs=false)
    if x isa Expr
        if x.head === :quote
            return Expr(:quote, (normalized_ast(a; strip_docs=false) for a in x.args if !(a isa LineNumberNode))...)
        end
        if strip_docs && isdoc(x)
            _, target = docparts(x)
            return normalized_ast(target; strip_docs)
        end
        return Expr(x.head, (normalized_ast(a; strip_docs) for a in x.args if !(a isa LineNumberNode))...)
    elseif x isa QuoteNode
        # Quoted syntax is program data; only line metadata may be ignored here.
        return QuoteNode(normalized_ast(x.value; strip_docs=false))
    end
    return x
end

function parse_source(path)
    ast = Meta.parseall(read(path, String); filename=String(path))
    function check_parse(x)
        x isa Expr || return nothing
        x.head in (:error, :incomplete) && throw(ArgumentError("cannot parse $path: $(exprtext(x))"))
        foreach(check_parse, x.args)
        return nothing
    end
    check_parse(ast)
    return ast
end

function split_signature(signature)
    current = signature
    where_parameters = Any[]
    return_type = nothing
    while current isa Expr && current.head in (:where, :(::))
        if current.head === :where
            append!(where_parameters, current.args[2:end])
        else
            length(current.args) == 2 || return nothing
            return_type = current.args[2]
        end
        current = current.args[1]
    end
    current isa Expr && current.head === :call || return nothing
    return (; call=current, return_type, where_parameters)
end

function argument_contract(x)
    default_expr = nothing
    if x isa Expr && x.head in (:kw, :(=))
        default_expr, x = x.args[2], x.args[1]
    end
    vararg = x isa Expr && x.head === :(...)
    vararg && (x = x.args[1])
    type_expr = nothing
    if x isa Expr && x.head === :(::)
        type_expr = x.args[end]
        x = length(x.args) == 1 ? nothing : x.args[1]
    end
    return ArgumentContract(x, type_expr, default_expr, vararg)
end

function make_contract(signature, source, line, docstring=nothing)
    parts = split_signature(signature)
    parts === nothing && return nothing
    name = parts.call.args[1]
    # Anonymous callable-object methods need a distinct dispatch contract. Fail
    # closed if selected indirectly instead of guessing a callable binding name.
    name isa Symbol || (name isa Expr && name.head in (:., :curly)) || return nothing
    positional = ArgumentContract[]
    keywords = ArgumentContract[]
    for arg in parts.call.args[2:end]
        if arg isa Expr && arg.head === :parameters
            append!(keywords, argument_contract.(arg.args))
        else
            push!(positional, argument_contract(arg))
        end
    end
    return MethodContract(exprtext(name), positional, keywords, parts.return_type,
        parts.where_parameters, signature, line, String(source), docstring)
end

"""Read syntactic named-method definitions without evaluating the inspected file."""
function method_contracts(path; names=nothing)
    isjoint(path) && throw(ArgumentError("joint files are hash-only and excluded from method analysis"))
    methods = MethodContract[]
    function visit(x, line=0, docstring=nothing)
        x isa Expr || return nothing
        if isdoc(x)
            doc, target = docparts(x)
            visit(target, line, doc)
        elseif x.head === :function || (x.head === :(=) && split_signature(x.args[1]) !== nothing)
            if length(x.args) >= 2 && x.args[2] isa Expr && x.args[2].head === :block
                first_line = findfirst(a -> a isa LineNumberNode, x.args[2].args)
                first_line === nothing || (line = x.args[2].args[first_line].line)
            end
            method = make_contract(x.args[1], path, line, docstring)
            method !== nothing && selected(method.name, names) && push!(methods, method)
            # Do not mistake local function definitions for module API overloads.
        elseif x.head in (:quote, :macro, :(->))
            return nothing
        else
            for child in x.args
                if child isa LineNumberNode
                    line = child.line
                else
                    visit(child, line, docstring)
                end
            end
        end
        return nothing
    end
    visit(parse_source(path))
    return methods
end

render_signature(method::MethodContract) = exprtext(normalized_ast(method.signature))

function documented_contracts(docstring, source, line)
    markdown = Markdown.parse(docstring)
    isempty(markdown.content) && throw(ArgumentError("empty docstring"))
    first(markdown.content) isa Markdown.Code || throw(ArgumentError(
        "docstring must start with an indented or fenced Julia signature block"))
    block = first(markdown.content)
    block.language in ("", "julia") || throw(ArgumentError("first code block is not a Julia signature block"))
    contracts = MethodContract[]
    position = firstindex(block.code)
    while position <= lastindex(block.code)
        parsed, next_position = Meta.parse(block.code, position; greedy=true, raise=true)
        if parsed !== nothing
            # The function context matters: an unwrapped `f(x)::T where T`
            # otherwise parses the where as part of the return type.
            raw = SubString(block.code, position, prevind(block.code, next_position))
            definition = Meta.parse("function " * raw * "\nend"; raise=true)
            definition isa Expr && definition.head === :function || throw(ArgumentError("not a complete method signature"))
            method = make_contract(definition.args[1], source, line)
            method === nothing && throw(ArgumentError("unsupported method signature: $raw"))
            push!(contracts, method)
        end
        position = next_position
    end
    isempty(contracts) && throw(ArgumentError("signature block contains no methods"))
    return contracts
end

same_expr(a, b) = isequal(normalized_ast(a), normalized_ast(b))

function argument_differences(actual, documented, label)
    errors = String[]
    for field in (:name, :type_expr, :default_expr, :vararg)
        a, d = getfield(actual, field), getfield(documented, field)
        same_expr(a, d) || push!(errors, "$label $field: source=$(exprtext(a)), doc=$(exprtext(d))")
    end
    return errors
end

function contract_differences(actual, documented)
    errors = String[]
    actual.name == documented.name || push!(errors, "binding: source=$(actual.name), doc=$(documented.name)")
    length(actual.positional) == length(documented.positional) || push!(errors,
        "positional count: source=$(length(actual.positional)), doc=$(length(documented.positional))")
    for (index, (a, d)) in enumerate(zip(actual.positional, documented.positional))
        append!(errors, argument_differences(a, d, "positional[$index]"))
    end
    # Keyword ordering is prose layout, but names/types/default expressions must match.
    actual_keys = Dict(exprtext(a.name) => a for a in actual.keywords)
    doc_keys = Dict(exprtext(a.name) => a for a in documented.keywords)
    length(doc_keys) == length(documented.keywords) || push!(errors, "duplicate documented keyword")
    for name in sort!(collect(union(keys(actual_keys), keys(doc_keys))))
        if !haskey(actual_keys, name)
            push!(errors, "extra documented keyword: $name")
        elseif !haskey(doc_keys, name)
            push!(errors, "missing documented keyword: $name")
        else
            append!(errors, argument_differences(actual_keys[name], doc_keys[name], "keyword $name"))
        end
    end
    same_expr(actual.return_type, documented.return_type) || push!(errors,
        "return type: source=$(exprtext(actual.return_type)), doc=$(exprtext(documented.return_type))")
    same_expr(Expr(:tuple, actual.where_parameters...), Expr(:tuple, documented.where_parameters...)) ||
        push!(errors, "where constraints differ: source=$(actual.where_parameters), doc=$(documented.where_parameters)")
    return errors
end

"""Require complete signature coverage for only the selected names in one source file."""
function check_doc_contracts(path; names=nothing)
    methods = method_contracts(path; names)
    errors = String[]
    documented = MethodContract[]
    isempty(methods) && push!(errors, "no selected named method definitions found in $path")
    seen_docs = Set{String}()
    for method in methods
        doc = method.docstring
        (doc === nothing || doc in seen_docs) && continue
        push!(seen_docs, doc)
        try
            append!(documented, documented_contracts(doc, path, method.line))
        catch err
            push!(errors, "$(method.name) at $(path):$(method.line): $(sprint(showerror, err))")
        end
    end
    for actual in methods
        candidates = filter(d -> d.name == actual.name, documented)
        if isempty(candidates)
            push!(errors, "missing documented method: $(render_signature(actual))")
            continue
        end
        differences = contract_differences.(Ref(actual), candidates)
        if !any(isempty, differences)
            nearest = differences[argmin(length.(differences))]
            push!(errors, "no matching documented method at $(path):$(actual.line): $(render_signature(actual)); " * join(nearest, "; "))
        end
    end
    for doc in documented
        candidates = filter(m -> m.name == doc.name, methods)
        # Other explicitly named APIs in a shared docstring are outside this check.
        selected(doc.name, names) || continue
        any(m -> isempty(contract_differences(m, doc)), candidates) || push!(errors,
            "documented signature has no matching source method: $(render_signature(doc))")
    end
    return (; ok=isempty(errors), source=String(path), checked_methods=length(methods),
        documented_signatures=length(documented), errors, methods, signatures=documented)
end

function first_ast_difference(a, b, path="root")
    isequal(a, b) && return nothing
    if a isa Expr && b isa Expr && a.head == b.head && length(a.args) == length(b.args)
        for index in eachindex(a.args)
            difference = first_ast_difference(a.args[index], b.args[index], "$path/$(a.head)[$index]")
            difference === nothing || return difference
        end
    end
    return "$path: source=$(exprtext(a)); baseline=$(exprtext(b))"
end

"""Compare full executable syntax, or raw hashes only when either file is joint."""
function compare_code_ast(source, baseline; joint=false)
    source_hash = bytes2hex(sha256(read(source)))
    baseline_hash = bytes2hex(sha256(read(baseline)))
    if joint || isjoint(source) || isjoint(baseline)
        equal = source_hash == baseline_hash
        return (; ok=equal, mode=:hash_only, source_hash, baseline_hash,
            errors=equal ? String[] : ["joint file bytes differ; no parsing or algorithm analysis performed"])
    end
    source_ast = normalized_ast(parse_source(source); strip_docs=true)
    baseline_ast = normalized_ast(parse_source(baseline); strip_docs=true)
    difference = first_ast_difference(source_ast, baseline_ast)
    return (; ok=difference === nothing, mode=:docstripped_ast, source_hash, baseline_hash,
        errors=difference === nothing ? String[] : [difference])
end

"""Combine selected signature checks with preservation against one baseline file."""
function check_file(source; baseline=nothing, names=nothing, joint=false)
    hash_only = joint || isjoint(source) || (baseline !== nothing && isjoint(baseline))
    hash_only && baseline === nothing && throw(ArgumentError("hash-only joint verification requires --baseline"))
    code = baseline === nothing ? nothing : compare_code_ast(source, baseline; joint=hash_only)
    docs = hash_only ? nothing : check_doc_contracts(source; names)
    errors = String[]
    code === nothing || append!(errors, code.errors)
    docs === nothing || append!(errors, docs.errors)
    return (; ok=isempty(errors), source=String(source), baseline, code, docs, errors)
end

function main(args=ARGS)
    source = baseline = nothing
    names = String[]
    joint = false
    index = 1
    while index <= length(args)
        option = args[index]
        if option == "--joint"
            joint = true
        elseif option in ("--source", "--baseline", "--function")
            index += 1
            index <= length(args) || throw(ArgumentError("missing value for $option"))
            value = args[index]
            if option == "--source"
                source = value
            elseif option == "--baseline"
                baseline = value
            else
                push!(names, value)
            end
        else
            throw(ArgumentError("unknown option: $option"))
        end
        index += 1
    end
    source === nothing && throw(ArgumentError("usage: julia contract_check.jl --source FILE --baseline FILE [--function NAME] [--joint]"))
    result = check_file(source; baseline, names=isempty(names) ? nothing : names, joint)
    println("source=", result.source)
    result.code === nothing || println("code_preserved=", result.code.ok, " mode=", result.code.mode)
    result.docs === nothing || println("documented_contracts=", result.docs.ok,
        " methods=", result.docs.checked_methods, " signatures=", result.docs.documented_signatures)
    foreach(error -> println(stderr, "ERROR: ", error), result.errors)
    return result.ok ? 0 : 1
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    exit(DocContractCheck.main())
end
