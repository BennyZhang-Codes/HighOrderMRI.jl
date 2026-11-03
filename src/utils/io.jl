"""
    struct_to_dict(x)

Return a dictionary whose keys are string field names and whose values are
`getfield(x, name)`. The conversion is shallow: array or mutable field values
remain shared with `x`. It does not recursively serialize nested objects.
"""
struct_to_dict(x) = Dict(string(name) => getfield(x, name) for name in fieldnames(typeof(x)))