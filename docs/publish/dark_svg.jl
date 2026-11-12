const DARK_PALETTE = Dict(
    "#000000" => "#e6e6eb", "#202020" => "#e6e6eb", "#545454" => "#b9bbc6", "#555555" => "#b9bbc6",
    "#355fa0" => "#8fb8ff", "#167d78" => "#69d0bd", "#8a2436" => "#f19aab", "#a45e20" => "#e9b577",
    "#ffffff" => "#23242b", "#f7f7f7" => "#24252c", "#eaf5f2" => "#193d38", "#deedec" => "#193d38",
    "#f7f1eb" => "#3a2c22", "#909090" => "#777b89", "#bcbcbc" => "#4e515e", "#d7d7d7" => "#4e515e",
    "#509d9a" => "#5fa89e", "#a2cbc9" => "#3d7970", "#b9d8d6" => "#3d7970",
)

function recolor_svg(value)
    property = match(r"\A(fill|stroke):([^;]+);\z", value)
    color = property[2]
    color == "none" && return value
    parts = match(r"\Argb\(([\d.]+)%,([\d.]+)%,([\d.]+)%\)\z", color)
    require(parts !== nothing, "Unrecognized SVG color: $color")
    hex = "#" * join(string(round(Int, parse(Float64, part) * 255 / 100); base=16, pad=2) for part in parts.captures)
    require(haskey(DARK_PALETTE, hex), "Add an explicit dark palette mapping for $hex")
    return "$(property[1]):$(DARK_PALETTE[hex]);"
end

function dark_svg(original)
    document = EzXML.parsexml(original)
    bounds = split(EzXML.root(document)["viewBox"])
    require(length(bounds) == 4, "Expected a four-component SVG viewBox")
    x, y, width, height = bounds
    themed = replace(original, r"(fill|stroke):([^;]+);" => recolor_svg)
    background = "<rect x=\"$x\" y=\"$y\" width=\"$width\" height=\"$height\" fill=\"#1b1b1f\"/>"
    themed = replace(themed, r"(<svg\b[^>]*)(>)" => s"\1 style=\"fill:#e6e6eb;\"\2"; count=1)
    themed = replace(themed, r"<svg\b[^>]*>" => tag -> tag * "\n" * background; count=1)
    EzXML.parsexml(themed)
    return themed
end

function build_dark_svg(source, target)
    # Transform the original bytes, rather than reserializing paths or glyphs.
    write(target, dark_svg(read(source, String)))
    println(target)
    return nothing
end
