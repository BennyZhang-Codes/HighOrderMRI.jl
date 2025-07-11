include("basisfunction.jl")
export basisfunc_spha


export spha_color, spha_basis, spha_basis_latex
export SphericalHarmonics
spha_color = [
        "#5f4690", 
        "#1d6996", "#38a6a5", "#0f8554", 
        "#73af48", "#edad08", "#e17c05", "#cc503e", "#94346e", 
        "#6f4070", "#666666", "#527f99", "#6db88c", "#b6890e", "#a85144", "#875573"
    ];
spha_basis = [
        "1", 
        "x", "y", "z", 
        "xy", "zy", "3z²-(x²+y²+z²)", "xz", "x²-y²",
        "3yx²-y³", "xzy", "(5z²-(x²+y²+z²))y", "5z³-3z(x²+y²+z²)", "(5z²-(x²+y²+z²))x", "x²z-y²z", "x³-3xy²"
    ];
spha_basis_latex = [
        L"1", 
        L"x", L"y", L"z", 
        L"xy", L"zy", L"3z^2-(x^2+y^2+z^2)", L"xz", L"x^2-y^2",
        L"3yx^2-y^3", L"xzy", L"(5z^2-(x^2+y^2+z^2))y", L"5z^3-3z(x^2+y^2+z^2)", L"(5z^2-(x^2+y^2+z^2))x", L"x^2z-y^2z", L"x^3-3xy^2"
    ];

struct sphericalharmonic
    name::String
    unit::String
    cumtrapz_unit::String
    expression::String
    color::String
end
Base.show(io::IO, s::sphericalharmonic) = print(io, s.name, " [", s.unit, "] = ", s.expression)

Base.@kwdef struct SphericalHarmonics
    h0  :: sphericalharmonic = sphericalharmonic("h0" ,    "T",    "",                    "1", "#5f4690")
    h1  :: sphericalharmonic = sphericalharmonic("h1" ,  "T/m", "m⁻¹",                    "x", "#1d6996")
    h2  :: sphericalharmonic = sphericalharmonic("h2" ,  "T/m", "m⁻¹",                    "y", "#38a6a5")
    h3  :: sphericalharmonic = sphericalharmonic("h3" ,  "T/m", "m⁻¹",                    "z", "#0f8554")
    h4  :: sphericalharmonic = sphericalharmonic("h4" , "T/m²", "m⁻²",                   "xy", "#73af48")
    h5  :: sphericalharmonic = sphericalharmonic("h5" , "T/m²", "m⁻²",                   "zy", "#edad08")
    h6  :: sphericalharmonic = sphericalharmonic("h6" , "T/m²", "m⁻²",       "3z²-(x²+y²+z²)", "#e17c05")
    h7  :: sphericalharmonic = sphericalharmonic("h7" , "T/m²", "m⁻²",                   "xz", "#cc503e")
    h8  :: sphericalharmonic = sphericalharmonic("h8" , "T/m²", "m⁻²",                "x²-y²", "#94346e")
    h9  :: sphericalharmonic = sphericalharmonic("h9" , "T/m³", "m⁻³",              "3yx²-y³", "#6f4070")
    h10 :: sphericalharmonic = sphericalharmonic("h10", "T/m³", "m⁻³",                  "xzy", "#666666")
    h11 :: sphericalharmonic = sphericalharmonic("h11", "T/m³", "m⁻³",    "(5z²-(x²+y²+z²))y", "#527f99")
    h12 :: sphericalharmonic = sphericalharmonic("h12", "T/m³", "m⁻³",     "5z³-3z(x²+y²+z²)", "#6db88c")
    h13 :: sphericalharmonic = sphericalharmonic("h13", "T/m³", "m⁻³",    "(5z²-(x²+y²+z²))x", "#b6890e")
    h14 :: sphericalharmonic = sphericalharmonic("h14", "T/m³", "m⁻³",              "x²z-y²z", "#a85144")
    h15 :: sphericalharmonic = sphericalharmonic("h15", "T/m³", "m⁻³",              "x³-3xy²", "#875573")
    dict::Dict =Dict(
        "h0"=>h0, 
        "h1"=>h1, "h2"=>h2, "h3"=>h3, 
        "h4"=>h4, "h5"=>h5, "h6"=>h6, "h7"=>h7, "h8"=>h8,
        "h9"=>h9, "h10"=>h10, "h11"=>h11, "h12"=>h12, "h13"=>h13, "h14"=>h14, "h15"=>h15,
        )
end

Base.show(io::IO, s::SphericalHarmonics) = begin
    print(io, "Spherical Harmonics")
    print(io, "\n  ", s.h0)
    print(io, "\n  ", s.h1)
    print(io, "\n  ", s.h2)
    print(io, "\n  ", s.h3)
    print(io, "\n  ", s.h4)
    print(io, "\n  ", s.h5)
    print(io, "\n  ", s.h6)
    print(io, "\n  ", s.h7)
    print(io, "\n  ", s.h8)
    print(io, "\n  ", s.h9) 
    print(io, "\n  ", s.h10)
    print(io, "\n  ", s.h11)
    print(io, "\n  ", s.h12)
    print(io, "\n  ", s.h13)
    print(io, "\n  ", s.h14)
    print(io, "\n  ", s.h15)
end