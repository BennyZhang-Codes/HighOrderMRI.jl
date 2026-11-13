"""Prepare the 0.5-mm calibration grid for the existing single-readout loader."""

using ImageTransformations: imresize
using Interpolations: BSpline, Linear, Constant
using MAT

highres_scalar(x::Number) = Float64(x)
highres_scalar(x::AbstractArray) = Float64(only(x))

function prepare_highres_input(source_file::AbstractString, output_file::AbstractString)
    ispath(output_file) && realpath(source_file) == realpath(output_file) &&
        throw(ArgumentError("Prepared output must not overwrite the original input"))
    raw_highres = matread(source_file)
    matrix = Int.(vec(raw_highres["matrixSize"]))
    matrix == [400, 400, 1] || error("Expected a 400×400 single-slice acquisition")
    target = Tuple(matrix[1:2])
    ns, nc = size(raw_highres["kdata"])
    nc == 32 || error("Expected 32 physical receive channels")
    times = Float64.(vec(raw_highres["datatime"]))
    length(times) == ns || error("ADC times/data size mismatch")
    all(diff(times) .> 0) || error("ADC times must be strictly increasing")
    fields = Matrix{Float64}(raw_highres["kspha"])
    size(fields, 2) == 16 || error("Expected zeroth-through-third-order fields")
    length(vec(raw_highres["k0_adc"])) == ns || error("ECC phase/data size mismatch")

    csm_original = raw_highres["gre_csm"]
    b0_original = raw_highres["gre_b0"]
    mask_original = raw_highres["gre_mask"]
    size(csm_original) == (200, 200, nc) || error("Unexpected calibration grid")
    csm_resized = Array{ComplexF32}(undef, target..., nc)
    for channel in 1:nc
        csm_resized[:, :, channel] = imresize(csm_original[:, :, channel], target;
            method=BSpline(Linear()))
    end
    b0_resized = Float32.(imresize(b0_original, target; method=BSpline(Linear())))
    mask_resized = imresize(Float32.(mask_original), target;
        method=BSpline(Constant())) .>= 0.5f0

    prepared = Dict{String,Any}(
        "gre_csm" => csm_resized,
        "gre_b0" => b0_resized,
        "gre_mask" => mask_resized,
        "kdata" => raw_highres["kdata"],
        "datatime" => times,
        "matrixSize" => matrix,
        "FOV" => Float64.(vec(raw_highres["FOV"])),
        "k0_adc" => reshape(Float64.(vec(raw_highres["k0_adc"])), 1, ns),
        "dt_Measured" => highres_scalar(raw_highres["dt"]),
        "ksphaMeasured" => fields,
        "startMeasured" => highres_scalar(raw_highres["start"]),
        "tauMeasured" => highres_scalar(raw_highres["tau"]),
        "dt_adc" => highres_scalar(raw_highres["dt_adc"]),
    )
    mkpath(dirname(output_file))
    matwrite(output_file, prepared; compress=true)
    return output_file
end


function prepare_highres_benchmark!()
    sequence = lowercase(get(ENV, "HIGHORDER_HIGHRES_SEQUENCE", "spiral"))
    sequence in ("spiral", "epi") || throw(ArgumentError("HIGHORDER_HIGHRES_SEQUENCE must be spiral or epi"))
    source_root = get(ENV, "HIGHORDER_HIGHRES_SOURCE_ROOT",
        "/home/jyzhang/Desktop/pulseq/20250814_Terra_generalStitch/data/invivo")
    filename = sequence == "spiral" ?
        "7T_Spiral_0p5_400_r4_seg16_max49.mat" : "7T_EPI_0p5_400_r5_pf0p75_seg13.mat"
    source_file = get(ENV, "HIGHORDER_HIGHRES_SOURCE", joinpath(source_root, filename))
    isfile(source_file) || throw(ArgumentError("High-resolution source file not found: $source_file"))
    output_file = get(ENV, "HIGHORDER_HIGHRES_PREPARED", "")
    if isempty(output_file)
        output_file = joinpath(mktempdir(; prefix="highordermri-highres-"),
            "HighOrderMRI_$(sequence)_0p5_prepared.mat")
    end
    prepare_highres_input(source_file, output_file)
    ENV["HIGHORDER_BENCHMARK_DATA"] = output_file
    get!(ENV, "HIGHORDER_BENCHMARK_GPUS", "2,3,4,5,6")
    get!(ENV, "HIGHORDER_BENCHMARK_RSVD_CHUNK", "4096")
    return output_file
end
