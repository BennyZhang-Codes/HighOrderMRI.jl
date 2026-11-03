import Functors: @functor, functor, fmap, isleaf
import Adapt: adapt, adapt_storage
#Checks if CUDA is available for the session
const use_cuda = Ref{Union{Nothing,Bool}}(nothing)

"""
    print_gpus()

Internal diagnostic: check cached CUDA availability and log visible devices.
The listed IDs are zero based. This helper is not exported.
"""
function print_gpus()
    check_use_cuda()
    if use_cuda[]
	    cuda_devices = [Symbol("($(i-1)$(i == 1 ? "*" : " "))") => name(d) for (i,d) = enumerate(devices())]
		@info "$(length(devices())) CUDA capable device(s)." cuda_devices...
    else
        @info "0 CUDA capable devices(s)."
    end
	return
end

"""
    check_use_cuda()

Internal helper that initializes the cached result of `CUDA.functional()` on
first use. Later calls reuse that result. It is not exported and does not
transfer arrays or select a device.
"""
function check_use_cuda()
	if use_cuda[] === nothing
		use_cuda[] = CUDA.functional()
		if !(use_cuda[])
		@info """The GPU function is being called but the GPU is not accessible.
					Defaulting back to the CPU. (No action is required if you want to run on the CPU).""" maxlog=1
		end
	end
end

#Aux. funcitons to check if the variable we want to convert to CuArray is numeric
_isbitsarray(::AbstractArray{<:Real}) = true
_isbitsarray(::AbstractArray{T}) where T = isbitstype(T)
_isbitsarray(x) = false
_isleaf(x) = _isbitsarray(x) || isleaf(x)

# GPU adaptor
struct CUDAAdaptor end
adapt_storage(to::CUDAAdaptor, x) = CUDA.cu(x)

"""
    gpu(x)

Move supported numeric leaves to the current CUDA device using the package's
Functors/Adapt traversal. If CUDA is unavailable, return `x` unchanged and cache
that availability result. This fallback is not an explicit device-selection API.

The adaptor calls `CUDA.cu`, which can convert floating-point precision. Use an
explicit `CuArray` conversion when preserving a chosen element type matters.
Traversal follows Functors support; it is not an unconditional deep copy of an
arbitrary object. No GPU is selected by this function.
"""
function gpu(x)
	check_use_cuda()
	use_cuda[] ? fmap(x -> adapt(CUDAAdaptor(), x), x; exclude = _isleaf) : x
end

#CPU adaptor
struct CPUAdaptor end
adapt_storage(to::CPUAdaptor, x::AbstractArray) = adapt(Array, x)
adapt_storage(to::CPUAdaptor, x::AbstractRange) = x

# To CPU
"""
    cpu(x)

Adapt supported array leaves to ordinary CPU arrays through Functors/Adapt.
Ranges are retained. The result follows the input object's supported traversal;
already-host storage need not be copied independently. This function does not
change numeric precision deliberately.
"""
cpu(x) = fmap(x -> adapt(CPUAdaptor(), x), x)

#Precision
paramtype(T::Type{<:Real}, m) = fmap(x -> adapt(T, x), m)

adapt_storage(T::Type{<:Real}, xs::AbstractArray{<:Real}) = convert.(T, xs) #Type piracy
adapt_storage(T::Type{<:Real}, xs::AbstractArray{<:Complex}) = convert.(Complex{T}, xs) #Type piracy
adapt_storage(T::Type{<:Real}, xs::AbstractArray{<:Bool}) = xs #Type piracy

"""
    f32(m)

Convert supported real numeric array leaves to `Float32` and complex leaves to
`ComplexF32` through Functors/Adapt. Boolean arrays remain unchanged. This is a
precision conversion, not a CPU/GPU transfer; traversal follows the supported
object structure.
"""
f32(m) = paramtype(Float32, m)

"""
    f64(m)

Convert supported real numeric array leaves to `Float64` and complex leaves to
`ComplexF64` through Functors/Adapt. Boolean arrays remain unchanged. This is a
precision conversion, not a CPU/GPU transfer; traversal follows the supported
object structure.
"""
f64(m) = paramtype(Float64, m)

#Exporting functions
export gpu, cpu, f32, f64
