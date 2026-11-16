#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out_dir=${1:?Pass an output directory for experimental resources}
cuda_dir=${HIGHORDERMRI_CUDA_TOOLKIT:-/usr/local/cuda-12.4}
include_dir=${HIGHORDERMRI_VKFFT_INCLUDE:?Pass the include directory of the pinned VkFFT checkout}
architecture=${HIGHORDERMRI_NATIVE_ARCH:-sm_86}
mkdir -p "$out_dir"
"$cuda_dir/bin/nvcc" --ptx -std=c++17 -arch="$architecture" -O3 --ftz=false --fmad=true --prec-div=true --prec-sqrt=true \
    "$source_dir/highorder_native.cu" -o "$out_dir/highorder_native.ptx"
g++ -std=c++17 -O3 -fPIC -shared -I"$include_dir" -I"$cuda_dir/include" "$source_dir/vkfft_bridge.cpp" \
    -L"$cuda_dir/lib64" -Wl,-rpath,"$cuda_dir/lib64" -lcuda -lnvrtc -lcudart -o "$out_dir/libhighorder_vkfft.so"
python3 - "$source_dir" "$out_dir" "$include_dir" "$cuda_dir" "$architecture" <<'PY'
from pathlib import Path
import hashlib,json,subprocess,sys
source,out,include,cuda=map(Path,sys.argv[1:5]);arch=sys.argv[5]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
record=dict(native_arch=arch,source_sha256={p.name:digest(p) for p in source.iterdir() if p.is_file()},
    binary_sha256={p.name:digest(p) for p in out.iterdir() if p.suffix in ['.ptx','.so']},
    VkFFT_commit=subprocess.check_output(['git','-C',str(include.parent),'rev-parse','HEAD'],text=True).strip(),
    VkFFT_headers_sha256={str(p.relative_to(include)):digest(p) for p in include.rglob('*.h')},
    nvcc_version=subprocess.check_output([str(cuda/'bin/nvcc'),'--version'],text=True),
    host_compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
    native_flags='--ptx -std=c++17 -O3 --ftz=false --fmad=true --prec-div=true --prec-sqrt=true',
    precision='ComplexF32; unnormalized FFT in both directions; original Float32 windows')
(out/'build_metadata.json').write_text(json.dumps(record,indent=2)+'\n')
PY
