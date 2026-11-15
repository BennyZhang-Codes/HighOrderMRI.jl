# Experimental pruned FFT execution

This is a default-off experiment for the existing NonuniformFFTs 0.9.6,
3D Float32 shared-factor batched path. It combines native scatter and gather
with a loop-unroll factor of 9, VkFFT pruning of known FFT padding, compact image
padding, and an optional budgeted Float32 coefficient cache. Shared LR and Joint factor
construction, the logical FFT grid, weights and normalization remain unchanged.

Build resources explicitly from the package root; package loading never invokes
the compiler:

```bash
HIGHORDERMRI_VKFFT_INCLUDE=/path/to/VkFFT/vkFFT \
HIGHORDERMRI_CUDA_TOOLKIT=/usr/local/cuda-12.4 \
HIGHORDERMRI_NATIVE_ARCH=sm_86 \
bash deps/pruned_fft/build.sh /path/to/experiment/resources
```

The tested VkFFT revision is `066a17c17068c0f11c9298d848c2976c71fad1c1`.
The bridge retains its internal experimental C ABI names. Build metadata records
source/header/binary hashes and compiler flags. Resources are local experiment
outputs, not portable release artifacts. No prebuilt binary is stored here.

Pass these options to `HighOrderLowRankOp`, together with existing physical and
factor-construction parameters:

```julia
normal_method=:batched,
normal_options=(
    fft=:vkfft_pruned,
    native_ptx="/path/to/experiment/resources/highorder_native.ptx",
    vkfft_library="/path/to/experiment/resources/libhighorder_vkfft.so",
    coefficient_cache_bytes=1 << 30,  # Per worker; zero disables the cache.
),
```

Batch sizes 2, 4 and 8 are built. A cache is allocated only when its exact
size fits the supplied budget; the worker's execution object records the actual
cache bytes. No interpolation coefficients are quantized. The noncached native
scatter remains available. Parent `close(op)` synchronizes each worker and
closes the foreign FFT plan and native module before stopping its workers.

Missing files, unsupported configurations and plan failures throw errors. The
options do not silently change the NUFFT library or select a coarser model.
`normal_options=(;)` retains Julia kernels and cuFFT, and `normal_method` still
defaults to `:standard`.

Native window and coefficient offsets must fit Int32; unsupported ranges are
rejected before allocating the native metadata. The opt-in GPU tests in
`test/HighOrderLowRank/test_pruned_normal.jl` cover independent FP64 FFT sums,
weighted normals and adjoints, coil tails, multiple GPUs, cross-context close,
and recovery from failed native setup. Run the test suite in an environment
with NonuniformFFTs 0.9.6 installed, setting `HIGHORDERMRI_TEST_PRUNED_NORMAL=true`,
`HIGHORDERMRI_NATIVE_PTX` and `HIGHORDERMRI_VKFFT_LIBRARY` to the built resources,
and `HIGHORDERMRI_TEST_GPUS` to the visible CUDA device ids to test.
