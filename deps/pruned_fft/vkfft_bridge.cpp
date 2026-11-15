#define VKFFT_BACKEND 1
#include "vkFFT.h"
#include <new>

struct JointVkFFT {
    VkFFTApplication app{};
    VkFFTConfiguration config{};
    CUdevice device{};
    cudaStream_t stream{};
    void* buffer{};
    pfUINT bytes{};
};
extern "C" void* joint_vkfft_create(uint64_t nx,uint64_t ny,uint64_t nz,uint64_t batch,
        void* buffer,void* stream,int threads,int boost,int lut,
        const uint64_t* left,const uint64_t* right,int* result) {
    JointVkFFT* p=new(std::nothrow) JointVkFFT;
    if(!p) {*result=-1;return nullptr;}
    CUresult cuda_result=cuCtxGetDevice(&p->device);
    if(cuda_result!=CUDA_SUCCESS) {*result=-10000-int(cuda_result);delete p;return nullptr;}
    p->buffer=buffer;p->stream=reinterpret_cast<cudaStream_t>(stream);
    p->bytes=8*nx*ny*nz*batch;
    auto& c=p->config;
    c.FFTdim=3;c.size[0]=nx;c.size[1]=ny;c.size[2]=nz;c.numberBatches=batch;
    c.device=&p->device;c.stream=&p->stream;c.num_streams=1;
    c.buffer=&p->buffer;c.bufferSize=&p->bytes;
    c.normalize=0;c.doublePrecision=0;c.halfPrecision=0;c.disableSetLocale=1;
    if(threads>0)c.aimThreads=threads;
    if(boost>0){c.registerBoost=boost;c.registerBoostNonPow2=1;}
    c.useLUT=lut;
    if(left&&right) {
        c.frequencyZeroPadding=1;
        for(int d=0;d<3;++d) {
            c.performZeropadding[d]=right[d]>left[d];
            c.fft_zeropad_left[d]=left[d];c.fft_zeropad_right[d]=right[d];
        }
    }
    *result=int(initializeVkFFT(&p->app,c));
    // initializeVkFFT releases its own partial application on failure.
    if(*result!=0){delete p;return nullptr;}
    return p;
}
extern "C" int joint_vkfft_execute(void* handle,void* buffer,void* stream,int direction) {
    auto* p=static_cast<JointVkFFT*>(handle);
    p->stream=reinterpret_cast<cudaStream_t>(stream);p->buffer=buffer;
    VkFFTLaunchParams launch{};launch.buffer=&p->buffer;
    // VkFFT uses -1 for forward and +1 for unnormalized inverse, like cuFFT.
    return int(VkFFTAppend(&p->app,direction,&launch));
}
extern "C" void joint_vkfft_destroy(void* handle) {
    if(handle){auto* p=static_cast<JointVkFFT*>(handle);deleteVkFFT(&p->app);delete p;}
}
