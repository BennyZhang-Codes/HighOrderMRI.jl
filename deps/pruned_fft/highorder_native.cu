#include <cuda_runtime.h>
#include <stdint.h>

template<int B, bool Recurrence,int Unroll=1>
__device__ __forceinline__ void gather_body(float2* __restrict__ samples,
        const float2* __restrict__ grid,const int32_t* __restrict__ starts,
        const float* __restrict__ windows,const int32_t* __restrict__ permutation,
        int nx,int ny,int nz,int nt,int nf,float prefactor) {
    int t=blockIdx.x*blockDim.x+threadIdx.x,p=t/8,lane=t%8;
    if(p>=nt) return;
    int sx=starts[p]-3,sy=starts[p+nt]-3,sz=starts[p+2*nt]-3;
    int aa=lane%6,bb=lane/6,cc=0;
    float2 sum[B];
    #pragma unroll
    for(int coil=0;coil<B;++coil) sum[coil]=make_float2(0.f,0.f);
    #pragma unroll (Unroll)
    for(int iter=0;iter<27;++iter) {
        int k=lane+8*iter;
        int a=Recurrence ? aa : k%6;
        int b=Recurrence ? bb : (k/6)%6;
        int c=Recurrence ? cc : k/36;
        int x=sx+a+1,y=sy+b+1,z=sz+c+1;
        x=x<1 ? x+nx : (x>nx ? x-nx : x);
        y=y<1 ? y+ny : (y>ny ? y-ny : y);
        z=z<1 ? z+nz : (z>nz ? z-nz : z);
        int index=(x-1)+nx*((y-1)+ny*(z-1));
        float factor=(windows[p+nt*a]*windows[p+nt*(b+6)])*windows[p+nt*(c+12)];
        #pragma unroll
        for(int coil=0;coil<B;++coil) {
            float2 value=grid[index+nf*coil];
            sum[coil].x=fmaf(value.x,factor,sum[coil].x);
            sum[coil].y=fmaf(value.y,factor,sum[coil].y);
        }
        if(Recurrence) {
            int nexta=aa+2,carry=nexta>=6 ? 2 : 1;
            aa=nexta>=6 ? nexta-6 : nexta;
            int nextb=bb+carry;
            cc+=nextb>=6;bb=nextb>=6 ? nextb-6 : nextb;
        }
    }
    #pragma unroll
    for(int step=4;step>0;step/=2) {
        #pragma unroll
        for(int coil=0;coil<B;++coil) {
            sum[coil].x+=__shfl_xor_sync(0xffu << (8*((threadIdx.x%32)/8)),sum[coil].x,step,8);
            sum[coil].y+=__shfl_xor_sync(0xffu << (8*((threadIdx.x%32)/8)),sum[coil].y,step,8);
        }
    }
    if(lane==0) {
        int j=permutation[p]-1;
        #pragma unroll
        for(int coil=0;coil<B;++coil)
            samples[j+nt*coil]=make_float2(sum[coil].x*prefactor,sum[coil].y*prefactor);
    }
}

template<int B,int P,bool Weighted=false,int S=4,bool Masked=false>
__device__ __forceinline__ void spread_body(float2* __restrict__ grid,
        const float2* __restrict__ samples,const int32_t* __restrict__ point_ids,
        const int32_t* __restrict__ chunks,const void* __restrict__ origins,
        const float* __restrict__ windows,const int32_t* __restrict__ permutation,
        int nchunks,int nx,int ny,int nz,int nt,int nf,
        const float2* __restrict__ q=nullptr,const float* __restrict__ D=nullptr,int rank=0) {
    constexpr int fields=20+2*B;
    extern __shared__ __align__(16) float tile_values[];
    int chunk=blockIdx.x,tid=threadIdx.x;
    int first=chunks[chunk]-1,last=chunks[chunk+nchunks]-1;
    int tile=chunks[chunk+2*nchunks]-1;
    bool single=chunks[chunk+3*nchunks]!=0;
    int tx=(nx+S-1)/S,ty=(ny+S-1)/S;
    int ox=(tile%tx)*S,oy=((tile/tx)%ty)*S,oz=(tile/(tx*ty))*S;
    int cx=tid%S,cy=(tid/S)%S,cz=tid/(S*S);
    uint32_t required=(1u<<(12+cx))|(1u<<(16+cy))|(1u<<(20+cz));
    int x=ox+cx,y=oy+cy,z=oz+cz;
    bool valid=tid<S*S*S && x<nx && y<ny && z<nz;
    int output=x+nx*(y+ny*z);
    float2 sum[B];
    #pragma unroll
    for(int coil=0;coil<B;++coil) sum[coil]=make_float2(0.f,0.f);
    for(int base=first;base<=last;base+=P) {
        int count=min(P,last-base+1);
        for(int local=tid;local<count;local+=blockDim.x) {
            int entry=base+local,p=point_ids[entry]-1,offset=local*fields;
            uint32_t origin=Masked ? reinterpret_cast<const uint32_t*>(origins)[entry] :
                                    reinterpret_cast<const uint16_t*>(origins)[entry];
            tile_values[offset]=__uint_as_float(origin);
            #pragma unroll
            for(int d=0;d<3;++d) {
                #pragma unroll
                for(int k=0;k<6;++k)
                    tile_values[offset+1+6*d+k]=windows[p+nt*(k+6*d)];
            }
            int j=permutation[p]-1;
            float weight=Weighted ? D[j] : 1.f;
            float2 temporal=Weighted ? q[j+nt*rank] : make_float2(1.f,0.f);
            #pragma unroll
            for(int coil=0;coil<B;++coil) {
                float2 value=samples[j+nt*coil];
                if(Weighted) {
                    value.x*=weight;value.y*=weight;
                    value=make_float2(value.x*temporal.x+value.y*temporal.y,
                                     value.y*temporal.x-value.x*temporal.y);
                }
                tile_values[offset+20+2*coil]=value.x;
                tile_values[offset+21+2*coil]=value.y;
            }
        }
        __syncthreads();
        #pragma unroll 1
        for(int local=0;local<count;++local) {
            int offset=local*fields;
            uint32_t bits=__float_as_uint(tile_values[offset]);
            if(Masked && (bits&required)!=required) continue;
            int a=int(bits&15)+cx-(S-2),b=int((bits>>4)&15)+cy-(S-2),c=int((bits>>8)&15)+cz-(S-2);
            // Short FFT axes can wrap inside a four-cell output tile.
            a=a<1 ? a+nx : (a>nx ? a-nx : a);
            b=b<1 ? b+ny : (b>ny ? b-ny : b);
            c=c<1 ? c+nz : (c>nz ? c-nz : c);
            if(tid<S*S*S&&(Masked||(a>=1&&a<=6&&b>=1&&b<=6&&c>=1&&c<=6))) {
                float factor=(tile_values[offset+a]*tile_values[offset+6+b])*tile_values[offset+12+c];
                #pragma unroll
                for(int pair=0;pair<B/2;++pair) {
                    float4 values=*reinterpret_cast<const float4*>(tile_values+offset+20+4*pair);
                    sum[2*pair].x=fmaf(values.x,factor,sum[2*pair].x);
                    sum[2*pair].y=fmaf(values.y,factor,sum[2*pair].y);
                    sum[2*pair+1].x=fmaf(values.z,factor,sum[2*pair+1].x);
                    sum[2*pair+1].y=fmaf(values.w,factor,sum[2*pair+1].y);
                }
            }
        }
        __syncthreads();
    }
    if(valid) {
        #pragma unroll
        for(int coil=0;coil<B;++coil) {
            if(single) grid[output+nf*coil]=sum[coil];
            else {
                atomicAdd(&grid[output+nf*coil].x,sum[coil].x);
                atomicAdd(&grid[output+nf*coil].y,sum[coil].y);
            }
        }
    }
}

// Cache the same twelve tile-local 1D window values. Zero entries encode
// stencil exclusion; valid contributions use the original Float32 product.
template<int B>
__device__ __forceinline__ void spread_coeffs_body(float2* __restrict__ grid,
        const float2* __restrict__ samples,const int32_t* __restrict__ ids,
        const int32_t* __restrict__ chunks,const float* __restrict__ coefficients,
        const int32_t* __restrict__ permutation,int nc,int nx,int ny,int nz,int nt,int nf) {
    constexpr int P=32,fields=12+2*B;
    extern __shared__ __align__(16) float tile_values[];
    int chunk=blockIdx.x,tid=threadIdx.x;
    int first=chunks[chunk]-1,last=chunks[chunk+nc]-1,tile=chunks[chunk+2*nc]-1;
    bool single=chunks[chunk+3*nc]!=0;
    int tx=(nx+3)/4,ty=(ny+3)/4,cx=tid%4,cy=(tid/4)%4,cz=tid/16;
    int x=(tile%tx)*4+cx,y=((tile/tx)%ty)*4+cy,z=(tile/(tx*ty))*4+cz;
    int output=x+nx*(y+ny*z);bool valid=x<nx&&y<ny&&z<nz;
    float2 sum[B];
    #pragma unroll
    for(int coil=0;coil<B;++coil)sum[coil]=make_float2(0.f,0.f);
    for(int base=first;base<=last;base+=P) {
        int count=min(P,last-base+1);
        for(int local=tid;local<count;local+=64) {
            int entry=base+local,p=ids[entry]-1,offset=local*fields;
            #pragma unroll
            for(int axis=0;axis<3;++axis) {
                float4 values=*reinterpret_cast<const float4*>(coefficients+12*entry+4*axis);
                *reinterpret_cast<float4*>(tile_values+offset+4*axis)=values;
            }
            int j=permutation[p]-1;
            #pragma unroll
            for(int coil=0;coil<B;++coil) {
                float2 value=samples[j+nt*coil];
                tile_values[offset+12+2*coil]=value.x;tile_values[offset+13+2*coil]=value.y;
            }
        }
        __syncthreads();
        #pragma unroll 1
        for(int local=0;local<count;++local) {
            int offset=local*fields;
            float factor=(tile_values[offset+cx]*tile_values[offset+4+cy])*tile_values[offset+8+cz];
            if(factor!=0.f) {
                #pragma unroll
                for(int pair=0;pair<B/2;++pair) {
                    float4 v=*reinterpret_cast<const float4*>(tile_values+offset+12+4*pair);
                    sum[2*pair].x=fmaf(v.x,factor,sum[2*pair].x);
                    sum[2*pair].y=fmaf(v.y,factor,sum[2*pair].y);
                    sum[2*pair+1].x=fmaf(v.z,factor,sum[2*pair+1].x);
                    sum[2*pair+1].y=fmaf(v.w,factor,sum[2*pair+1].y);
                }
            }
        }
        __syncthreads();
    }
    if(valid) {
        #pragma unroll
        for(int coil=0;coil<B;++coil) {
            if(single)grid[output+nf*coil]=sum[coil];
            else {
                atomicAdd(&grid[output+nf*coil].x,sum[coil].x);
                atomicAdd(&grid[output+nf*coil].y,sum[coil].y);
            }
        }
    }
}
// Only the coil batches exposed by the package are built.
#define GATHER(B) extern "C" __global__ void native_gather_u9_##B(float2* s,const float2* g,const int32_t* a,const float* w,const int32_t* p,int x,int y,int z,int t,int f,float n) {gather_body<B,false,9>(s,g,a,w,p,x,y,z,t,f,n);}
#define SPREAD(B) extern "C" __global__ void native_spread_##B(float2* g,const float2* s,const int32_t* ids,const int32_t* chunks,const uint16_t* o,const float* w,const int32_t* p,int nc,int x,int y,int z,int t,int f) {spread_body<B,32>(g,s,ids,chunks,o,w,p,nc,x,y,z,t,f);}
#define SPREAD_COEFFS(B) extern "C" __global__ void native_spread_coeffs_##B(float2* g,const float2* s,const int32_t* ids,const int32_t* chunks,const float* w,const float* unused,const int32_t* p,int nc,int x,int y,int z,int t,int f) {spread_coeffs_body<B>(g,s,ids,chunks,w,p,nc,x,y,z,t,f);}
GATHER(2)
GATHER(4)
GATHER(8)
SPREAD(2)
SPREAD(4)
SPREAD(8)
SPREAD_COEFFS(2)
SPREAD_COEFFS(4)
SPREAD_COEFFS(8)
