#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace jittor_ordered_sum {
// One physical arithmetic owner for explicit sum and native broadcast VJP.
// Four independently rounded accumulators, then a left-associated fold.
__global__ static void nonlast_fp32(const float* input, float* output,
                                   int outputs, int rows, int columns) {
    for (int index = blockIdx.x*blockDim.x+threadIdx.x;
         index < outputs; index += blockDim.x*gridDim.x) {
        const int batch = index/columns, column = index%columns;
        const float* first = input+batch*rows*columns+column;
        float a0=0.0f, a1=0.0f, a2=0.0f, a3=0.0f;
        int row=0;
        for (; row+3<rows; row+=4) {
            a0=a0+first[row*columns];
            a1=a1+first[(row+1)*columns];
            a2=a2+first[(row+2)*columns];
            a3=a3+first[(row+3)*columns];
        }
        if (row<rows) a0=a0+first[row*columns];
        if (row+1<rows) a1=a1+first[(row+1)*columns];
        if (row+2<rows) a2=a2+first[(row+2)*columns];
        output[index]=((a0+a1)+a2)+a3;
    }
}
// ATen2.8 Reduce.cuh: output_vec_size=4, block=(32,4), vt0=4,
// input_mult[Y]=1, step_input=4, no CTA/global reduction. Restrict dispatch
// to dense aligned full-leading-prefix R76/C1024; not a general large-C tree.
__global__ static void leading76c1024_fp32(const float* input, float* output) {
    const int column4=blockIdx.x*32+threadIdx.x;
    float a[4][4] = {};
    int row=threadIdx.y;
    for (; row+12<76; row+=16) {
        #pragma unroll
        for (int slot=0; slot<4; ++slot) {
            const float4 v=reinterpret_cast<const float4*>(input)[(row+slot*4)*256+column4];
            a[slot][0]=__fadd_rn(a[slot][0],v.x);
            a[slot][1]=__fadd_rn(a[slot][1],v.y);
            a[slot][2]=__fadd_rn(a[slot][2],v.z);
            a[slot][3]=__fadd_rn(a[slot][3],v.w);
        }
    }
    #pragma unroll
    for (int slot=0; slot<4; ++slot) {
        if (row+slot*4<76) {
            const float4 v=reinterpret_cast<const float4*>(input)[(row+slot*4)*256+column4];
            a[slot][0]=__fadd_rn(a[slot][0],v.x);
            a[slot][1]=__fadd_rn(a[slot][1],v.y);
            a[slot][2]=__fadd_rn(a[slot][2],v.z);
            a[slot][3]=__fadd_rn(a[slot][3],v.w);
        }
    }
    float4 value;
    value.x=__fadd_rn(__fadd_rn(__fadd_rn(a[0][0],a[1][0]),a[2][0]),a[3][0]);
    value.y=__fadd_rn(__fadd_rn(__fadd_rn(a[0][1],a[1][1]),a[2][1]),a[3][1]);
    value.z=__fadd_rn(__fadd_rn(__fadd_rn(a[0][2],a[1][2]),a[2][2]),a[3][2]);
    value.w=__fadd_rn(__fadd_rn(__fadd_rn(a[0][3],a[1][3]),a[2][3]),a[3][3]);
    __shared__ float4 exchange[128];
    const int address=threadIdx.y*32+threadIdx.x;
    exchange[address]=value;
    // First (y0+y2), (y1+y3), then ((y0+y2)+(y1+y3)).
    for (int offset=2; offset>0; offset>>=1) {
        __syncthreads();
        if (threadIdx.y<offset) {
            const float4 other=exchange[address+offset*32];
            value.x=__fadd_rn(value.x,other.x);
            value.y=__fadd_rn(value.y,other.y);
            value.z=__fadd_rn(value.z,other.z);
            value.w=__fadd_rn(value.w,other.w);
            exchange[address]=value;
        }
    }
    if (threadIdx.y==0) reinterpret_cast<float4*>(output)[column4]=value;
}

// One last-axis arithmetic owner for public sum, mean and broadcast VJP.
// ATen2.8 uses scalar input leaves through width128, float4 above it.
// Preserve alignment heads/tails and separately rounded accumulation lanes.
template<bool Mean>
__global__ static void last_fp32(const float* input, float* output,
                                int rows, int columns) {
    const int row = blockIdx.x*blockDim.y+threadIdx.y;
    const int lane = threadIdx.x;
    const bool valid = row < rows;
    float a0=0.0f, a1=0.0f, a2=0.0f, a3=0.0f;
    if (valid) {
        const float* first=input+static_cast<long long>(row)*columns;
        if (columns<=128) {
            if (lane<columns) a0=a0+first[lane];
            if (lane+blockDim.x<columns) a1=a1+first[lane+blockDim.x];
            if (lane+2*blockDim.x<columns) a2=a2+first[lane+2*blockDim.x];
            if (lane+3*blockDim.x<columns) a3=a3+first[lane+3*blockDim.x];
        } else {
            const int offset=(reinterpret_cast<uintptr_t>(first)&15u)/sizeof(float);
            const int head=offset ? 4-offset : 0;
            if (offset && lane>=offset && lane<4)
                a0=a0+first[lane-offset];
            const float* aligned=first+head;
            const int remaining=columns-head;
            const int groups=remaining/4;
            for (int group=lane; group<groups; group+=blockDim.x) {
                const float4 values=reinterpret_cast<const float4*>(aligned)[group];
                a0=a0+values.x;
                a1=a1+values.y;
                a2=a2+values.z;
                a3=a3+values.w;
            }
            const int tail_start=groups*4;
            if (tail_start+lane<remaining)
                a0=a0+aligned[tail_start+lane];
        }
    }
    float value=((a0+a1)+a2)+a3;
    __shared__ float exchange[512];
    const int address=threadIdx.y*blockDim.x+lane;
    if (blockDim.x>32) {
        exchange[address]=value;
        for (int offset=blockDim.x/2; offset>=32; offset>>=1) {
            __syncthreads();
            if (lane<offset) {
                value=value+exchange[address+offset];
                exchange[address]=value;
            }
        }
    }
    // Invalid final rows participate in every block barrier as well.
    __syncthreads();
    for (int offset=1; offset<32; offset<<=1)
        value=value+__shfl_down_sync(0xffffffffu,value,offset);
    if (valid && lane==0) {
        const float factor=static_cast<float>(1.0/static_cast<double>(columns));
        output[row]=Mean ? value*factor : value;
    }
}

inline void launch_last_fp32(const float* input, float* output,
                            int rows, int columns, bool mean=false) {
    int column_power=1, row_power=1;
    const int groups=columns<=128 ? columns : columns/4;
    while (column_power<=groups/2) column_power*=2;
    while (row_power<=rows/2 && row_power<512) row_power*=2;
    if (column_power>512) column_power=512;
    int width=column_power<32 ? column_power : 32;
    int height=row_power<512/width ? row_power : 512/width;
    width=column_power<512/height ? column_power : 512/height;
    if (mean)
        last_fp32<true><<<(rows+height-1)/height,dim3(width,height)>>>(input,output,rows,columns);
    else
        last_fp32<false><<<(rows+height-1)/height,dim3(width,height)>>>(input,output,rows,columns);
}

} // namespace jittor_ordered_sum
