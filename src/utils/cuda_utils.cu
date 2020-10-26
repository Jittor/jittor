#include <cuda_runtime.h>

__global__ static void OutputIsAR(int n) {
    return;
}

void mark_output_is_ar(cudaStream_t stream) {
    OutputIsAR<<<1,1,0,stream>>>(0);
}
