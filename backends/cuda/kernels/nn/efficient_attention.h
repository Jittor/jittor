// C boundary for the optional, directly compiled CUTLASS attention kernels.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

struct JtEfficientAttention {
    int batches, heads, queries, keys, head_dim, value_dim, causal, dtype;
    int64_t bias_batch_stride, bias_head_stride, bias_row_stride;
    float scale, dropout;
};

extern "C" {
const char *jt_efficient_attention_error();
int64_t jt_efficient_attention_workspace(const JtEfficientAttention *);
int jt_efficient_attention_forward(const JtEfficientAttention *, const void *, const void *,
                                   const void *, const void *, void *, float *, float *,
                                   const int64_t *, int64_t *, cudaStream_t);
int jt_efficient_attention_backward(const JtEfficientAttention *, const void *, const void *,
                                    const void *, const void *, const void *, const float *,
                                    const void *, void *, void *, void *, float *, float *,
                                    const int64_t *, cudaStream_t);
}
