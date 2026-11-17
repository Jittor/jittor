// Bit-preserving dense-to-strided copies on the compute stream.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include "core/common.h"
#include "helper_cuda.h"

namespace jittor {
namespace {
constexpr int kMaxRank = 32;
struct Layout {
    int64 shape[kMaxRank];
    int64 stride[kMaxRank];
    int64 count;
    int rank;
};

__device__ int64 storage_offset(int64 index, const Layout& layout) {
    int64 offset = 0;
    for (int d = layout.rank - 1; d >= 0; --d) {
        offset += (index % layout.shape[d]) * layout.stride[d];
        index /= layout.shape[d];
    }
    return offset;
}

template<int Bytes> struct alignas(Bytes) Word { unsigned char data[Bytes]; };

template<int Bytes>
__global__ void copy_words(Word<Bytes>* dst, const Word<Bytes>* src, Layout layout) {
    for (int64 i = int64(blockIdx.x) * blockDim.x + threadIdx.x;
            i < layout.count; i += int64(blockDim.x) * gridDim.x)
        dst[storage_offset(i, layout)] = src[i];
}

// External storage need not have the alignment of its reported element type.
__global__ void copy_bytes(unsigned char* dst, const unsigned char* src,
                           Layout layout, size_t item_size) {
    for (int64 i = int64(blockIdx.x) * blockDim.x + threadIdx.x;
            i < layout.count * item_size; i += int64(blockDim.x) * gridDim.x) {
        int64 offset = storage_offset(i / item_size, layout) * item_size + i % item_size;
        dst[offset] = src[i];
    }
}
} // namespace

bool cuda_copy_strided_from_dense(void* dst, const void* src,
                                  const int64* shape, const int64* stride,
                                  int rank, size_t item_size) {
    if (rank < 0 || rank > kMaxRank || !item_size || item_size > 16
            || (item_size & (item_size - 1))) return false;
    Layout layout{};
    layout.count = 1;
    for (int d = 0; d < rank; ++d) {
        layout.count *= shape[d];
        if (shape[d] == 1) continue;
        int n = layout.rank;
        if (n && layout.stride[n - 1] == shape[d] * stride[d]) {
            layout.shape[n - 1] *= shape[d];
            layout.stride[n - 1] = stride[d];
        } else {
            layout.shape[n] = shape[d];
            layout.stride[n] = stride[d];
            layout.rank++;
        }
    }
    if (!layout.count) return true;
    int blocks = int(std::min<int64>((layout.count + 255) / 256, 4096));
    if ((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) & (item_size - 1)) {
        copy_bytes<<<blocks, 256, 0, cudaStreamPerThread>>>(
            static_cast<unsigned char*>(dst), static_cast<const unsigned char*>(src), layout, item_size);
    } else {
#define COPY_WORDS(N) case N: copy_words<N><<<blocks, 256, 0, cudaStreamPerThread>>>( \
        static_cast<Word<N>*>(dst), static_cast<const Word<N>*>(src), layout); break
        switch (item_size) {
            COPY_WORDS(1);
            COPY_WORDS(2);
            COPY_WORDS(4);
            COPY_WORDS(8);
            COPY_WORDS(16);
        }
#undef COPY_WORDS
    }
    checkCudaErrors(cudaGetLastError());
    return true;
}
} // namespace jittor
