#pragma once
#include "ops/composite/index_bounds.h"
namespace jittor {
struct IndexedStorageLayout {
    int64 shape[16], stride[16];
    const void* indices[16];
    int axes[16], free_axes[16], is64[16];
    int rank, nindex, nfree;
    int64 count, row_size;
};
template<class T>
__global__ void indexed_storage_write_cuda(T* dst, const T* src,
                                      IndexedStorageLayout layout, IndexFault* fault) {
    for (int64 linear=int64(blockIdx.x)*blockDim.x+threadIdx.x;
            linear<layout.count;linear+=int64(blockDim.x)*gridDim.x) {
        int64 row=linear/layout.row_size, rest=linear%layout.row_size, offset=0;
        for (int d=layout.nfree-1;d>=0;--d) {
            int axis=layout.free_axes[d];
            offset+=(rest%layout.shape[axis])*layout.stride[axis];
            rest/=layout.shape[axis];
        }
        for (int k=0;k<layout.nindex;++k) {
            int axis=layout.axes[k];
            int64 index=layout.is64[k]
                ? static_cast<const int64*>(layout.indices[k])[row]
                : static_cast<const int32*>(layout.indices[k])[row];
            offset+=::jittor::bounded_index(index,layout.shape[axis],axis,fault)*layout.stride[axis];
        }
        dst[offset]=src[linear];
    }
}
}
