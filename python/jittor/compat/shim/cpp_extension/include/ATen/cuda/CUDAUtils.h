// ATen/cuda/CUDAUtils.h — at::cuda::check_device. Host-only header (included by
// exts only under #ifndef __CUDACC__).
#pragma once
#include <torch/extension.h>

namespace at { namespace cuda {

// True if every defined tensor resides on CUDA on the same device index.
// (Under CUDA_VISIBLE_DEVICES there is a single visible device == index 0.)
// Single ArrayRef<Tensor> overload only: it constructs from both {a,b}
// initializer-lists and std::vector<Tensor>, so a second overload would make
// brace-init calls ambiguous.
inline bool check_device(at::ArrayRef<at::Tensor> ts) {
    int64_t dev = -1;
    for (const at::Tensor& t : ts) {
        if (!t.defined()) continue;
        if (!t.is_cuda()) return false;
        int64_t d = t.get_device();
        if (dev < 0) dev = d;
        else if (d != dev) return false;
    }
    return true;
}

}} // namespace at::cuda
