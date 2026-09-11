// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
// The cache key for "which cuDNN algorithm did we pick for this convolution".
//
// The six legacy algorithm caches -- forward, backward-data and
// backward-filter, each shared between the 2-D and the 3-D op -- used to build
// their key by writing decimal and hexadecimal text into the *shared jit key
// buffer* and taking a `std::string` off it:
//
//     JK& jk = get_jk();
//     jk.clear();
//     jk << dimX[0] << "," << dimX[1] << ...;
//     auto iter = bwdx_algo_cache.find(jk.to_string());
//
// That is a buffer the executor owns and had already filled with the key of the
// kernel currently running, plus a `std::string` construction, on every
// convolution. This struct replaces the text: it is hashed and compared as
// bytes.
//
// Deliberately dependent on nothing -- no cuDNN, no jittor headers, only the
// three standard ones below. `tests/backends/cuda/test_cudnn_conv_algo_key.py`
// compiles it on its own to check that different convolutions get different
// keys, which is the only way that is checked rather than asserted in a
// comment here. Keep it that way: one `#include "cudnn_wrapper.h"` and that
// case stops building, silently.
#include <cstdint>
#include <cstring>
#include <unordered_map>

namespace jittor {

// Which of the three passes an algorithm was chosen for. There is a table per
// pass, and the 2-D and the 3-D op share each one, so this does not separate
// them by itself -- `spatial_dims` does.
enum ConvAlgoPass {
    CONV_ALGO_FWD = 0,
    CONV_ALGO_BWD_DATA = 1,
    CONV_ALGO_BWD_FILTER = 2,
};

/* Everything that changes which cuDNN algorithm is right for a convolution.

   All `int64_t` and nothing else, so there is no padding and the struct's bytes
   *are* the key: `ConvAlgoKeyHash` hashes the object whole and `ConvAlgoKeyEq`
   compares it with `memcmp`. Two consequences, both of which the text keys this
   replaced got wrong:

    * a field left out does not fail loudly. It makes two different
      convolutions share an algorithm, and the wrong one still runs and still
      produces numbers. The 2-D backward keys carried neither the operand
      dtypes, nor the tensor strides, nor the output extent, nor the workspace
      budget -- so an fp16 convolution took the algorithm measured for the fp32
      one of the same shape, and an NHWC one took the NCHW one's (`dimX` is read
      through the layout string and so is identical for both; only the strides
      differ).

    * padding would be hashed along with the fields, so an uninitialised gap
      would make the same convolution miss its own entry. `conv_algo_key`
      memsets before filling in, and the static_assert below is what keeps
      "no padding" true rather than assumed.

   Extents are a fixed 5 wide and parameters a fixed 3 so that one struct serves
   rank 4 and rank 5; the unused tail stays zero. */
struct ConvAlgoKey {
    int64_t pass;
    int64_t spatial_dims;
    int64_t dtype_x, dtype_w, dtype_y;
    int64_t compute_type, math_type;
    int64_t filter_format;
    int64_t groups;
    // The `float`'s bit pattern, not a rounded value: an algorithm measured
    // under one workspace budget must not be reused under another.
    int64_t workspace_ratio_bits;
    int64_t dim_x[5], stride_x[5];
    int64_t dim_w[5];
    int64_t dim_y[5], stride_y[5];
    int64_t pad[3], conv_stride[3], dilation[3];
};

// Ten scalars, five 5-wide extents, three 3-wide parameters.
static_assert(sizeof(ConvAlgoKey) == sizeof(int64_t) * (10 + 5 * 5 + 3 * 3),
              "ConvAlgoKey has padding; its bytes are the cache key");

struct ConvAlgoKeyHash {
    size_t operator()(const ConvAlgoKey& k) const {
        const unsigned char* p = (const unsigned char*)&k;
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < sizeof(k); i++) { h ^= p[i]; h *= 1099511628211ull; }
        return (size_t)h;
    }
};

struct ConvAlgoKeyEq {
    bool operator()(const ConvAlgoKey& a, const ConvAlgoKey& b) const {
        return memcmp(&a, &b, sizeof(a)) == 0;
    }
};

/* One of the three algorithm caches. `Algo` is the cuDNN enum for the pass;
   naming those here would mean including cuDNN, which is what keeps this
   header compilable on its own. */
template<class Algo>
using ConvAlgoCache = std::unordered_map<ConvAlgoKey, Algo, ConvAlgoKeyHash, ConvAlgoKeyEq>;

/* Fill in a key.

   `spatial_dims` is 2 or 3 and the tensor rank is `spatial_dims + 2`.
   `dim_x` and `dim_y` are in normalised (N, C, spatial...) order and the memory
   layout is carried by `stride_x` and `stride_y`, exactly as the tensor
   descriptors take them; the filter's layout is `filter_format`. */
inline ConvAlgoKey conv_algo_key(
        int64_t pass, int64_t spatial_dims,
        int64_t dtype_x, int64_t dtype_w, int64_t dtype_y,
        int64_t compute_type, int64_t math_type,
        int64_t filter_format, int64_t groups, float workspace_ratio,
        const int* dim_x, const int* stride_x,
        const int* dim_w,
        const int* dim_y, const int* stride_y,
        const int* pad, const int* conv_stride, const int* dilation) {
    ConvAlgoKey k;
    memset(&k, 0, sizeof(k));
    k.pass = pass;
    k.spatial_dims = spatial_dims;
    k.dtype_x = dtype_x; k.dtype_w = dtype_w; k.dtype_y = dtype_y;
    k.compute_type = compute_type; k.math_type = math_type;
    k.filter_format = filter_format;
    k.groups = groups;
    uint32_t bits;
    memcpy(&bits, &workspace_ratio, sizeof(bits));
    k.workspace_ratio_bits = bits;
    int64_t rank = spatial_dims + 2;
    for (int64_t i = 0; i < rank && i < 5; i++) {
        k.dim_x[i] = dim_x[i];  k.stride_x[i] = stride_x[i];
        k.dim_w[i] = dim_w[i];
        k.dim_y[i] = dim_y[i];  k.stride_y[i] = stride_y[i];
    }
    for (int64_t i = 0; i < spatial_dims && i < 3; i++) {
        k.pad[i] = pad[i];
        k.conv_stride[i] = conv_stride[i];
        k.dilation[i] = dilation[i];
    }
    return k;
}

} // jittor
