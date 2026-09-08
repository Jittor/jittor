#pragma once
#include "core/common.h"
#include <array>

namespace jittor {
struct Var;
EXTERN_LIB void check_onednn_conv_args(Var*, Var*, int strideh, int stridew,
    int paddingh, int paddingw, int dilationh, int dilationw, int groups,
    const string& xformat, const string& wformat, const string& yformat);
struct OneDnnConvSpec {
    // 0: forward, 1: backward input, 2: backward weight.
    int kind = 0;
    std::array<int64, 4> source{}, weights{}, destination{};
    std::array<int64, 2> stride{}, padding{}, dilation{};
    int64 groups = 1;
    string xformat, wformat, yformat;
};
EXTERN_LIB OneDnnConvSpec onednn_conv_spec(int kind, Var* source, Var* weights, Var* destination,
    int strideh, int stridew, int paddingh, int paddingw, int dilationh, int dilationw,
    int groups, const string& xformat, const string& wformat, const string& yformat);
EXTERN_LIB void onednn_conv_execute(const OneDnnConvSpec&, void* source, void* weights, void* destination);
EXTERN_LIB void onednn_matmul_execute(int64 batch, int64 n, int64 m, int64 k,
                                    bool trans_a, bool trans_b, void* a, void* b, void* c);
// @pyjt(onednn_cache_info)
vector<int64> onednn_cache_info();
// Explicit administrative clear, also used to compare rebuild versus reuse.
// @pyjt(onednn_cache_clear)
void onednn_cache_clear();
// @pyjt(onednn_version)
vector<int> onednn_version();
}
