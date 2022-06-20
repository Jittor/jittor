// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

static inline int lzcnt(int64 v) {
    #ifdef __clang__
    #if __has_feature(__builtin_ia32_lzcnt_u64)
        return __builtin_ia32_lzcnt_u64(v);
    #else
        return v ? __builtin_clzll(v) : 64;
    #endif
    #else
    #ifdef _MSC_VER
        unsigned long index;
        _BitScanReverse64(&index, v);
        return v ? 63-index : 64;
    #else
        return __builtin_clzll(v);
    #endif
    #endif
}

}