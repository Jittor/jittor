// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "misc/fast_shared_ptr.h"

namespace jittor {

JIT_TEST(fast_shared_ptr) {
    unordered_map<string,int> a;
    fast_shared_ptr<unordered_map<string,int>> ap(move(a));
    ASSERT(ap.ptr==0);
    ap = {{"a",1}};
    auto bp = ap;
    ASSERT(bp.ptr==ap.ptr && bp.ref_cnt()==2);
    ap = nullptr;
    ASSERT(ap.ptr==nullptr && bp.ref_cnt()==1);
    ap = clone(bp.data());
    ASSERT(ap.data().size()==1 && bp.ref_cnt()==1);
}

} // jittor