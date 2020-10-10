// ***************************************************************
// Copyright (c) 2020 Jittor. All Rights Reserved.
// Authors: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "common.h"
#include "misc/nano_vector.h"

namespace jittor {

void cuda_loop_schedule(NanoVector o_shape, int* masks, int* tdims);

JIT_TEST(cuda_loop_schedule) {
    auto check = [&](const vector<int64>& shape, const vector<int>& masks, vector<int> tdims={}) {
        int masks2[shape.size()];
        int tdims2[6];
        cuda_loop_schedule(shape, masks2, tdims2);
        while (tdims.size() < 6) tdims.push_back(1);
        for (int i=0; i<shape.size(); i++)
            ASSERT(masks2[i] == masks[i]) << i << shape << masks << vector<int>(masks2, masks2+shape.size());
        for (int i=0; i<6; i++)
            ASSERT(tdims.at(i)==tdims2[i]) << tdims << vector<int>(tdims2, tdims2+6);
    };
    check({0}, {1}, {0,1,1,1,1,1});
    check({2,2,2,2}, {8, 4, 2, 1}, {2,2,2,2,1,1});
    check({2048,1024}, {8, 1}, {1024,1,1,2048,1,1});
    check({2048,1025}, {8, 1+(1<<6)}, {1024,1,1,2048,1,1});
    check({2048,3025}, {8, 1+(1<<6)}, {1024,1,1,2048,1,1});
    check({2048,4425}, {16, 1+8+(1<<6)}, {1024,1,1,5,2048,1});
    check({2048, 2048,4425}, {0, 16, 1+8+(1<<6)}, {1024,1,1,5,2048,1});
    check({3,3,3,4425}, {0, 32, 16, 1+8+(1<<6)}, {1024,1,1,5,3,3});
    check({3,3,3,4425, 3,3}, {0, 32, 16, 8+4+(1<<6), 2, 1}, {3,3,64,70,3,3});
    check({3,3,3,12, 9,9}, {32, 16, 8, 4, 2, 1}, {9,9,12,3,3,3});
    check({3,3,3,13, 9,9}, {32, 16, 8, 4+64, 2, 1}, {9,9,12,3,3,3});
    check({3,3,3,13*4, 9,9}, {0, 32, 16, 8+4+64, 2, 1}, {9,9,12,5,3,3});
    check({3,3,3,100, 3,3}, {32, 16, 8, 4+64, 2, 1}, {3,3,64,3,3,3});
    check({3,3,3,400, 3,3}, {0, 32, 16, 8+4+64, 2, 1}, {3,3,64,7,3,3});
}

} // jittor
