// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "misc/nano_vector.h"

namespace jittor {

JIT_TEST(nano_vector) {
    NanoVector nv;
    ASSERTop(nv.get_nbits(0),==,1);
    ASSERTop(nv.get_nbits(-1),==,1);

    ASSERTop(nv.get_nbits(1),==,2);
    ASSERTop(nv.get_nbits(-2),==,2);

    ASSERTop(nv.get_nbits(3),==,3);
    ASSERTop(nv.get_nbits(-4),==,3);
    
    nv.push_back(0);
    ASSERT(nv.size()==1 && nv[0]==0 && nv.total_bits()==1) << nv << nv.total_bits()
     << nv.size() << nv.offset;
    nv.push_back(0);
    ASSERT(nv.size()==2 && nv[1]==0 && nv.total_bits()==2);
    nv.push_back(-1);
    ASSERT(nv.size()==3 && nv[2]==-1 && nv.total_bits()==3) << nv;
    nv.push_back(2);
    ASSERT(nv.size()==4 && nv[3]==2 && nv.total_bits()==6) << nv << nv.total_bits();
    nv.push_back(3);
    ASSERT(nv.size()==5 && nv[4]==3 && nv.total_bits()==9) << nv << nv.total_bits();
    nv.push_back(-3);
    ASSERT(nv.size()==6 && nv[5]==-3 && nv.total_bits()==12)
        << nv << nv.total_bits() << nv[5] << nv.size();
    nv.push_back(1ull<<40);
    ASSERT(nv.size()==7 && nv[6]==(1ull<<40) && nv.total_bits()==54)
        << nv << nv.total_bits();
    nv.push_back(-(1<<5));
    ASSERT(nv.size()==8 && nv[7]==(-(1<<5)) && nv.total_bits()==60)
        << nv << nv.total_bits();
    nv.push_back(1);
    ASSERT(nv.size()==9 && nv[8]==1 && nv.total_bits()==62)
        << nv << nv.total_bits();
    nv.push_back(-2);
    ASSERT(nv.size()==10 && nv[9]==-2);

    nv.clear();
    nv.reserve(10, 10);
    nv.set_data(0, 10);
    nv.set_data(9, -10);
    nv.set_data(5, 4);
    ASSERT(nv.to_string()=="[10,0,0,0,0,4,0,0,0,-10,]") << nv;

    nv.clear();
    nv.reserve(10*8, 8);
    nv.set_data(0, 10);
    nv.set_data(7, -10);
    nv.set_data(5, 4);
    ASSERT(nv.to_string()=="[10,0,0,0,0,4,0,-10,]") << nv;
}

} // jittor
