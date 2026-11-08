// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "type/nano_vector.h"

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

    // An entry ending exactly at bit 64 must not corrupt the next push.
    nv.clear();
    int64 strides[] = {0, 1024, 512, 256, 128, 64, 16, 8, 4, 2, 1};
    for (auto v : strides) nv.push_back(v);
    ASSERT(nv.is_spilled() && nv.size()==11) << nv;
    for (int i=0; i<11; i++) ASSERTop(nv[i],==,strides[i]);

    // More than ten entries spills; copies share storage and copy on write.
    NanoVector big;
    for (int i=0; i<40; i++) big.push_back(i*3-7);
    ASSERT(big.is_spilled() && big.size()==40);
    NanoVector copy = big;
    copy.set_data(5, 1000);
    ASSERTop(big[5],==,8);
    ASSERTop(copy[5],==,1000);
    ASSERT(big != copy);
    copy.set_data(5, 8);
    ASSERT(big == copy && std::hash<NanoVector>{}(big) == std::hash<NanoVector>{}(copy));
    ASSERT(!(big < copy) && !(copy < big));
    // Shrinking back within the packed bounds repacks, so the raw form of a
    // vector built by appending stays unique.
    for (int i=0; i<30; i++) big.pop_back();
    NanoVector small;
    for (int i=0; i<10; i++) small.push_back(i*3-7);
    ASSERT(!big.is_spilled() && big == small && big.data == small.data
        && big.offset == small.offset) << big << small;
    // Values too wide for 64 packed bits also spill.
    NanoVector wide{(int64)1<<40, (int64)1<<40};
    ASSERT(wide.is_spilled() && wide[0]==((int64)1<<40) && wide[1]==((int64)1<<40));
    NanoVector moved = std::move(wide);
    ASSERT(moved.size()==2 && wide.size()==0);
    NanoVector rs;
    rs.reserve(20, 20);
    rs.set_data(19, 19);
    ASSERT(rs.size()==20 && rs[19]==19 && rs[0]==0);
}

} // jittor
