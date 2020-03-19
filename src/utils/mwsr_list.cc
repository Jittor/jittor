// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "utils/mwsr_list.h"

#ifdef TEST

#include <cassert>
#include <thread>
#include <iostream>

using namespace std;

MWSR_LIST(test, int64_t);

int n, m, tnum;

void reduce() {
    int64_t sum=0;
    mwsr_list_test::reduce([&](const int64_t& s) {
        sum += s;
    }, [](){});
    
    int64_t expect = int64_t(m)*(m-1)/2*n*tnum;
    cout << "get sum " << sum << ' ' << sum - expect << endl;
    assert(expect == sum);
}

void add() {
    for (int i=0; i<n; i++) {
        int64_t ts = 0;
        for (int j=0; j<m; j++) ts += j;
        mwsr_list_test::push(move(ts));
    }
}

void test(int _n, int _m, int _tnum) {
    mwsr_list_test::clear();
    n = _n;
    m = _m;
    tnum = _tnum;
    list<thread> ts;
    thread checker(reduce);
    for (int i=0; i<tnum; i++)
        ts.emplace_back(add);
    for (auto& t : ts) t.join();
    mwsr_list_test::stop();
    checker.join();
}

int main() {
    test(1000, 1000, 3);
    test(1000, 100000, 3);
    test(100, 10000, 8);
    test(100000, 10, 16);
    test(1000, 100000, 16);
    return 0;
}

#endif