// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <vector>
#include <iostream>
#include <map>

namespace jittor {
struct CacheConfig {
    size_t size, ways, line_size;
    CacheConfig(size_t size, size_t ways, size_t line_size=64);
};

struct Cache {
    CacheConfig config;
    int miss_time;

    Cache(const CacheConfig config);
    virtual ~Cache();
    void clear();
    bool check_hit(size_t paddr);
    virtual bool check_hit_(size_t paddr) = 0;
    virtual void clear_() = 0;
};

struct DefaultReplacementCache : Cache {
    std::map<size_t, std::vector<size_t>> data;

    DefaultReplacementCache(const CacheConfig config);
    bool check_hit_(size_t paddr);
    void clear_();
};

struct LRUCache : Cache {
    std::map<size_t, std::vector<size_t>> data;

    LRUCache(const CacheConfig config);
    bool check_hit_(size_t paddr);
    void clear_();
};

} // jittor