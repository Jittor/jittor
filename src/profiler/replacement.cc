// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "profiler/replacement.h"

namespace jittor {
CacheConfig::CacheConfig(size_t size, size_t ways, size_t line_size) : size(size), ways(ways), line_size(line_size) {} 


Cache::Cache(const CacheConfig config) : config(config), miss_time(0) {
}

Cache::~Cache() {
}

void Cache::clear() {
    miss_time = 0;
    clear_();
}

bool Cache::check_hit(size_t paddr) {
    bool hit = check_hit_(paddr);
    if (!hit) ++miss_time;
    return hit;
}

DefaultReplacementCache::DefaultReplacementCache(const CacheConfig config) : Cache(config) {}

bool DefaultReplacementCache::check_hit_(size_t paddr) {
    size_t cache_set = paddr % (config.size/config.ways) / config.line_size;
    size_t tag = paddr / (config.size/config.ways);
    for (auto t : data[cache_set])
        if (t == tag) return true;
    if (data[cache_set].size() >= config.ways) return false;
    data[cache_set].push_back(tag);
    return false;
}

void DefaultReplacementCache::clear_() {
    data.clear();
}

LRUCache::LRUCache(const CacheConfig config) : Cache(config) {}

bool LRUCache::check_hit_(size_t paddr) {
    size_t cache_set = paddr % (config.size/config.ways) / config.line_size;
    size_t tag = paddr / (config.size/config.ways);

    for (int i = 0; i < (int)data[cache_set].size(); ++i) {
        size_t t = data[cache_set][i];
        if (t == tag) {
            data[cache_set].erase(data[cache_set].begin() + i);
            data[cache_set].insert(data[cache_set].begin(), tag);
            return true;
        }
    }
    data[cache_set].insert(data[cache_set].begin(), tag);
    if (data[cache_set].size() > config.ways) {
        data[cache_set].pop_back();
    }
    return false;
}

void LRUCache::clear_() {
    data.clear();
}

} //jittor