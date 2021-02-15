// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <experimental/string_view>
#include "common.h"

namespace jittor {

using std::experimental::string_view;

template<class T>
struct string_view_map {
    typedef typename std::unordered_map<string_view, T> umap_t;
    typedef typename umap_t::iterator iter_t;
    umap_t umap;
    vector<string> holder;

    iter_t find(string_view sv) {
        return umap.find(sv);
    }

    iter_t begin() { return umap.begin(); }
    iter_t end() { return umap.end(); }

    const T& at(string_view sv) { return umap.at(sv); }
    size_t size() { return umap.size(); }

    T& operator[](string_view sv) {
        auto iter = find(sv);
        if (iter != end()) return iter->second;
        holder.emplace_back(sv);
        string_view nsv = holder.back();
        return umap[nsv];
    }
};


} // jittor
