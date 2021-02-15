// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <typeinfo>
#include <typeindex>
#include "common.h"

namespace jittor {

struct OpInfo {
    string name, source_path, extra_flags;
    vector<pair<const std::type_info*, void*>> constructors;
    // string: var member name, uint64: var member offset
    vector<pair<string, uint64>> var_members;

    template<class To, class ...Ts> auto get_constructor() {
        typedef To (*func_t)(Ts...);
        const auto& tid = typeid(func_t);
        for (uint i=0; i<constructors.size(); i++)
            if (std::type_index(*(constructors[i].first)) == std::type_index(tid))
                return func_t(constructors[i].second);
        LOGf << "constructor" << name << tid.name() << "not found.";
        return func_t(nullptr);
    }
};

void op_registe(const OpInfo& op_info);
bool has_op(const string& name);
OpInfo get_op_info(const string& name);

} // jittor