// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "op.h"
#include "ops/op_register.h"

namespace jittor {

unordered_map<string, OpInfo> op_info_map;

void op_registe(const OpInfo& op_info) {
    if (has_op(op_info.name)) {
        string op_file_name = Op::op_name_to_file_name(op_info.name);
        auto iter = op_info_map.find(op_file_name);
        if (iter != op_info_map.end() && iter->second.source_path == op_info.source_path) {
            LOGvv << "replace duplicated op registration" << op_info.name
                << "\nsource_path:" << op_info.source_path
                << "\nextra_flags:" << op_info.extra_flags
                << "\nold_extra_flags:" << iter->second.extra_flags;
            iter->second = op_info;
            return;
        }
        ASSERT(false) << "Op" << op_info.name << "is already registed, "
            << "source_path:" << op_info.source_path << "extra_flags" << op_info.extra_flags;
    }
    LOGvv << "registe op" << op_info.name
        << "\nsource_path:" << op_info.source_path
        << "\nextra_flags:" << op_info.extra_flags
        << "\nconstructors:" << op_info.constructors
        << "\nvar_members:" << op_info.var_members;
    op_info_map[op_info.name] = op_info;
}

bool has_op(const string& name) {
    string op_file_name = Op::op_name_to_file_name(name);
    return op_info_map.count(op_file_name);
}

OpInfo get_op_info(const string& name) {
    string op_file_name = Op::op_name_to_file_name(name);
    ASSERT(has_op(op_file_name)) << "Op" << name << "not found.";
    return op_info_map.at(op_file_name);
}

vector<OpByType*> op_types;

int registe_op_type(OpByType* op_type) {
    op_types.push_back(op_type);
    return 0;
}


} // jittor
