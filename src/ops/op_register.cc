// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "op.h"
#include "ops/op_register.h"

namespace jittor {

unordered_map<string, OpInfo> op_info_map;

void op_registe(const OpInfo& op_info) {
    ASSERT(!has_op(op_info.name)) << "Op" << op_info.name << "is already registed, "
        << "source_path:" << op_info.source_path << "extra_flags" << op_info.extra_flags;
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

} // jittor