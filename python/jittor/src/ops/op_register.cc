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
static OpId next_op_id = 1;

//: The one key the map is read and written by.
//:
//: `op_registe` used to insert under `op_info.name` while `has_op` and
//: `get_op_info` looked up `op_name_to_file_name(name)`, which truncates at the
//: first '.'. For every name without a dot the two agree and nothing showed;
//: register a name *with* a dot and it went in under the full name and could
//: never be found again, by any spelling. One key, computed in one place.
static inline string op_key(const string& name) {
    return Op::op_name_to_file_name(name);
}

void op_registe(const OpInfo& op_info) {
    if (has_op(op_info.name)) {
        string op_file_name = op_key(op_info.name);
        auto iter = op_info_map.find(op_file_name);
        if (iter != op_info_map.end() && iter->second.source_path == op_info.source_path) {
            LOGvv << "replace duplicated op registration" << op_info.name
                << "\nsource_path:" << op_info.source_path
                << "\nextra_flags:" << op_info.extra_flags
                << "\nold_extra_flags:" << iter->second.extra_flags;
            OpInfo replacement = op_info;
            replacement.id = iter->second.id;
            iter->second = move(replacement);
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
    OpInfo registered = op_info;
    registered.id = next_op_id++;
    op_info_map[op_key(op_info.name)] = move(registered);
}

bool has_op(const string& name) {
    return op_info_map.count(op_key(name));
}

OpInfo get_op_info(const string& name) {
    auto iter = op_info_map.find(op_key(name));
    ASSERT(iter != op_info_map.end()) << "Op" << name << "not found.";
    return iter->second;
}

OpId get_op_id(const string& name) {
    return get_op_info(name).id;
}

#define DEFINE_BUILTIN_OP_ID(name) \
    OpId op_ids::name() { \
        static OpId id = get_op_id(#name); \
        return id; \
    }

DEFINE_BUILTIN_OP_ID(array)
DEFINE_BUILTIN_OP_ID(binary)
DEFINE_BUILTIN_OP_ID(broadcast_to)
DEFINE_BUILTIN_OP_ID(empty)
DEFINE_BUILTIN_OP_ID(fused)
DEFINE_BUILTIN_OP_ID(getitem)
DEFINE_BUILTIN_OP_ID(index)
DEFINE_BUILTIN_OP_ID(reduce)
DEFINE_BUILTIN_OP_ID(reindex)
DEFINE_BUILTIN_OP_ID(reindex_reduce)
DEFINE_BUILTIN_OP_ID(safe_clip)
DEFINE_BUILTIN_OP_ID(setitem)

#undef DEFINE_BUILTIN_OP_ID

vector<OpByType*> op_types;

int registe_op_type(OpByType* op_type) {
    op_types.push_back(op_type);
    return 0;
}


} // jittor
