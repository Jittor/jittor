// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

// @pyjt(op_compiler)
// @attrs(submodule)
struct OpCompiler {
    // origin op ptr
    Op* _op;
    // if _op is a fused_op then op==_op, else op==nullptr
    FusedOp* op;
    // op source
    string src;
    // only available when op is fused op
    // op_members[i][j] represents i-th op's j-th member
    vector<vector<string>> op_members;

    OpCompiler(Op*);
    string get_src();
    void get_op_var_by_name(const string& name, uint& op_id, uint& opvar_id, Op*& op, Var*& var);
    string get_name_by_op_var(Op* op, Var* var);
    string get_name_by_op_input(Op* op, uint i);
    string get_name_by_op_output(Op* op, uint i);
    // op may be relay and not exist
    bool op_exist(Op* op);
    int total_member_count();

    string get_jit_src(Op* op);
    string get_fused_src(FusedOp* op);
    jit_op_entry_t compile(const string& jit_key, const string& src);
    static string __get_fused_src(
        const vector<Op*>& ops,
        const vector<string>& op_srcs,
        vector<vector<string>>& op_members
    );
    // @pyjt(eval)
    static int64 eval(const string& expr, const unordered_map<string,string>& vars);
    // @pyjt(precompile)
    static string precompile(const unordered_map<string,string>& defs, const string& src);
    static jit_op_entry_t do_compile(Op* op);
};

}