// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "node.h"
#include "jit_key.h"
#include "misc/string_view_map.h"

namespace jittor {

enum OpType {other=0, element=1, broadcast=2, reduce=3};
struct Op : Node {
    vector<VarPtr> outputs_holder;
    static int64_t number_of_lived_ops;
    
    inline Caster<Var*, Node::input_t> inputs() { CHECK_EXIST; return &_inputs; }
    inline Caster<Var*, Node::output_t> outputs() { CHECK_EXIST; return &_outputs; }
    inline Var* input(uint i) { return Node::input(i)->var(); }
    inline Var* output(uint i) { return Node::output(i)->var(); }
    inline uint type() const { CHECK_EXIST; return flags.get(NodeFlags::_op_type, NodeFlags::_op_type_nbits); }
    inline void set_type(OpType t) { CHECK_EXIST; flags.set(NodeFlags::_op_type, t, NodeFlags::_op_type_nbits); }
    
    Var* create_output(NanoVector shape, NanoString dtype);
    void init();

    // Op::forward should be call in constructor
    // A forwarded operator will suicide in after constructor
    void forward(Var* input);
    static string get_filename_from_jit_key(const string& jit_key, const string& suffix);
    static string op_name_to_file_name(const string& s);
    static string file_name_to_class_name(const string& s);
    Op();
    ~Op();
    
    virtual VarPtr grad(Var* out, Var* dout, Var* v, int v_index);
    virtual void grads(Var** douts, VarPtr* dins);
    virtual void infer_shape();
    virtual void run();
    virtual void jit_prepare(JK& jk);
    virtual void do_jit_prepare(JK& jk);
    virtual const char* name() const = 0;
    virtual void statistics(uint64_t& in, uint64_t& out, uint64_t& compute);
    virtual void do_prepare(JK& jk);
    virtual void do_run_after_prepare(JK& jk);
    virtual void do_run();
    virtual VarPtr duplicate();
    virtual void compile_optimize(string& src);
    virtual void graph_optimize();
    void jit_run();

    string name_ex() const;
    string get_jit_key(JK& jk);
    vector<pair<string,string>> get_jit_define();
    string get_hash_name();
};

std::ostream& operator<<(std::ostream& os, const Op* var);

extern string_view_map<jit_op_entry_t> jit_ops;
// jit_key_mapper: map origin jit_key -> tuned jit_key
extern string_view_map<string> jit_key_mapper;

#ifdef JIT
    #define DECLARE_jit_run void jit_run();
#else
    #define DECLARE_jit_run void jit_prepare(JK& jk) override;
#endif

} // jittor