// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"
#include "core/node.h"
#include "codegen/jit_key.h"
#include "utils/jit_cache_map.h"
#include "ops/composite/op_dispatch.h"

namespace jittor {

enum OpType {other=0, element=1, broadcast=2, reduce=3};
// Selection belongs to the execution plan / compiler invocation, never to an
// operator's capability flags. Nested preparation restores its caller's target.
BackendId execution_target_backend();
struct ExecutionBackendScope {
    int previous;
    explicit ExecutionBackendScope(BackendId backend);
    ~ExecutionBackendScope();
    ExecutionBackendScope(const ExecutionBackendScope&) = delete;
    ExecutionBackendScope& operator=(const ExecutionBackendScope&) = delete;
};
struct Op : Node {
    // Dense-only kernels receive explicit contiguous graph inputs at the
    // generated construction boundary, before storing their Var members.
    static constexpr bool accepts_storage_strides = false;
    static constexpr bool mutates_storage_inputs = false;
    virtual bool is_storage_view() const { return false; }
    static constexpr uint32 backend_mask = OpBackendAny;
    vector<VarPtr> outputs_holder;
    static int64 number_of_lived_ops;
    // Monotone count of every operator ever constructed; the auto-flush
    // pipeline measures how much graph was built since it last launched.
    static int64 number_of_created_ops;
    mutable OpId registered_op_id = 0;
    mutable shared_ptr<const OpDef> registered_definition;
    
    inline Caster<Var*, Node::input_t> inputs() { CHECK_EXIST; return &_inputs; }
    inline Caster<Var*, Node::output_t> outputs() { CHECK_EXIST; return &_outputs; }
    inline Var* input(uint i) { return Node::input(i)->var(); }
    inline Var* output(uint i) { return Node::output(i)->var(); }
    // The Op-private half of the flag word; see Var::set_flag and NodeFlags.
    inline void set_flag(OpFlags::Flags f, int a=1, int nbits=1)
        { flags.set_bit((int)f, a, nbits); }
    inline NodeFlags::nf_t flag(OpFlags::Flags f, int nbits=1) const
        { return flags.get_bit((int)f, nbits); }
    inline uint type() const { CHECK_EXIST; return flag(OpFlags::_op_type, OpFlags::_op_type_nbits); }
    inline void set_type(OpType t) { CHECK_EXIST; set_flag(OpFlags::_op_type, t, OpFlags::_op_type_nbits); }
    OpId type_id() const;
    void bind_definition(bool required = true) const;
    const OpDef& definition() const;
    BackendId execution_backend() const;
    bool executes_on_accelerator() const { return execution_backend() != BackendId::Cpu; }
    const OpImplementation& implementation() const;
    const Codegen& codegen() const;
    void prepare_fragment(JK& key);
    void optimize_generated_source(string& source);
    void prepare_codegen_key(JK& key);
    void prepare_execution(JK& key);
    void execute_prepared(JK& key);
    void run_registered();
    inline bool is_op(OpId id) const { return type_id() == id; }
    
    Var* create_output(NanoVector shape, NanoString dtype);
    void init();
    // Give every output the device its inputs are on, and refuse a mix of
    // two devices the way torch does. See Var::device_id.
    void propagate_device();

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
    void jit_run(JK& jk);

    string name_ex() const;
    string get_jit_key(JK& jk);
    vector<pair<string,string>> get_jit_define();
    string get_hash_name();
};

std::ostream& operator<<(std::ostream& os, const Op* var);

// The two process-wide kernel caches for non-fused ops: the compiled entry
// point by jit key, and the map from the key an op prepares to the key its
// tuned kernel was actually compiled under.
//
// Bounded, and keyed by owned strings. They used to be `string_view_map`s,
// which had neither property: see utils/jit_cache_map.h.
EXTERN_LIB jit_cache_map<jit_op_entry_t> jit_ops;
EXTERN_LIB jit_cache_map<string> jit_key_mapper;

#ifdef JIT
    #define DECLARE_jit_run void jit_run();
#else
    #define DECLARE_jit_run void jit_prepare(JK& jk) override;
#endif

} // jittor
