// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/fused_op.h"
#include "core/var.h"
#include "codegen/op_compiler.h"
#include "runtime/profiler/profiler.h"
#include "utils/fast_shared_ptr.h"
#include "runtime/device.h"
#include "runtime/jit_policy.h"
#include "ops/op_register.h"
#include "utils/graph_build_profile.h"

namespace jittor {

#ifndef JIT

jit_cache_map<shared_ptr<FusedOpContext>> jit_fused_ops;

std::ostream& operator<<(std::ostream& os, const VarInfo& vi) {
    return os << vi.var << " type:" << vi.type;
}

int FusedOp::get_loop_option(const string& key, const int& _default) {
    auto iter = loop_options->find(key);
    return iter == loop_options->end() ? _default : iter->second;
}

loop_options_t& FusedOp::get_loop_options_tuned() {
    loop_options_tuned = *loop_options_origin;
    loop_options = &loop_options_tuned;
    return loop_options_tuned;
}

void FusedOp::update_jit_key() {
    JT_GBP_SCOPE(gbp_jit_key);
    JK& jk = get_jk();
    jk.clear();
    do_jit_prepare(jk);
}

void FusedOp::update_ops() {
    loop_options_merged.clear();
    loop_options_tuned.clear();
    loop_options = loop_options_origin = nullptr;

    _inputs.clear();
    _outputs.clear();
    vars.clear();
    op_index.clear();
    var_index.clear();
    for (uint i=0; i<ops.size(); i++)
        op_index[ops[i]] = i;
    for (Op* op : ops) {
        for (Var* o : op->outputs()) {
            if (o->loop_options) {
                if (loop_options_origin == nullptr)
                    loop_options_origin = &o->loop_options.data();
                else if (loop_options_origin != &o->loop_options.data()) {
                    // merge loop options
                    for (auto& kv : o->loop_options.data())
                        loop_options_merged[kv.first] = kv.second;
                }
            }
            // A var that has to stay in memory is an output of the fused op.
            // The verdict comes from the batch now; it used to be bit 0 of
            // o->custom_data, written there by the executor.
            if (var_stays_in_memory((Node*)o))
                _outputs.emplace_back((Node*)o, 0);
        }
    }

    if (loop_options_origin) {
        if (loop_options_merged.size()) {
            // merge loop_options_origin into loop_options_merged
            for (auto& kv : *loop_options_origin)
                loop_options_merged.emplace(kv);
        }
    } else {
        loop_options_origin = &loop_options_merged;
    }
    loop_options = loop_options_origin;

    ASSERT(outputs().size());
    LOGvvvv << "set fused output" << outputs();
    
    for (Op* opi : ops) {
        for (Var* i : opi->inputs()) {
            if (!var_index.count(i)) {
                var_index[i] = vars.size();
                vars.push_back({i, 0});
                _inputs.emplace_back((Node*)i);
            }
        }
        for (Var* o : opi->outputs()) {
            if (!var_index.count(o)) {
                var_index[o] = vars.size();
                // intermediate(can fuse) or output
                vars.push_back({o, var_stays_in_memory((Node*)o) ? 2 : 1});
            }
        }
    }
    LOGvvvv << "Var info" << vars;
}


FusedOp::FusedOp() {
    Op::number_of_lived_ops--;
    set_flag(OpFlags::_cpu, 1);
    set_flag(OpFlags::_cuda, 1);
}

FusedOp::FusedOp(const FusedOp& other) {
    Op::number_of_lived_ops--;
    set_flag(OpFlags::_cpu, other.flag(OpFlags::_cpu));
    set_flag(OpFlags::_cuda, other.flag(OpFlags::_cuda));
    registered_definition = other.registered_definition;
    registered_op_id = other.registered_op_id;
    ops = other.ops;
    op_index = other.op_index;
    var_index = other.var_index;
    edges = other.edges;
    vars = other.vars;
    loop_options_merged = other.loop_options_merged;
    loop_options_tuned = other.loop_options_tuned;
    loop_options = other.loop_options;
    loop_options_origin = other.loop_options_origin;
    context = other.context;
    context_owner = other.context_owner;
    // Deliberately not copied: it borrows a vector owned by the run_sync frame
    // that made `other`, and the copy is only kept for the compiler threads.
    // Anything that needed the verdict read it in update_ops(), before this.
    batch_var_fused = nullptr;
    batch_stamp_wanted = 0;
}

FusedOp::~FusedOp() {
    _inputs.clear();
    _outputs.clear();
    Op::number_of_lived_ops++;
}

void FusedOp::infer_shape() {
    for (Op* op : ops) {
        op->init();
    }
}

void FusedOp::statistics(uint64_t& in, uint64_t& out, uint64_t& compute) {
    in = out = compute = 0;
    for (auto& vi : vars) {
        compute = std::max(compute, (uint64_t)vi.var->num);
        if (vi.type == 0) in += vi.var->size;
        if (vi.type == 2) out += vi.var->size;
    }
}

void FusedOp::prepare_fused_key(JK& jk) {
    JT_GBP_SCOPE(gbp_jit_key);
    jk.clear();
    for (uint i=0; i<ops.size(); i++) {
        Op* op = ops[i];
        jk << "«opkey" << i << JK::val;
        jk << op->name();
        op->prepare_fragment(jk);
        const auto& identity = op->definition().compile_identity;
        if (!identity.empty()) add_jit_define(jk, "op_definition", i, identity);
    }
    jk << "«JIT:1";
    if (!executes_on_accelerator()) {
        // only cpu
        jk << "«JIT_cpu:1";
    } else {
        jk << "«JIT_cuda:1";
        add_cuda_math_jit_define(jk);
    }
    jk << "«graph:";
    for (auto& t : edges) {
        uint i,j,k,l;
        std::tie(i,j,k,l) = t;
        // Variable-width edge encoding.
        //
        // i is the producer: an op index for an edge internal to the fusion, or
        // ops.size()+var_index for a var coming from outside it (executor.cc).
        // j is the producer's output slot, k the consumer op index, l its input
        // slot. These four used to be written as hex2/hex1/hex2/hex1 -- 8 bits
        // for the op ids and 4 for the slots -- so a fusion holding more than
        // 255 ops plus external input vars wrapped: two structurally different
        // fusions produced the same jit key, the kernel cache lookup hit, and an
        // unrelated compiled kernel ran, giving a silently wrong result. That
        // was not a theoretical limit: F.interpolate(mode="bicubic") builds a
        // fusion of 292 ops over 296 vars, whose edge ids reach 462.
        //
        // JK::hex is variable length with no leading zeros, and '.' and ',' are
        // outside the hex alphabet, so each field is the maximal run of hex
        // digits between two delimiters and each edge the run between two
        // commas. The encoding is therefore injective for any field width: two
        // different edge sequences cannot produce the same string. ('«' must not
        // be used as a delimiter here -- it separates jit key entries.)
        jk << JK::hex(i) << '.' << JK::hex(j) << '.'
           << JK::hex(k) << '.' << JK::hex(l) << ',';
    }
    jk << "«var_info:" << JK::val;
    bool use_int64_t = false;
    for (auto& vi : vars) {
        jk << JK::hex1(vi.type) << JK::hex1(vi.var->shape.size());
        if (vi.type != 1 && (vi.var->num >= std::numeric_limits<int32_t>::max()
                || vi.var->storage_span_bytes() / vi.var->dsize() >= std::numeric_limits<int32_t>::max()))
            use_int64_t = true;
    }
    if (use_int64_t)
        jk << "«index_t:int64";
    else
        jk << "«index_t:int32";
    if (loop_options->size()) {
        if (get_loop_option("compile_shapes")) {
            jk << "«shapes:";
            for (auto& vi : vars) {
                jk << '[';
                for (auto a : vi.var->shape)
                    jk << a << ',';
                jk << "],";
            }
        }
        jk << "«choices:";
        // Every option, including the ones whose name starts with '_'.
        //
        // Those used to be skipped, which made two configurations that differ
        // only in an underscore-prefixed option share one compiled product --
        // and the key is the only thing that decides that. Nothing reads such
        // an option today, so nothing needed the exemption; the cost of the
        // exemption was that the day something did, the wrong kernel would run
        // and nothing would say so. (The other half of the plan's remedy,
        // refusing them where they are set, is not enough on its own: the
        // tuners write straight into `loop_options_tuned`, which no setter
        // sees.)
        for (auto& kv : *loop_options) {
            if (kv.first.size())
                jk << kv.first << ':' << kv.second << ',';
        }
    }
    jk.finilize();
}

void FusedOp::execute_fused_prepared(JK& jk) {
    // Keep the cache lookup independent from JK's reusable thread-local
    // buffer; preparation of the next op may overwrite it immediately.
    string jit_key = jk.to_string();
    if (auto* cached = jit_fused_ops.find(jit_key)) {
        // Take a reference, not a bare pointer: the cache is bounded, and the
        // kernel about to run may itself compile (relay ops) and evict.
        context_owner = *cached;
        context = context_owner.get();
        LOGvvv <<  "Jit fused op key found:" << jit_key << "jit op entry:" << (void*)context;
        // This used to also do `context->vrm.fop = this`, repointing the
        // cached context at the current stack FusedOp on every hit. That was
        // the only thing keeping that field from being read stale, and it
        // dangled again the moment run_sync returned. Nothing needs it: the
        // relay manager takes the FusedOp as an argument now.
        Profiler::record_and_run(context->entry, this, jit_key.c_str());
        return;
    }
    LOGvv << "Jit op key not found:" << jit_key;
    // compile JIT op
    context_owner = std::make_shared<FusedOpContext>();
    context = context_owner.get();
    context->setup(this);
    string prev_jit_key = jit_key;
    context->entry = OpCompiler::do_compile(this);
    string new_jit_key = get_jit_key(jk);
    // Two statements, not `t[a] = t[b] = ctx`: see the same spot in op.cc.
    jit_fused_ops[prev_jit_key] = context_owner;
    jit_fused_ops[new_jit_key] = context_owner;
    jit_key_mapper[prev_jit_key] = new_jit_key;
    LOGvv << "Get jit op entry:" << (void*)(context->entry);
    Profiler::record_and_run(context->entry, this, new_jit_key.c_str());
}

void FusedOpContext::setup(FusedOp* fop) {
    // Copies the numbering out of `fop`; keeps no reference to it.
    node_id.clear();
    for (int i=0; i<fop->ops.size(); i++)
        node_id[fop->ops[i]] = i;
    for (int i=0; i<fop->vars.size(); i++)
        node_id[fop->vars[i].var] = i;
}

int FusedOp::get_node_id(Node* node) {
    ASSERT(context);
    return context->node_id.at(node);
}

int FusedOp::has(Node* node) {
    ASSERT(context);
    return context->node_id.count(node);
}

#else // JIT
void FusedOp::jit_run() {
    for (uint i=0; i<ops.size(); i++) {
        LOGvvvv << "fuse run:" << ops[i] << ops[i]->inputs() << ops[i]->outputs();
        ops[i]->run_registered();
    }
}
#endif // JIT

}
