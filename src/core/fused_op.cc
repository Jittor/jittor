// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/fused_op.h"
#include "core/var.h"
#include "codegen/op_compiler.h"
#include "codegen/nvrtc_compiler.h"
#include "runtime/profiler/profiler.h"
#include "utils/fast_shared_ptr.h"
#include "runtime/device.h"
#include "runtime/jit_policy.h"
#include "runtime/tier0_ops.h"
#include "runtime/traversal_state.h"
#include "ops/op_register.h"
#include "utils/graph_build_profile.h"
#include <cstdlib>
#include <sstream>

namespace jittor {

#ifndef JIT

jit_cache_map<shared_ptr<FusedOpContext>> jit_fused_ops;

std::ostream& operator<<(std::ostream& os, const VarInfo& vi) {
    return os << vi.var << " type:" << vi.type;
}

// A fused kernel reads a strided or broadcast operand by recovering each axis'
// coordinate from the flat index, one division and one modulo per axis that
// moves it, for every such operand of every op. Over a [2]*20 state with
// broadcasts along different axes that was ~700 runtime divisions per element
// (a 50-op Ising-energy backward: 9.3 ms for 2^20 floats). With the shapes
// compiled in, every divisor is a constant -- a power of two here, so a shift
// -- at the price of one kernel per distinct shape, which a high-rank operand
// (a state, not a batch) rarely varies.
EXTERN_LIB int para_opt_level;

DEFINE_FLAG(int, auto_compile_shapes_rank, 8,
    "Compile the shapes into an accelerator kernel that reads a strided or "
    "broadcast operand of at least this rank, so its index arithmetic folds. "
    "0 disables it.");

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

FusedOp::Snapshot<Var*> FusedOp::snapshot_outputs_slow(Op* op) const {
    unique_ptr<vector<Var*>> owned(new vector<Var*>());
    for (Var* o : op->outputs()) owned->push_back(o);
    return Snapshot<Var*>(move(owned));
}

FusedOp::Snapshot<pair<Var*, int>> FusedOp::snapshot_inputs_slow(Op* op) const {
    unique_ptr<vector<pair<Var*, int>>> owned(new vector<pair<Var*, int>>());
    for (auto ve : op->_inputs) owned->emplace_back(ve.node->var(), ve.reverse().index);
    return Snapshot<pair<Var*, int>>(move(owned));
}

pair<Op*, int> FusedOp::snapshot_producer_slow(Var* v) const {
    int slot = 0;
    if (!v->_inputs.empty()) slot = v->_inputs.front().reverse().index;
    return {v->input(), slot};
}

static void update_ops_impl(FusedOp& fop, bool build_edges) {
    const uint n_ops = fop.ops.size();
    if (n_ops) {
        fop.float32_precision = fop.ops.back()->float32_precision;
        fop.launch_origin = fop.ops.back()->launch_origin;
    }
    if (!fop.loop_options_merged.empty()) fop.loop_options_merged.clear();
    if (!fop.loop_options_tuned.empty()) fop.loop_options_tuned.clear();
    fop.loop_options = fop.loop_options_origin = nullptr;
    fop.streamed_inputs = 0;

    fop._inputs.clear();
    fop._outputs.clear();
    fop.vars.clear();
    fop.op_index.reset(n_ops);
    // Every var indexed below is an edge of one of these ops, so the edge
    // count bounds them.
    size_t var_bound = 0;
    for (uint i = 0; i < n_ops; i++) {
        Op* op = fop.ops[i];
        fop.op_index[op] = i;
        auto ins = fop.snapshot_inputs(op);
        var_bound += ins.end() - ins.begin();
        auto outs = fop.snapshot_outputs(op);
        var_bound += outs.end() - outs.begin();
        for (Var* o : outs) {
            if (o->loop_options) {
                if (fop.loop_options_origin == nullptr)
                    fop.loop_options_origin = &o->loop_options.data();
                else if (fop.loop_options_origin != &o->loop_options.data()) {
                    // merge loop options
                    for (auto& kv : o->loop_options.data())
                        fop.loop_options_merged[kv.first] = kv.second;
                }
            }
            // A var that has to stay in memory is an output of the fused op.
            // The verdict comes from the batch now; it used to be bit 0 of
            // o->custom_data, written there by the executor.
            if (fop.var_stays_in_memory((Node*)o))
                fop._outputs.emplace_back((Node*)o, 0);
        }
    }

    if (fop.loop_options_origin) {
        if (fop.loop_options_merged.size()) {
            // merge loop_options_origin into loop_options_merged
            for (auto& kv : *fop.loop_options_origin)
                fop.loop_options_merged.emplace(kv);
        }
    } else {
        fop.loop_options_origin = &fop.loop_options_merged;
    }
    fop.loop_options = fop.loop_options_origin;

    if (PREDICT_BRANCH_NOT_TAKEN(fop.outputs().size() == 0)) {
        // This segment was classified with no output that has to stay in
        // memory, so the assertion below is about to fire. Dump the batch
        // verdict and the marks the planner read it out of: an op or var whose
        // tflag is not this batch's stamp means the traversal that built the
        // verdict was looking at a different one.
        auto& tstate = runtime_traversal_state();
        std::ostringstream os;
        os << "fused segment with no in-memory output: ops=" << fop.ops.size()
           << " batch_stamp_wanted=" << fop.batch_stamp_wanted
           << " batch_var_fused=" << (fop.batch_var_fused ? "set" : "null")
           << " stamp_count=" << tstate.stamp_count()
           << " active_epochs=" << tstate.active_epochs();
        for (Op* op : fop.ops) {
            os << "\n  op " << op->name() << " addr=" << (void*)op
               << " tflag=" << op->tflag
               << " batch_stamp=" << op->batch_stamp
               << " outputs=" << op->outputs().size()
               << " holder=" << op->outputs_holder.size()
               << " inputs=" << op->inputs().size();
            for (Var* o : op->outputs()) {
                os << "\n     out tflag=" << o->tflag
                   << " batch_stamp=" << o->batch_stamp
                   << " stays=" << (int)fop.var_stays_in_memory((Node*)o);
                if (fop.batch_var_fused && o->batch_stamp == fop.batch_stamp_wanted) {
                    int idx = o->batch_index_at(fop.batch_stamp_wanted);
                    os << " batch_index=" << idx;
                    if (idx >= 0 && (uint)idx < fop.batch_var_fused->size())
                        os << " var_fused=" << (*fop.batch_var_fused)[idx];
                }
            }
        }
        LOGw << os.str();
    }
    ASSERT(fop.outputs().size());
    LOGvvvv << "set fused output" << fop.outputs();

    fop.var_index.reset(var_bound);
    for (uint fid1 = 0; fid1 < n_ops; fid1++) {
        Op* opi = fop.ops[fid1];
        int iid = 0;
        for (auto& in : fop.snapshot_inputs(opi)) {
            Var* i = in.first;
            auto inserted = fop.var_index.try_emplace(i, (int32)fop.vars.size());
            if (inserted.second) {
                fop.vars.push_back({i, 0});
                fop._inputs.emplace_back((Node*)i);
            }
            if (build_edges && in.second >= 0) {
                int iop_id;
                int iv_id;
                int producer_id = -1;
                pair<Op*, int> producer{nullptr, 0};
                if (n_ops > 1) {
                    producer = fop.snapshot_producer(i);
                    if (producer.first) producer_id = fop.op_index.get(producer.first);
                }
                if (producer_id >= 0) {
                    iop_id = producer_id;
                    iv_id = producer.second;
                } else {
                    iv_id = inserted.first;
                    // add iv_id, prevent iv_id jit key overflow
                    iop_id = n_ops + iv_id;
                }
                fop.edges.emplace_back(iop_id, iv_id, fid1, iid++);
            }
        }
        for (Var* o : fop.snapshot_outputs(opi)) {
            auto inserted = fop.var_index.try_emplace(o, (int32)fop.vars.size());
            if (inserted.second) {
                // intermediate(can fuse) or output
                fop.vars.push_back({o, fop.var_stays_in_memory((Node*)o) ? 2 : 1});
            }
        }
    }
    LOGvvvv << "Var info" << fop.vars;
}

void FusedOp::update_ops() { update_ops_impl(*this, false); }
void FusedOp::update_ops_and_edges() { update_ops_impl(*this, true); }


FusedOp::FusedOp() {
    Op::number_of_lived_ops--;
    set_flag(OpFlags::_cpu, 1);
    set_flag(OpFlags::_cuda, 1);
}

FusedOp::FusedOp(const FusedOp& other) {
    float32_precision = other.float32_precision;
    launch_origin = other.launch_origin;
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
    batch_plan = nullptr;
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
    // While no tensor has an explicit placement, every op's requested backend
    // is this thread's execution target -- `graph_placement` has nothing to
    // say -- so it is asked once here, not three calls deep in the fragment of
    // every member and again for the segment.
    const bool uniform = !explicit_tensor_placement_made.load(std::memory_order_relaxed);
    const BackendId requested = uniform ? execution_target_backend() : BackendId::Cpu;
    for (uint i=0; i<ops.size(); i++) {
        Op* op = ops[i];
        jk << "«opkey" << i << JK::val;
        jk << op->name();
        const auto backend = uniform ? op->execution_backend_for(requested) : op->execution_backend();
        if ((backend == BackendId::Cpu || backend == BackendId::Cuda)
                && ops.size() == 1 && op->is_op(op_ids::array())
                && op->output(0)->num == 1) {
            // Avoid resolving a codegen callback for a standalone CPU or
            // CUDA scalar array. Its fragment is just the output dtype.
            jk << "«T:" << op->output(0)->dtype();
        } else {
            op->prepare_fragment_on(jk, backend);
        }
        const auto& identity = op->definition().compile_identity;
        if (!identity.empty()) add_jit_define(jk, "op_definition", i, identity);
    }
    jk << "«JIT:1";
    const bool on_accelerator = uniform
        ? execution_backend_for(requested) != BackendId::Cpu : executes_on_accelerator();
    // SharedReducePass reads the thread split's `rangeN` names, which compiled
    // shapes replace with numbers; a kernel it will rewrite keeps its ranges.
    const bool has_loop_options = !loop_options->empty();
    const int shared_reduce = has_loop_options ? get_loop_option("use_shared_reduce", 1) : 1;
    const bool shared_reduce_runs = shared_reduce != 0
        && (para_opt_level >= 4 || shared_reduce == 2);
    if (on_accelerator && auto_compile_shapes_rank > 0 && !shared_reduce_runs
            && !(has_loop_options && get_loop_option("compile_shapes"))) {
        bool high_rank_strided = false;
        for (auto& vi : vars)
            if (vi.type == 0 && (int)vi.var->shape.size() >= auto_compile_shapes_rank
                    && !vi.var->is_contiguous()) {
                high_rank_strided = true;
                break;
            }
        if (high_rank_strided) {
            if (loop_options_origin != &loop_options_merged) {
                for (auto& kv : *loop_options_origin) loop_options_merged.emplace(kv);
                loop_options_origin = &loop_options_merged;
            }
            loop_options_merged["compile_shapes"] = 1;
            if (loop_options == &loop_options_tuned) loop_options_tuned["compile_shapes"] = 1;
            else loop_options = loop_options_origin;
        }
    }
    if (!on_accelerator) {
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
        // Rank can exceed one hex digit, so each entry is delimited.
        jk << JK::hex(vi.type) << JK::hex(vi.var->shape.size()) << ',';
        if (vi.type != 1 && (vi.var->num >= std::numeric_limits<int32_t>::max()
                || (vi.var->storage_strides.size() && vi.var->num
                    && vi.var->storage_span_bytes() / vi.var->dsize() >= std::numeric_limits<int32_t>::max())))
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
    if (PREDICT_BRANCH_NOT_TAKEN(jit_tier_mode() == 0) && tier0_try_execute_fused(this, jk))
        return;
    if (jit_fused_ops.find_copy(jk.to_cstring(), (size_t)jk.size, context_owner)) {
        // Take a reference under the cache lock, not a bare pointer: the cache
        // is bounded, and concurrent inserts or relay compiles may rehash or
        // evict entries.
        context = context_owner.get();
        if (PREDICT_BRANCH_NOT_TAKEN(profiler_enable)) {
            // Own the key before running: relay ops may overwrite `jk`.
            string jit_key = jk.to_string();
            LOGvvv <<  "Jit fused op key found:" << jit_key << "jit op entry:" << (void*)context;
            Profiler::record_and_run(context->entry, this, jit_key.c_str());
            return;
        }
        LOGvvv <<  "Jit fused op key found:" << jk.to_cstring() << "jit op entry:" << (void*)context;
        // This used to also do `context->vrm.fop = this`, repointing the
        // cached context at the current stack FusedOp on every hit. That was
        // the only thing keeping that field from being read stale, and it
        // dangled again the moment run_sync returned. Nothing needs it: the
        // relay manager takes the FusedOp as an argument now.
        context->entry(this);
        return;
    }
    if (tier0_try_execute_fused(this, jk))
        return;
    if (nvrtc_try_compile_and_run(this, jk))
        return;
    string jit_key = jk.to_string();
    LOGvv << "Jit op key not found:" << jit_key;
    // compile JIT op
    context_owner = std::make_shared<FusedOpContext>();
    context = context_owner.get();
    context->setup(this);
    string prev_jit_key = jit_key;
    context->entry = OpCompiler::do_compile(this);
    string new_jit_key = get_jit_key(jk);
    // Thread-safe locked writes (background Tier-1 promotion may insert concurrently).
    jit_fused_ops.set(prev_jit_key, context_owner);
    jit_fused_ops.set(new_jit_key, context_owner);
    jit_key_mapper.set(prev_jit_key, new_jit_key);
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
