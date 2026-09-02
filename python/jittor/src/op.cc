// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <limits>

#include "node.h"
#include "op.h"
#include "var.h"
#include "op_compiler.h"
#include "profiler/profiler.h"
#include "mem/allocator.h"
#include "misc/cuda_flags.h"
#include "pybind/py_var_tracer.h"
#include "executor.h"
#include "var_holder.h"
#include "fused_op.h"
#include "graph.h"

namespace jittor {

DECLARE_FLAG(string, cache_path);
// DECLARE_FLAG(uint8, th_mode);
extern uint8 th_mode;

DEFINE_FLAG(int, try_use_32bit_index, 0,
    "If not overflow, try to use 32 bit type as index type.");

string_view_map<jit_op_entry_t> jit_ops;
string_view_map<string> jit_key_mapper;

int64 Op::number_of_lived_ops = 0;
int64 Op::number_of_created_ops = 0;

// Only Ops with disabled inputs have an entry. Node flag bits record whether
// an Op has already snapshotted and provide the common no-map-lookup fast path.
static unordered_map<int64, vector<int64>>& requires_grad_disabled_edges() {
    static auto* edges = new unordered_map<int64, vector<int64>>();
    return *edges;
}

bool lookup_requires_grad_disabled_edge(Node* source, Node* target) {
    auto& edges = requires_grad_disabled_edges();
    auto found = edges.find(target->id);
    if (found == edges.end()) return false;
    for (int64 source_id : found->second)
        if (source_id == source->id) return true;
    return false;
}

Op::Op() {
    flags.set(NodeFlags::_var, 0);
    set_flag(OpFlags::_cpu, 1);
    // The six amp bits are one field, so they move as one. `set_flag` with a
    // width also *clears* the field first, which `|=` did not -- identical
    // here (a fresh Op has them at zero) and the honest spelling of "this is
    // the amp field".
    set_flag(OpFlags::_prefer_32, amp_reg & ((1<<OpFlags::_amp_nbits)-1),
        OpFlags::_amp_nbits);
    number_of_lived_ops++;
    number_of_created_ops++;
    if (PREDICT_BRANCH_NOT_TAKEN(trace_py_var)) trace_data.record_node(this);
}

Op::~Op() {
    if (flag(OpFlags::_requires_grad_disabled))
        requires_grad_disabled_edges().erase(id);
    number_of_lived_ops--;
}

void Op::forward(Var* input) {
    set_flag(OpFlags::_forwarded);
    outputs_holder.emplace_back(input);
}

VarPtr Op::duplicate() {
    return nullptr;
}

VarPtr Op::grad(Var* out, Var* dout, Var* v, int v_index) {
    LOGw << "Grad of" << name() << "return zeros";
    return nullptr;
}

void Op::grads(Var** douts, VarPtr* dins) {
    LOGw << "Grads of" << name() << "return zeros";
}

Var* Op::create_output(NanoVector shape, NanoString dtype) {
    VarPtr vp(shape, dtype);
    Var* output = vp.ptr;
    outputs_holder.emplace_back(move(vp));
    return output;
}

// A pending Var marked _is_scalar is torch's CPU scalar: it has no data
// anywhere yet, and it is the `2` in `x * 2`, the `1` a gradient starts from,
// or a broadcast/unary of one (array_op.cc sets the flag on a shape-[1]
// source; broadcast_to_op.cc and unary_op.cc carry it forward). Such a value
// follows the operand it meets instead of forcing a device error.
//
// Both halves of that test are load-bearing, and each has a test:
// element count alone sweeps in a real one-element user tensor that already
// holds data; pendingness alone retargets a `jt.array(np.ones(1000))` the
// caller deliberately built on cuda:0 and merely has not synced yet --
// silently, where torch raises. See tests/backends/cuda/test_multi_device.py.
//
// One edge remains and is accepted: `jt.zeros(n)` is `unary(0).broadcast(n)`,
// and _is_scalar comes through the broadcast, so an unsynced zeros/ones does
// follow its operand. It is a constant with no data anywhere, produced
// bit-identically on either card, so this is constant placement rather than
// moving something the caller computed. device-placement.md §5 has the
// reasoning and why a third condition would cost more than it buys.
static inline bool is_pending_scalar(Var* v) {
    return !v->is_finished() && v->flag(VarFlags::_is_scalar);
}

// Move a pending scalar, and the pending subgraph that produces it, onto
// `dev`. Refuses (returning false) if that subgraph reaches data that already
// exists on another device -- then it is not a scalar constant being placed,
// it is a genuine cross-device use.
static bool retarget_pending(Var* v, int dev) {
    if (v->device_id == dev) return true;
    if (v->is_finished()) return false;
    vector<Var*> seen;
    vector<Node*> queue{v};
    for (size_t i = 0; i < queue.size(); i++) {
        auto node = queue[i];
        if (node->is_var()) {
            Var* var = node->var();
            if (var->is_finished()) {
                if (var->device_id != dev) return false;
                continue;
            }
            seen.push_back(var);
        }
        // A scalar's producer chain is a handful of nodes (array, broadcast,
        // unary). A long walk means this is not one, so stop rather than pay
        // for it on every op.
        if (seen.size() > 32) return false;
        for (auto& e : node->_inputs) {
            bool dup = false;
            for (auto* q : queue) if (q == e.node) { dup = true; break; }
            if (!dup) queue.push_back(e.node);
        }
    }
    for (Var* var : seen) var->device_id = dev;
    return true;
}

void Op::propagate_device() {
    // Placement only has a question to answer when more than one device is
    // visible; with one, every Var already carries the same index.
    static int device_count = get_device_count();
    if (device_count < 2) return;
    int dev = -1;
    for (Var* v : inputs()) {
        if (v->device_id < 0 || is_pending_scalar(v)) continue;
        if (dev < 0) dev = v->device_id;
        else if (dev != v->device_id)
            LOGf << "Expected all inputs to be on the same CUDA device, but found"
                << "cuda:" >> dev << "and cuda:" >> v->device_id << "for op" << name() >> "."
                << "\nMove one side with Var.to_device() / .to(\"cuda:N\") first.";
    }
    if (dev < 0)
        // Every input is a pending scalar (or device-less): the first one
        // decides and the rest follow it.
        for (Var* v : inputs())
            if (v->device_id >= 0) { dev = v->device_id; break; }
    if (dev < 0) return;
    for (Var* v : inputs())
        if (v->device_id != dev && v->device_id >= 0
                && !(is_pending_scalar(v) && retarget_pending(v, dev)))
            LOGf << "Expected all inputs to be on the same CUDA device, but found"
                << "cuda:" >> dev << "and cuda:" >> v->device_id << "for op" << name() >> "."
                << "\nMove one side with Var.to_device() / .to(\"cuda:N\") first.";
    for (Var* v : outputs())
        v->device_id = dev;
}

void Op::init() {
    bool first_init = !flag(OpFlags::_requires_grad_snapshot);
    bool has_disabled_input = false;
    bool has_first_order_only_input = false;
    bool all_inputs_stopped = _inputs.size() != 0;
    if (first_init) {
        set_flag(OpFlags::_requires_grad_snapshot);
        for (Var* v : inputs()) {
            bool disabled = v->flag(VarFlags::_requires_grad_disabled);
            has_disabled_input |= disabled;
            has_first_order_only_input |=
                v->flag(VarFlags::_first_order_only);
            all_inputs_stopped &= disabled || v->is_stop_grad();
        }
        if (has_disabled_input) {
            set_flag(OpFlags::_requires_grad_disabled);
            auto& sources = requires_grad_disabled_edges()[id];
            for (Var* v : inputs())
                if (v->flag(VarFlags::_requires_grad_disabled))
                    sources.push_back(v->id);
        }
    }
    infer_shape();
    if (first_init && _inputs.size() && !flag(OpFlags::_manual_device))
        propagate_device();
    if (first_init && has_first_order_only_input)
        for (Var* v : outputs())
            v->set_flag(VarFlags::_first_order_only);
    bool manual_set_vnbb = flag(OpFlags::_manual_set_vnbb)
        || _inputs.size()==0
        || (_outputs.size()==1 && _outputs.front().node->is_stop_grad());
    for (Var* v : inputs()) {
        if (!manual_set_vnbb) {
            v->set_flag(VarFlags::_needed_by_backward);
        }
    }
    Var* need_sync = nullptr;
    for (Var* v : outputs()) {
        if (!manual_set_vnbb)
            v->set_flag(VarFlags::_needed_by_backward);
        if (v->num < 0)
            need_sync = v;
    }
    if (need_sync) {
        exe.run_sync(vector<Var*>({need_sync}), false);
        CHECK(need_sync->num >= 0) << need_sync << "'s shape is error";
    }
    if (first_init && _inputs.size()) {
        if (all_inputs_stopped && has_disabled_input) {
            for (Var* v : outputs())
                v->set_flag(VarFlags::_requires_grad_disabled);
        } else if (all_inputs_stopped && th_mode) {
            for (Var* v : outputs()) {
                v->set_stop_grad();
            }
        }
    }
}

void Op::compile_optimize(string& src) {}

void Op::infer_shape() {}
void Op::run() {}
void Op::jit_prepare(JK& jk) {}
void Op::graph_optimize() {}

string Op::name_ex() const {
    string a=name();
    if (ns.data) {
        a += '.';
        a += ns.to_cstring();
    }
    return a;
}

string Op::get_jit_key(JK& jk) {
    jk.clear();
    do_jit_prepare(jk);
    return jk.to_string();
}

vector<pair<string,string>> Op::get_jit_define() {
    return parse_jit_keys(get_jit_key(get_jk()));
}

string Op::get_hash_name() {
    string hash_name;
    std::stringstream ss;
    JK& jk = get_jk();
    do_prepare(jk);
    ss << std::hex << std::hash<string>()(jk.to_string());
    hash_name = ss.str();
    return hash_name;
}

void Op::do_jit_prepare(JK& jk) {
    memcheck_all_exist();
    jk << name();
    auto pre_size = jk.size;
    jit_prepare(jk);
    if (jk.size == pre_size) {
        // not a jit op
        bool has_cuda = flag(OpFlags::_cuda);
        bool has_cpu = flag(OpFlags::_cpu);
        CHECK(has_cuda || has_cpu);
        if (has_cuda && has_cpu && !use_cuda)
            set_flag(OpFlags::_cuda, 0);
        jk.clear();
    } else {
        bool use_int64_t = false;
        // TODO: fused op do not have inputs,
        //   check use_cuda_op from outputs may not be enough
        bool use_cuda_op = use_cuda;
        for (Var* var : inputs()) {
            if (var->num >= std::numeric_limits<int32_t>::max())
                use_int64_t = true;
        }
        for (Var* var : outputs()) {
            if (var->num >= std::numeric_limits<int32_t>::max())
                use_int64_t = true;
        }
        jk << "«JIT:1";
        if (use_cuda_op && flag(OpFlags::_cuda)) {
            jk << "«JIT_cuda:1";
            set_flag(OpFlags::_cpu, 0);
            // TODO: 64bit index in CUDA
            // use_int64_t = false;
        } else {
            if (use_cuda==2) {
                if (flag(OpFlags::_cuda))
                    LOGf << "Op" << name() >> "'s vars are not allocated in cuda";
                else
                    LOGf << "Op" << name() << "doesn't have cuda version";
            }
            ASSERT(flag(OpFlags::_cpu))
                << "Op" << name() << "doesn't have cpu version";
            jk << "«JIT_cpu:1";
            set_flag(OpFlags::_cuda, 0);
        }
        if (try_use_32bit_index) use_int64_t = false;
        if (use_int64_t)
            jk << "«index_t:int64";
        else
            jk << "«index_t:int32";
    }
    jk.finilize();
}

void Op::do_prepare(JK& jk){
    jk.clear();
    do_jit_prepare(jk);
}

void Op::do_run_after_prepare(JK& jk) {
    if (!jk.empty())
        jit_run(jk);
    else
        run();
}

void Op::do_run() {
    JK& jk = get_jk();
    do_prepare(jk);
    do_run_after_prepare(jk);
}

string Op::get_filename_from_jit_key(const string& jit_key, const string& suffix) {
    auto iter = jit_key_mapper.find(jit_key);
    string s = iter==jit_key_mapper.end() ? jit_key : iter->second;
    std::stringstream ss;
    if (s.size() > 100) {
        ss << s.substr(0, 90) << "...hash_"
            << std::hex << std::hash<string>()(s);
    } else {
        ss << s << "_hash_" << 
            std::hex << std::hash<string>()(s);
    }
    s = ss.str();
    for (char& c : s) {
        if (!((c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9')))
            c = '_';
    }
    #ifndef _WIN32
    string filename = cache_path + "/jit/";
    #else
    string filename = cache_path + "\\jit\\";
    #endif
    filename += s;
    filename += "_op";
    filename += suffix;
    return filename;
}

// convert xxx.yyy -> xxx
string Op::op_name_to_file_name(const string& s) {
    auto pos = s.find('.');
    return pos == string::npos ? s : s.substr(0, pos);
}
// convert xxx_xxx -> XxxXxx
string Op::file_name_to_class_name(const string& s) {
    char prev = '_';
    string res;
    res.reserve(s.size());
    for (char c : s) {
        if (c != '_') {
            if (prev == '_')
                res += c-'a'+'A';
            else
                res += c;
        }
        prev = c;
    }
    return res;
}

void Op::jit_run(JK& jk) {
    const char* jit_key = jk.to_cstring();
    auto iter = jit_ops.find(jit_key);
    if (iter != jit_ops.end()) {
        LOGvvv <<  "Jit op key found:" << jit_key << "jit op entry:" << (void*)iter->second;
        Profiler::record_and_run(iter->second, this, jit_key);
        return;
    }
    LOGvv << "Jit op key not found:" << jit_key;
    // compile JIT op
    string prev_jit_key = jit_key;
    auto op_entry = OpCompiler::do_compile(this);
    string new_jit_key = get_jit_key(jk);
    jit_ops[new_jit_key] = jit_ops[prev_jit_key] = op_entry;
    jit_key_mapper[prev_jit_key] = new_jit_key;
    LOGvv << "Get jit op entry:" << (void*)op_entry;
    Profiler::record_and_run(op_entry, this, new_jit_key.c_str());
}

void Op::statistics(uint64_t& in, uint64_t& out, uint64_t& compute) {
    in = out = compute = 0;
    for (auto& e : _inputs) {
        auto var = e.node->var();
        if (e.back->index<0) continue;
        in += var->size;
        compute = std::max(compute, (uint64_t)var->num);
    }
    for (auto& e : _outputs) {
        auto var = e.node->var();
        if (e.index<0) continue;
        out += var->size;
        compute = std::max(compute, (uint64_t)var->num);
    }
}

std::ostream& operator<<(std::ostream& os, const Op* op) {
    if (!op) return os << "Op(0)";
    os << "Op(" << op->id
        << ':' << op->forward_liveness
        << ':' << op->backward_liveness
        << ':' << op->pending_liveness
        << ":i" << op->_inputs.size()
        << ":o" << op->_outputs.size()
        << ":s" << op->is_finished()
        << ":g" << !op->is_stop_grad()
        << "," << op->name_ex();
    if (op->_outputs.size()>1)
        os << "->...";
    else if (op->_outputs.size() == 1) {
        auto v = (Var*)op->_outputs.front().node;
        if (v->name.size())
            os << "->" << v->name;
        else
            os << "->" << v->id;
    }
    os << ')';
    if (trace_py_var) {
        os << '{';
        print_node_trace(op, os);
        os << '}';
    }
    if (op->name_ex() == "fused") {
        os << ((FusedOp*)op)->ops;
    }
    return os;
}

} // jittor
