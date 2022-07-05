// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
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

namespace jittor {

DECLARE_FLAG(string, cache_path);
// DECLARE_FLAG(uint8, th_mode);
extern uint8 th_mode;

DEFINE_FLAG(int, try_use_32bit_index, 0,
    "If not overflow, try to use 32 bit type as index type.");

string_view_map<jit_op_entry_t> jit_ops;
string_view_map<string> jit_key_mapper;

int64 Op::number_of_lived_ops = 0;

Op::Op() {
    flags.set(NodeFlags::_var, 0);
    flags.set(NodeFlags::_cpu, 1);
    flags.flags |= ((amp_reg & 7) << NodeFlags::_prefer_32);
    number_of_lived_ops++;
    if (PREDICT_BRANCH_NOT_TAKEN(trace_py_var)) trace_data.record_node(this);
}

Op::~Op() {
    number_of_lived_ops--;
}

void Op::forward(Var* input) {
    flags.set(NodeFlags::_forwarded);
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

void Op::init() {
    infer_shape();
    bool manual_set_vnbb = flags.get(NodeFlags::_manual_set_vnbb)
        || _inputs.size()==0
        || (_outputs.size()==1 && _outputs.front().node->is_stop_grad());
    for (Var* v : inputs()) {
        if (!manual_set_vnbb) {
            v->flags.set(NodeFlags::_needed_by_backward);
        }
    }
    Var* need_sync = nullptr;
    for (Var* v : outputs()) {
        if (!manual_set_vnbb)
            v->flags.set(NodeFlags::_needed_by_backward);
        if (v->num < 0)
            need_sync = v;
    }
    if (need_sync) {
        exe.run_sync(vector<Var*>({need_sync}), false);
        CHECK(need_sync->num >= 0) << need_sync << "'s shape is error";
    }
    if (th_mode) {
        bool stop_grad = true;
        for (Var* v : inputs()) {
            if (!v->is_stop_grad()) {
                stop_grad = false;
                break;
            }
        }
        if (stop_grad)
            for (Var* v : outputs()) {
                v->set_stop_grad();
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
        bool has_cuda = flags.get(NodeFlags::_cuda);
        bool has_cpu = flags.get(NodeFlags::_cpu);
        CHECK(has_cuda || has_cpu);
        if (has_cuda && has_cpu && !use_cuda)
            flags.set(NodeFlags::_cuda, 0);
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
        if (use_cuda_op && flags.get(NodeFlags::_cuda)) {
            jk << "«JIT_cuda:1";
            flags.set(NodeFlags::_cpu, 0);
            // TODO: 64bit index in CUDA
            // use_int64_t = false;
        } else {
            if (use_cuda==2) {
                if (flags.get(NodeFlags::_cuda))
                    LOGf << "Op" << name() >> "'s vars are not allocated in cuda";
                else
                    LOGf << "Op" << name() << "doesn't have cuda version";
            }
            ASSERT(flags.get(NodeFlags::_cpu))
                << "Op" << name() << "doesn't have cpu version";
            jk << "«JIT_cpu:1";
            flags.set(NodeFlags::_cuda, 0);
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
    return os;
}

} // jittor