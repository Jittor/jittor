// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Zheng-Ning Liu <lzhengning@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "opt/pass/shared_reduce_pass.h"
#include "ops/op_register.h"
#include <set>
#include <fstream>
#include <algorithm>
#include "ops/reduce_op.h"

namespace jittor {

// Read the decimal number that starts at str[pos].
// Callers first verify that the matched token is followed by a digit.
int parse_int_at(const string& str, int pos) {
    string digits = "";
    for (; pos < (int)str.size() && str[pos] >= '0' && str[pos] <= '9'; pos++) digits += str[pos];
    return stoi(digits);
}

// Find the kernel launched from `call` whose body contains an atomic.
// Returns (kernel function name, index into `funcs`), or ("", -1).
std::pair<string, int> find_atomic_kernel(unique_ptr<KernelIR>& call, vector<unique_ptr<KernelIR>>& funcs) {
    string func_name = "";
    for (uint i = 0; i < call->children.size(); ++i) {
        auto& code = call->children[i]->get_attr(kir::code);
        if (code.substr(0, 5) == "func_" && code.find("<<<") != string::npos &&
            code.find(">>>") != string::npos) {
            func_name = code.substr(0, code.find("<<<"));
            break;
        }
    }
    for (uint i = 0; i < funcs.size(); ++i)
        if (funcs[i]->get_attr(kir::lvalue) == func_name) {
            bool has_atomic = false;
            funcs[i]->dfs([&](unique_ptr<KernelIR>& c) {
                if (c->get_attr(kir::code).find("atomic") != string::npos) has_atomic = true;
            });
            if (has_atomic) return std::make_pair(func_name, i);
        }
    return std::make_pair("", -1);
}

// Index into op->ops of the first reduce op the kernel mentions, or -1.
// Operand names inside a fused kernel are "op{i}_{name}".
int find_reduce_op_id(unique_ptr<KernelIR>& kernel, FusedOp* op) {
    std::set<int> reduce_ids;
    kernel->dfs([&](unique_ptr<KernelIR>& c) {
        auto& code = c->attrs[kir::code];
        size_t search_pos = 0;
        while (true) {
            auto pos = code.find("op", search_pos);
            if (pos == string::npos) break;
            if (pos + 2 >= code.size() || !isdigit(code[pos + 2])) {
                search_pos = pos + 2;
                continue;
            }
            int op_id = parse_int_at(code, pos + 2);
            ASSERT(op_id >= 0 && op_id < (int)op->ops.size());
            if (op->ops[op_id]->is_op(op_ids::reduce())) reduce_ids.insert(op_id);
            search_pos = pos + 1;
        }
    });
    if (reduce_ids.empty())
        return -1;
    else
        return *reduce_ids.begin();
}

// Replace every atomic in the kernel with a block-wide shared-memory reduction
// followed by a single atomic from thread 0:
//
//     atomicAdd(&op0_xp[i], acc);
//         -->
//     acc = shared_reduce<T, shared_reduce_add>(acc);
//     if (threadIdx.x == 0) atomicAdd(&op0_xp[i], acc);
//
// Statements already guarded by an `if` are skipped, which is what stops the
// loop from rewriting the guarded atomic it just produced.
void rewrite_atomics_to_shared_reduce(unique_ptr<KernelIR>& kernel) {
    while (true) {
        string op = "";
        unique_ptr<KernelIR>* target = nullptr;
        kernel->dfs([&](unique_ptr<KernelIR>& c) {
            string& code = c->attrs[kir::code];
            if (c->father && c->father->type == KernelIRType::branch) return;
            if (code.find("atomic") == 0) {
                // "atomicAdd(" -> "add"
                op = code.substr(6, code.find("(") - 6);
                ASSERT(op == "Add" || op == "And" || op == "Or" || op == "Xor") << op;
                op[0] = op[0] - 'A' + 'a';
                target = &c;
            }
            if (code.find("cuda_atomic_") == 0) {
                // "cuda_atomic_max(" -> "max"
                op = code.substr(12, code.find("(") - 12);
                ASSERT(op == "min" || op == "max" || op == "mul") << op;
                target = &c;
            }
        });
        if (target == nullptr) break;
        auto& stmt = *target;
        string code = stmt->attrs[kir::code];
        // second argument of the atomic: the per-thread accumulator
        string value = code.substr(code.rfind("),") + 2, code.rfind(");") - (code.rfind("),") + 2));
        if (value.find("(") != string::npos) {
            value = value.substr(value.find("(") + 1, value.find(")") - (value.find("(") + 1));
        }
        auto father = stmt->father;
        string dtype = stmt->father->find_define(value)->get_attr(kir::dtype);
        uint pos = 0;
        for (; pos < stmt->flist->size() && stmt->flist->at(pos) != stmt; pos++);
        ASSERT(pos < stmt->flist->size());
        stmt->attrs[kir::code] =
            value + " = shared_reduce<" + dtype + ", shared_reduce_" + op + ">(" + value + ");";
        father->push_back("if (threadIdx.x == 0) " + code, stmt->flist, true);
    }
}

// tn index -> the loop dimensions folded into that thread range
typedef map<int, vector<int>> tn_range_map;

// Split the thread ranges of one launch into the ones that iterate a reduced
// dimension and the ones that do not, and order the reduced ones first.
//
// Returns (position of the last reduced range in the new order, the new order,
// the tn -> dims map). The position is -1 when no range is reduced, which the
// caller does not currently handle; SharedReducePass only reaches here for a
// kernel that contains a reduce op, so at least one range is reduced.
std::tuple<int, vector<int>, tn_range_map> plan_reduce_thread_order(unique_ptr<KernelIR>& call, ReduceOp* rop) {
    vector<int> tns;
    tn_range_map tn_ranges;
    for (auto& define : call->children) {
        if (define->type != KernelIRType::define) continue;
        if (define->get_attr(kir::lvalue).substr(0, 2) != "tn") continue;
        int tn = stoi(define->get_attr(kir::lvalue).substr(2));
        // rvalue is "get_thread_range_log(thread_num_left, range3 * range4)";
        // skip past the first argument and collect every rangeN it mentions
        string rvalue = define->get_attr(kir::rvalue);
        rvalue = rvalue.substr(rvalue.find(','));
        vector<int> dims;
        while (true) {
            auto pos = rvalue.find("range");
            if (pos == string::npos) break;
            dims.push_back(parse_int_at(rvalue, pos + 5));
            rvalue = rvalue.substr(pos + 5);
        }
        tns.push_back(tn);
        tn_ranges[tn] = dims;
    }
    auto reduce_mask = rop->reduce_mask;
    vector<int> reduce_tns;
    vector<int> other_tns;
    for (uint i = 0; i < tns.size(); ++i) {
        auto is_reduce_dim = [&](int d) -> bool { return (1 << d) & reduce_mask; };
        int tn = tns[i];
        // a thread range may not mix reduced and non-reduced dimensions
        for (uint j = 1; j < tn_ranges[tn].size(); ++j)
            ASSERT(is_reduce_dim(tn_ranges[tn][j]) == is_reduce_dim(tn_ranges[tn][0]));
        if (is_reduce_dim(tn_ranges[tn][0]))
            reduce_tns.push_back(tn);
        else
            other_tns.push_back(tn);
    }
    vector<int> order;
    for (uint i = 0; i < reduce_tns.size(); ++i) order.push_back(reduce_tns[i]);
    for (uint i = 0; i < other_tns.size(); ++i) order.push_back(other_tns[i]);
    return std::make_tuple((int)reduce_tns.size() - 1, order, tn_ranges);
}

// Put reduced dimensions in the low thread-id bits and cap their combined
// width at one CUDA block. ParallelPass may insert balancing branches between
// its tn definitions, so this pass must update named definitions rather than
// assuming those nodes are contiguous.
void apply_reduce_thread_order(unique_ptr<KernelIR>& call, unique_ptr<KernelIR>& kernel, ReduceOp* rop) {
    auto plan = plan_reduce_thread_order(call, rop);
    int last_reduce = std::get<0>(plan);
    auto order = std::get<1>(plan);
    ASSERT(last_reduce >= 0);
    int ndim = order.size();
    for (int d = 0; d < ndim; ++d)
        ASSERT(std::find(order.begin(), order.end(), d) != order.end());

    uint thread_num_pos = 0;
    for (; thread_num_pos < call->children.size(); ++thread_num_pos) {
        auto& child = call->children[thread_num_pos];
        if (child->has_attr(kir::code) &&
            startswith(child->attrs[kir::code], "thread_num=1<<tn"))
            break;
    }
    ASSERT(thread_num_pos < call->children.size());

    // Snapshot ParallelPass's cumulative tn boundaries, then retain at most ten
    // reduced bits (1024 threads). Removed bits become serial loop iterations.
    for (int d = 0; d < ndim; ++d) {
        string next = d + 1 < ndim ? "tn" + std::to_string(d + 1) : "0";
        call->insert(thread_num_pos++, "int _srw" + std::to_string(d) +
            "=tn" + std::to_string(d) + "-" + next + ";");
    }
    call->insert(thread_num_pos++, "int _sr_left=10;");
    for (int i = 0; i <= last_reduce; ++i) {
        string width = "_srw" + std::to_string(order[i]);
        call->insert(thread_num_pos++, width + "=std::min(" + width + ",_sr_left);");
        call->insert(thread_num_pos++, "_sr_left-=" + width + ";");
    }
    for (int d = ndim - 1; d >= 0; --d) {
        string next = d + 1 < ndim ? "+tn" + std::to_string(d + 1) : "";
        call->insert(thread_num_pos++, "tn" + std::to_string(d) +
            "=_srw" + std::to_string(d) + next + ";");
    }

    string offset = "0";
    string reduce_bits = "";
    for (uint i = 0; i < order.size(); ++i) {
        int d = order[i];
        string width = "(tn" + std::to_string(d) + "-" +
            (d + 1 < ndim ? "tn" + std::to_string(d + 1) : "0") + ")";
        string tnum = "tnum" + std::to_string(d);
        auto* tnum_def = kernel->find_define(tnum);
        auto* tid_def = kernel->find_define("tid" + std::to_string(d));
        ASSERT(tnum_def && tid_def);
        tnum_def->attrs[kir::rvalue] = "1<<" + width;
        tid_def->attrs[kir::rvalue] =
            "(thread_id>>(" + offset + ")) & (" + tnum + "-1)";
        offset += "+" + width;
        if ((int)i <= last_reduce)
            reduce_bits += (reduce_bits.empty() ? "" : "+") + width;
    }

    // Low bits now cover all parallel reduction lanes, so each block owns an
    // output and the shared helper emits one global atomic for it.
    call->find_define("p1")->attrs[kir::rvalue] =
        "std::max(thread_num / std::min(1 << (" + reduce_bits + "), 1024), 1)";
    call->find_define("p2")->attrs[kir::rvalue] =
        "std::min(1 << (" + reduce_bits + "), 1024)";
}

extern int para_opt_level;

// The experimental level-4 CUDA path. apply_reduce_thread_order makes one block
// own one output, then shared_reduce performs a two-level fold: warp shuffle,
// one shared value per warp, and a final first-warp shuffle. That leaves one
// global atomic per block without the old 1024-value shared tree and its six
// barriers.
//
// The default remains WarpReducePass: the four-shape A/B recorded in the
// reduction strategy skill found this path 1.64% slower in aggregate and 16.6%
// slower on one representative shape. use_shared_reduce=0 also provides an
// explicit warp-only comparison at level 4. ROCm retains its prior atomics
// because its 64-lane wavefront needs a separate shuffle implementation.
void SharedReducePass::run() {
#ifdef IS_ROCM
    return;
#else
    auto parallel = op->get_loop_option("parallel");
    auto use_shared_reduce = op->get_loop_option("use_shared_reduce", 1);
    if (use_shared_reduce == 0) return;
    if (para_opt_level < 4) return;
    bool is_cuda = op->flag(OpFlags::_cuda);
    if (is_cuda) parallel = 1;
    if (!parallel) return;
    for (uint i = 0; i < ir->children.size(); ++i) {
        auto& call = ir->children[i];
        if (call->type != KernelIRType::loop) continue;
        auto found = find_atomic_kernel(call, ir->before);
        if (found.second == -1) continue;
        auto& kernel = ir->before[found.second];
        int reduce_op_id = find_reduce_op_id(kernel, op);
        if (reduce_op_id < 0) continue;
        apply_reduce_thread_order(call, kernel, dynamic_cast<ReduceOp*>(op->ops[reduce_op_id]));
        rewrite_atomics_to_shared_reduce(kernel);
    }
#endif
}

} // jittor
