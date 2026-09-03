// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Zheng-Ning Liu <lzhengning@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "opt/pass/shared_reduce_pass.h"
#include <set>
#include <fstream>
#include <algorithm>
#include "ops/reduce_op.h"

namespace jittor {

// Read the decimal number that starts at str[pos].
// NOTE: throws std::invalid_argument when str[pos] is not a digit. Every
// caller below only reaches it after matching a token that is always followed
// by digits ("tn", "range", "op"), which is why it has no guard.
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
            int op_id = parse_int_at(code, pos + 2);
            ASSERT(op_id >= 0 && op_id < (int)op->ops.size());
            if (op->ops[op_id]->name() == string("reduce")) reduce_ids.insert(op_id);
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

// Rewrite the thread range setup at the call site and the tid/tnum decoding
// inside the kernel so that the reduced dimensions occupy the low bits of the
// thread id, i.e. one CUDA block covers a whole reduction. That is what makes
// the shared-memory reduction emitted above legal.
void apply_reduce_thread_order(unique_ptr<KernelIR>& call, unique_ptr<KernelIR>& kernel, ReduceOp* rop) {
    auto plan = plan_reduce_thread_order(call, rop);
    int last_reduce = std::get<0>(plan);
    auto order = std::get<1>(plan);
    auto tn_ranges = std::get<2>(plan);

    // ParallelPass emitted the tn defines, the accumulation statements and the
    // final "thread_num=..." in a fixed order; walk that block and overwrite
    // it statement by statement.
    uint pos = 0;
    for (auto& child : call->children) {
        if (child->type == KernelIRType::define && child->get_attr(kir::lvalue).substr(0, 2) == "tn") break;
        ++pos;
    }
    for (int tn : order) {
        string range_expr = "";
        for (int d : tn_ranges[tn]) range_expr += " * range" + std::to_string(d);
        call->children[pos]->attrs[kir::lvalue] = "tn" + std::to_string(tn);
        call->children[pos]->attrs[kir::rvalue] =
            "get_thread_range_log(thread_num_left, " + range_expr.substr(2) + ")";
        ++pos;
    }
    if (last_reduce == 0) {
        // a single reduced range: give it at least a warp
        string cur = "tn" + std::to_string(order[0]);
        call->children[pos++]->attrs[kir::code] = cur + " = std::max(" + cur + ", 5);";
    }
    for (uint i = 0; i + 1 < order.size(); ++i) {
        string cur = "tn" + std::to_string(order[i]);
        string next = "tn" + std::to_string(order[i + 1]);
        call->children[pos++]->attrs[kir::code] = next + " = " + cur + " + " + next + ";";
        if ((int)i + 1 == last_reduce) {
            call->children[pos++]->attrs[kir::code] = next + " = std::max(" + next + ", 5);";
        }
    }
    call->children[pos]->attrs[kir::code] = "thread_num=1<<tn" + std::to_string(order.back()) + ";";

    pos = 0;
    for (auto& child : kernel->children) {
        if (child->type == KernelIRType::define && child->get_attr(kir::lvalue).substr(0, 4) == "tnum") break;
        ++pos;
    }
    for (uint i = 0; i < order.size(); ++i) {
        auto& tnum_def = kernel->children[pos++];
        string prev_tn = "tn" + (i == 0 ? std::to_string(order.size()) : std::to_string(order[i - 1]));
        string tn = "tn" + std::to_string(order[i]);
        string tnum = "tnum" + std::to_string(order[i]);
        tnum_def->attrs[kir::lvalue] = tnum;
        tnum_def->attrs[kir::rvalue] = "1<<(" + tn + "-" + prev_tn + ")";
        auto& tid_def = kernel->children[pos++];
        tid_def->attrs[kir::lvalue] = "tid" + std::to_string(order[i]);
        tid_def->attrs[kir::rvalue] = "(thread_id>>" + prev_tn + ") & (" + tnum + "-1)";
    }
    // block size must cover all reduced dimensions, so it is 1<<tn{last
    // reduced range}, capped at 1024
    call->find_define("p1")->attrs[kir::rvalue] =
        string("std::max(thread_num / std::min(1 << tn") + std::to_string(order[last_reduce]) + ", 1024), 1)";
    call->find_define("p2")->attrs[kir::rvalue] =
        string("std::min(1 << tn") + std::to_string(order[last_reduce]) + ", 1024)";
}

extern int para_opt_level;

// Off by default: para_opt_level's default is 3 (loop_var_analyze_pass.cc), so
// the guard below returns before anything happens and no generated kernel in a
// stock build contains shared_reduce. That is deliberate, not an oversight.
//
// Measured on an RTX 4090, float32, four-dimensional reductions over (0,2,3) --
// the shapes a diffusers UNet backward produces -- as average device time per
// kernel:
//
//     shape           atomics only   + WarpReducePass   + this pass
//     8x384x32x32        157.0us          15.7us          25.3us
//     8x128x64x64         92.1us          14.0us          31.3us
//     16x192x32x32       159.2us          15.0us          25.3us
//     32x64x56x56        171.0us          18.1us          34.8us
//
// Both strategies remove the atomic contention (6-10x over plain atomics), but
// the warp shuffle in WarpReducePass is 1.6-2.0x faster than folding through
// shared memory: shared_reduce needs a 1024-entry __shared__ array, six
// __syncthreads(), and a volatile-memory warp tail, against five register-only
// __shfl_down_sync. What this pass still has over it is the summation order --
// relative error 2.3e-7 against 3.5e-7 -- and independence from whether a warp
// happens to share an output address.
//
// Raising the default would need a block-level fold built on the shuffle
// (warp reduce -> one value per warp -> shared memory -> one atomic), not this
// one. tests/backends/cuda/test_shared_reduce.py pins that the pass still
// produces correct code when switched on;
// agent/skills/cuda-reduction-strategy-comparison/ has the measurement method.
void SharedReducePass::run() {
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
}

} // jittor
