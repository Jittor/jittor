// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "opt/expr.h"
#include "opt/pass_manager.h"
#include "opt/pass/warp_reduce_pass.h"
#include "utils/str_utils.h"

namespace jittor {

// A CUDA reduction ends with every thread adding its private partial sum
// straight into the output element:
//
//     float tmp0 = 0;
//     for (id3 = tid1; id3 < range23; id3 += tnum1) tmp0 += xp[...];
//     atomicAdd(&(yp[yid]), tmp0);
//
// The threads cooperating on one output are consecutive, so a whole warp
// usually shares `yid` and issues 32 atomics to the same address. Reducing
// inside the warp first leaves one atomic per warp. Measured on an RTX 4090
// with the shapes a diffusers UNet backward produces, that is 2.0x to 6.5x on
// the reduction itself ([8,384,32,32] over the spatial dims: 21.1us -> 8.8us;
// [8,128,64,64]: 49.4us -> 7.6us).
//
// Whether a warp really shares the address depends on the thread decomposition
// the parallel pass chose, so the emitted code checks it at run time instead of
// assuming it: the fast path is taken only when the warp is whole and every
// lane agrees on the index. Both are warp-uniform conditions, so the branch
// never splits the warp, and the fallback is the original statement.
//
// The rewrite changes the summation order, which for floating point changes the
// last bits. Atomics have no defined order to begin with, so this replaces one
// unspecified order with another -- it does not make a defined result differ.
static bool is_identifier(const string& s) {
    if (s.empty()) return false;
    if (!(isalpha(s[0]) || s[0] == '_')) return false;
    for (char c : s)
        if (!(isalnum(c) || c == '_')) return false;
    return true;
}

// Types __shfl_down_sync has an overload for.
static bool shuffleable(NanoString dtype) {
    return dtype == ns_float32 || dtype == ns_float64
        || dtype == ns_int32 || dtype == ns_int64
        || dtype == ns_uint32 || dtype == ns_uint64;
}

void WarpReducePass::run() {
#ifdef IS_ROCM
    // A wavefront is 64 lanes wide there, so the mask and the shuffle offsets
    // below are simply wrong; the pass has never been measured on ROCm either.
    return;
#else
    if (!op->flags.get(NodeFlags::_cuda)) return;
    if (op->get_loop_option("no_warp_reduce")) return;
    ir->dfs([&](unique_ptr<KernelIR>& i) {
        if (!i->has_attr("code")) return;
        auto& code = i->attrs["code"];
        if (!startswith(code, "atomicAdd")) return;
        auto src = expr::make(code);
        vector<unique_ptr<expr::Expr>> results;
        // The reduction emits &(yp[yid]); accept the unparenthesised form too.
        auto target = expr::make("atomicAdd(&(x[y]), z)");
        if (!expr::match(src.get(), target.get(), {"x", "y", "z"}, {}, results)) {
            results.clear();
            target = expr::make("atomicAdd(&x[y], z)");
            if (!expr::match(src.get(), target.get(), {"x", "y", "z"}, {}, results))
                return;
        }
        string pointer = results.at(0)->to_string(true);
        string index = results.at(1)->to_string(true);
        string value = results.at(2)->to_string(true);
        // Both are read twice by the emitted code, so they must be plain
        // locals: re-evaluating an expression could repeat a side effect.
        if (!is_identifier(index) || !is_identifier(value)) return;
        if (!is_identifier(pointer) || pointer.back() != 'p') return;
        // Resolve the output var to check its type. A failure here only means
        // this statement is left alone; unlike a correctness pass, declining to
        // optimize is a valid outcome.
        uint op_id, opvar_id;
        Op* vop;
        Var* var;
        try {
            pm->oc->get_op_var_by_name(pointer.substr(0, pointer.size()-1),
                                       op_id, opvar_id, vop, var);
        } catch (const std::exception&) {
            return;
        }
        if (!shuffleable(var->dtype())) return;
        string fallback = "atomicAdd(&(" + pointer + "[" + index + "]), " + value + ");";
        code =
            "{ auto _wr_mask = __activemask();"
            // A lane is its linear thread index modulo 32; blockDim.x alone
            // only says so for a one-dimensional block. The parallel pass
            // emits one today, but the leader test should not depend on that.
            " unsigned _wr_tid = threadIdx.x + blockDim.x *"
            " (threadIdx.y + blockDim.y * threadIdx.z);"
            " auto _wr_value = " + value + ";"
            " if (_wr_mask == 0xffffffffu && __all_sync(_wr_mask,"
            " " + index + " == __shfl_sync(_wr_mask, " + index + ", 0))) {"
            " for (int _wr_off = 16; _wr_off > 0; _wr_off >>= 1)"
            " _wr_value += __shfl_down_sync(_wr_mask, _wr_value, _wr_off);"
            " if ((_wr_tid & 31) == 0)"
            " atomicAdd(&(" + pointer + "[" + index + "]), _wr_value);"
            " } else " + fallback + " }";
        LOGvvvv << "warp reduce" << pointer << index;
    });
#endif
}

} // jittor
