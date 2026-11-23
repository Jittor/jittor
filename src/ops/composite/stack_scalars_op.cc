// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include <cstring>
#include "ops/composite/stack_scalars_op.h"
#include "ops/op_register.h"
#include "core/var.h"
#include "core/var_slices.h"

namespace jittor {

#ifndef JIT
static auto make_getitem = op_constructor<VarPtr, Var*, VarSlices&&>("getitem");
static auto make_reshape = op_constructor<VarPtr, Var*, NanoVector>("reshape");

StackScalarsOp::StackScalarsOp(vector<Var*>&& xs) : xs(xs) {
    USER_CHECK(this->xs.size()) << "stack_scalars needs at least one input";
    auto dtype = this->xs[0]->dtype();
    for (Var* x : this->xs)
        USER_CHECK(x->dtype() == dtype) << "stack_scalars inputs must share a dtype";
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    y = create_output(nullptr, dtype);
}

void StackScalarsOp::infer_shape() {
    for (Var* x : xs)
        USER_CHECK(x->num == 1) << "stack_scalars inputs must hold one element, got" << x->shape;
    y->set_shape({(int64)xs.size()});
}

VarPtr StackScalarsOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    VarSlices vs(1);
    vs.slices[0].set_int(v_index);
    return make_reshape(make_getitem(dout, move(vs)), v->shape);
}

void StackScalarsOp::jit_prepare(JK& jk) {
    jk << "«T:" << y->dtype();
}

#else // JIT
#ifdef JIT_cuda
namespace {
constexpr int kStackEntries = 256;
struct StackScalarsLaunch {
    const T* src[kStackEntries];
};
__global__ void stack_scalars_kernel(StackScalarsLaunch launch, T* dst, int count) {
    int i = threadIdx.x;
    if (i < count) dst[i] = launch.src[i][0];
}
} // namespace

void StackScalarsOp::jit_run() {
    T* dst = y->ptr<T>();
    StackScalarsLaunch launch;
    for (size_t base = 0; base < xs.size(); base += kStackEntries) {
        int count = (int)std::min<size_t>(kStackEntries, xs.size() - base);
        for (int i = 0; i < count; i++) launch.src[i] = xs[base + i]->ptr<T>();
        stack_scalars_kernel<<<1, kStackEntries>>>(launch, dst + base, count);
    }
}
#else
void StackScalarsOp::jit_run() {
    T* dst = y->ptr<T>();
    for (size_t i = 0; i < xs.size(); i++) dst[i] = xs[i]->ptr<T>()[0];
}
#endif // JIT_cuda
#endif // JIT

} // jittor
