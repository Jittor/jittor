// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "core/var.h"
#include "mem/allocator.h"
#include "ops/composite/numpy_code_op.h"
#include "ops/op_register.h"

#ifndef JIT

namespace jittor {
    
static auto make_numpy_code = op_constructor<VarPtr, NanoVector, NanoString, vector<Var*>&&, NumpyFunc, NumpyResult&&>("numpy_code");

// Where grad() parked the two gradient operands in the backward op's input
// list. They live in NumpyResult::ints because that is what travels through
// the op constructor, and run() strips them before the callback sees the dict,
// so user code still observes only the documented keys.
static const char* dout_input_index_key = "__dout_input_index";
static const char* f_outputs_input_index_key = "__f_outputs_input_index";

NumpyCodeOp::NumpyCodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, NumpyFunc&& forward, vector<NumpyFunc>&& sbackward)
    : _inputs(inputs), forward(move(forward))
{
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    _outputs.push_back(create_output(shape, dtype));
    USER_CHECKop(_inputs.size(),<=,10) << "numpy_code supports at most ten inputs";
    USER_CHECK(_outputs[0]->num >= 0) << "numpy_code requires a known nonnegative output shape";
    for (int i=0; i<sbackward.size(); i++) {
        backward.push_back(sbackward[i]);
    }
}

NumpyCodeOp::NumpyCodeOp(vector<NanoVector>&& shapes, vector<NanoString>&& dtypes, vector<Var*>&& inputs, NumpyFunc&& forward, vector<NumpyFunc>&& sbackward)
    : _inputs(inputs), forward(move(forward))
{
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    USER_CHECKop(shapes.size(),==,dtypes.size()) << "Number of outputs' shapes and dtypes should be the same";
    _outputs.resize(shapes.size());
    USER_CHECKop(_inputs.size(),<=,10) << "numpy_code supports at most ten inputs";
    USER_CHECKop(_outputs.size(),<=,10) << "numpy_code supports at most ten outputs";
    USER_CHECKop(_outputs.size(),>,0);
    for (int i=0; i<shapes.size(); i++) {
        _outputs[i] = create_output(shapes[i], dtypes[i]);
        USER_CHECK(_outputs[i]->num >= 0) << "numpy_code requires known nonnegative output shapes";
    }
    for (int i=0; i<sbackward.size(); i++) {
        backward.push_back(sbackward[i]);
    }
}

NumpyCodeOp::NumpyCodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, NumpyFunc&& forward)
    : _inputs(inputs), forward(move(forward))
{
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    _outputs.push_back(create_output(shape, dtype));
    USER_CHECKop(_inputs.size(),<=,10) << "numpy_code supports at most ten inputs";
    USER_CHECK(_outputs[0]->num >= 0) << "numpy_code requires a known nonnegative output shape";
}

NumpyCodeOp::NumpyCodeOp(vector<NanoVector>&& shapes, vector<NanoString>&& dtypes, vector<Var*>&& inputs, NumpyFunc&& forward)
    : _inputs(inputs), forward(move(forward))
{
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    USER_CHECKop(shapes.size(),==,dtypes.size()) << "Number of outputs' shapes and dtypes should be the same";
    _outputs.resize(shapes.size());
    USER_CHECKop(_inputs.size(),<=,10) << "numpy_code supports at most ten inputs";
    USER_CHECKop(_outputs.size(),<=,10) << "numpy_code supports at most ten outputs";
    USER_CHECKop(_outputs.size(),>,0);
    for (int i=0; i<shapes.size(); i++) {
        _outputs[i] = create_output(shapes[i], dtypes[i]);
        USER_CHECK(_outputs[i]->num >= 0) << "numpy_code requires known nonnegative output shapes";
    }
}

NumpyCodeOp::NumpyCodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, NumpyFunc forward, NumpyResult&& results)
    : _inputs(inputs), forward(forward), _results(move(results))
{
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    _outputs.push_back(create_output(shape, dtype));
    USER_CHECKop(_inputs.size(),<=,10) << "numpy_code supports at most ten inputs";
    USER_CHECK(_outputs[0]->num >= 0) << "numpy_code requires a known nonnegative output shape";
}

VarPtr NumpyCodeOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    USER_CHECK(v_index >= 0 && v_index < (int)backward.size())
        << "numpy_code has no backward callback for input" << v_index;
    NumpyResult result;
    
    int out_index=-1;
    for (int i=0; i<_outputs.size(); i++) {
        if (_outputs[i] == out) {
            out_index = i;
            break;
        }
    }
    ASSERT(out_index!=-1);
    result.ints["out_index"] = out_index;
    vector<DataView> outputs(_outputs.size());
    auto inputs = clone(_inputs);
    // dout and the forward outputs reach the callback as extra *inputs* of the
    // backward op, and run() has to find them again by position rather than by
    // remembering the Var: make_numpy_code passes this vector through
    // adapt_storage_input, which replaces a non-contiguous operand with a
    // contiguous copy before the op is constructed. The gradient seed is
    // always non-contiguous -- it is a stride-0 broadcast of a number -- so the
    // Var recorded here is not an input of the op that actually runs, is not
    // kept alive by it, and has a null mem_ptr by the time run() executes.
    result.ints[dout_input_index_key] = (int)inputs.size();
    inputs.push_back(dout);
    result.arrays["dout"].shape=dout->shape;
    result.arrays["dout"].dtype=dout->dtype();
    result.ints[f_outputs_input_index_key] = (int)inputs.size();
    for (int i=0; i<outputs.size(); i++) {
        outputs[i].shape=_outputs[i]->shape;
        outputs[i].dtype=_outputs[i]->dtype();
        inputs.push_back(_outputs[i]);
    }
    result.varrays["f_outputs"] = move(outputs);

    return make_numpy_code(
        _inputs[v_index]->shape,
        _inputs[v_index]->dtype(),
        move(inputs),
        backward[v_index],
        move(result));
}

void NumpyCodeOp::run() {
    NumpyResult result;
    result.varrays = _results.varrays;
    result.ints = _results.ints;
    result.arrays = _results.arrays;

    // Resolve the gradient operands from this op's *current* inputs (see
    // grad()), and drop the bookkeeping keys so the callback only ever sees
    // "inputs", "outputs", "dout", "f_outputs" and "out_index".
    auto take_index = [&](const char* key) {
        auto iter = result.ints.find(key);
        if (iter == result.ints.end()) return -1;
        int index = iter->second;
        result.ints.erase(iter);
        return index;
    };
    int dout_index = take_index(dout_input_index_key);
    int f_outputs_index = take_index(f_outputs_input_index_key);
    Var* dout_var = nullptr;
    vector<Var*> f_output_vars;
    if (result.arrays.count("dout") > 0) {
        ASSERT(dout_index >= 0 && dout_index < (int)_inputs.size())
            << "numpy_code backward lost track of dout";
        dout_var = _inputs[dout_index];
    }
    if (result.varrays.count("f_outputs") > 0) {
        int n_f_outputs = (int)result.varrays["f_outputs"].size();
        ASSERT(f_outputs_index >= 0 &&
               f_outputs_index + n_f_outputs <= (int)_inputs.size())
            << "numpy_code backward lost track of the forward outputs";
        for (int i=0; i<n_f_outputs; i++)
            f_output_vars.push_back(_inputs[f_outputs_index+i]);
    }

    // A null buffer is not a harmless empty array: to_py_object(DataView) hands
    // the pointer to PyArray_New, which allocates a *host* buffer of its own
    // when it is null, and on CUDA numpy2cupy then wraps that host address as
    // device memory -- the first CuPy kernel to touch it kills the context.
    auto bind = [](DataView& view, Var* var) {
        view.ptr = var->ptr<DataView>();
        view.shape = var->shape;
        view.dtype = var->dtype();
        ASSERT(var->num == 0 || view.ptr != nullptr)
            << "numpy_code operand has no allocated buffer:" << var;
    };

#ifdef IS_ACL
    // On Ascend ACL the numpy callback runs on the HOST, but a Var's mem_ptr is a device
    // address the host cannot dereference -> migrate every operand to host first (mirrors
    // fallback_cpu in backends/acl/src/acl_op_exec.cc). Outputs then live in host (cpu) memory
    // and jittor re-migrates them to device for downstream ACL ops. Gated #ifdef IS_ACL so
    // the CUDA build (host-accessible managed memory) is unchanged. Unblocks all
    // jt.numpy_code consumers on NPU (linalg cholesky/inv/svd/eigh/solve/det, MVN, ...).
    auto _acl_to_host = [](Var* v) {
        if (v && v->mem_ptr && v->allocator && v->allocator->is_cuda())
            migrate_to_cpu(v, cpu_allocator);
    };
    for (auto v : _inputs) _acl_to_host(v);
    for (auto v : _outputs) _acl_to_host(v);
    // dout and the forward outputs are elements of _inputs, so the loop above
    // has already migrated them.
#endif

    if (dout_var)
        bind(result.arrays["dout"], dout_var);
    for (int i=0; i<(int)f_output_vars.size(); i++)
        bind(result.varrays["f_outputs"][i], f_output_vars[i]);
    vector<DataView> inputs(_inputs.size());
    vector<DataView> outputs(_outputs.size());
    for (int i=0; i<inputs.size(); i++)
        bind(inputs[i], _inputs[i]);
    for (int i=0; i<outputs.size(); i++)
        bind(outputs[i], _outputs[i]);
    result.varrays["inputs"] = move(inputs);
    result.varrays["outputs"] = move(outputs);
    forward.callback(&result);
}

} // jittor

#endif // JIT
