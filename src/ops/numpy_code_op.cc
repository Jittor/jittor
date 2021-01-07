// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "var.h"
#include "ops/numpy_code_op.h"
#include "ops/op_register.h"

#ifndef JIT

namespace jittor {
    
static auto make_numpy_code = get_op_info("numpy_code")
    .get_constructor<VarPtr, NanoVector, NanoString, vector<Var*>&&, NumpyFunc, NumpyResult&&>();

NumpyCodeOp::NumpyCodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, NumpyFunc&& forward, vector<NumpyFunc>&& sbackward)
    : _inputs(inputs), forward(move(forward))
{
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    _outputs.push_back(create_output(shape, dtype));
    CHECKop(_inputs.size(),<=,10);
    ASSERT(_outputs[0]->num >= 0);
    for (int i=0; i<sbackward.size(); i++) {
        backward.push_back(sbackward[i]);
    }
}

NumpyCodeOp::NumpyCodeOp(vector<NanoVector>&& shapes, vector<NanoString>&& dtypes, vector<Var*>&& inputs, NumpyFunc&& forward, vector<NumpyFunc>&& sbackward)
    : _inputs(inputs), forward(move(forward))
{
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    CHECKop(shapes.size(),==,dtypes.size()) << "Number of outputs' shapes and dtypes should be the same";
    _outputs.resize(shapes.size());
    CHECKop(_inputs.size(),<=,10);
    CHECKop(_outputs.size(),<=,10);
    CHECKop(_outputs.size(),>,0);
    for (int i=0; i<shapes.size(); i++) {
        _outputs[i] = create_output(shapes[i], dtypes[i]);
        ASSERT(_outputs[i]->num >= 0);
    }
    for (int i=0; i<sbackward.size(); i++) {
        backward.push_back(sbackward[i]);
    }
}

NumpyCodeOp::NumpyCodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, NumpyFunc&& forward)
    : _inputs(inputs), forward(move(forward))
{
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    _outputs.push_back(create_output(shape, dtype));
    CHECKop(_inputs.size(),<=,10);
    ASSERT(_outputs[0]->num >= 0);
}

NumpyCodeOp::NumpyCodeOp(vector<NanoVector>&& shapes, vector<NanoString>&& dtypes, vector<Var*>&& inputs, NumpyFunc&& forward)
    : _inputs(inputs), forward(move(forward))
{
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    CHECKop(shapes.size(),==,dtypes.size()) << "Number of outputs' shapes and dtypes should be the same";
    _outputs.resize(shapes.size());
    CHECKop(_inputs.size(),<=,10);
    CHECKop(_outputs.size(),<=,10);
    CHECKop(_outputs.size(),>,0);
    for (int i=0; i<shapes.size(); i++) {
        _outputs[i] = create_output(shapes[i], dtypes[i]);
        ASSERT(_outputs[i]->num >= 0);
    }
}

NumpyCodeOp::NumpyCodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, NumpyFunc forward, NumpyResult&& results)
    : _inputs(inputs), forward(forward), _results(move(results))
{
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    _outputs.push_back(create_output(shape, dtype));
    CHECKop(_inputs.size(),<=,10);
    ASSERT(_outputs[0]->num >= 0);
}

VarPtr NumpyCodeOp::grad(Var* out, Var* dout, Var* v, int v_index) {
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
    result.arrays["dout"].ptr=dout;
    result.arrays["dout"].shape=dout->shape;
    result.arrays["dout"].dtype=dout->dtype();
    vector<DataView> outputs(_outputs.size());
    for (int i=0; i<outputs.size(); i++) {
        outputs[i].ptr=_outputs[i]->ptr<DataView>();
        outputs[i].shape=_outputs[i]->shape;
        outputs[i].dtype=_outputs[i]->dtype();
    }
    result.varrays["f_outputs"] = move(outputs);
    auto inputs = clone(_inputs);
    inputs.push_back(dout);

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
    
    if (result.arrays.count("dout") > 0){
        result.arrays["dout"].ptr=((Var*)result.arrays["dout"].ptr)->ptr<DataView>();
    }
    vector<DataView> inputs(_inputs.size());
    vector<DataView> outputs(_outputs.size());
    for (int i=0; i<inputs.size(); i++) {
        inputs[i].ptr=_inputs[i]->ptr<DataView>();
        inputs[i].shape=_inputs[i]->shape;
        inputs[i].dtype=_inputs[i]->dtype();
    }
    for (int i=0; i<outputs.size(); i++) {
        outputs[i].ptr=_outputs[i]->ptr<DataView>();
        outputs[i].shape=_outputs[i]->shape;
        outputs[i].dtype=_outputs[i]->dtype();
    }
    result.varrays["inputs"] = move(inputs);
    result.varrays["outputs"] = move(outputs);
    forward.callback(&result);
}

} // jittor

#endif // JIT