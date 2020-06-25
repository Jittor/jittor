// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
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
    .get_constructor<VarPtr, NanoVector, NanoString, vector<Var*>&&, NumpyFunc&&, NumpyResults&&>();
    
static inline void check_vary_shape(NanoVector v) {
    ASSERT(v.size()) << "Vary shape should not be zero dimension";
    for (int i=0; i<v.size(); i++)
        ASSERT((i == 0) ^ (v[i] >= 0))
            << "Vary shape should only occur in the first dimension:" << v;
}

NumpyCodeOp::NumpyCodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs={}, NumpyFunc&& forward, vector<NumpyFunc>&& backward)
    : _inputs(inputs), forward(forward),backward(backward)
{
    _outputs.push_back(create_output(shape, dtype));
    CHECKop(_inputs.size(),<=,10);

    if (_outputs[0]->num < 0) {
        flags.set(NodeFlags::_vary_shape);
        check_vary_shape(_outputs[0]->shape);
    }
}

NumpyCodeOp::NumpyCodeOp(vector<NanoVector>&& shapes, vector<NanoString>&& dtypes, vector<Var*>&& inputs={}, NumpyFunc&& forward, vector<NumpyFunc>&& backward)
    : _inputs(inputs), forward(forward),backward(backward)
{
    CHECKop(shapes.size(),==,dtypes.size()) << "Number of outputs' shapes and dtypes should be the same";
    _outputs.resize(shapes.size());
    CHECKop(_inputs.size(),<=,10);
    CHECKop(_outputs.size(),<=,10);
    CHECKop(_outputs.size(),>,0);
    for (int i=0; i<shapes.size(); i++) {
        _outputs[i] = create_output(shapes[i], dtypes[i]);
        if (_outputs[i]->num < 0) {
            flags.set(NodeFlags::_vary_shape);
            check_vary_shape(_outputs[i]->shape);
        }
    }
}

NumpyCodeOp::NumpyCodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs={}, NumpyFunc&& forward, NumpyResults&& results)
    : _inputs(inputs), forward(forward), _results(results)
{
    _outputs.push_back(create_output(shape, dtype));
    CHECKop(_inputs.size(),<=,10);

    if (_outputs[0]->num < 0) {
        flags.set(NodeFlags::_vary_shape);
        check_vary_shape(_outputs[0]->shape);
    }
}

void NumpyCodeOp::grad(Var* out, Var* dout, Var* v, int v_index) {
	NumpyResult result;
	// set results
	// set dout index
	result.ints["dout_index"] = _outputs.find(out);
	result.arrays["dout"] = ArrayArgs{
            dout->ptr,
            dout->shape,
            dout->dtype(),
        };
	
	// code op:
	/*
    return make_code(
        _inputs[v_index]->shape,
        _inputs[v_index]->dtype(),
        move(inputs),
        move(cpu_src), {}, alias+cpu_header,
        move(cuda_src), {}, alias+cuda_header
    );
	*/
	return make_numpy_code(
        _inputs[v_index]->shape,
        _inputs[v_index]->dtype(),
		inputs, 
		backward[v_index],
		result,
	)
}

void NumpyCodeOp::run() {
    NumpyResult result=_results;
    vector<ArrayArgs> inputs(_inputs.size());
    vector<ArrayArgs> outputs(_outputs.size());
    /*
    const void* ptr;
    NanoVector shape;
    NanoString dtype;
    */
    for (int i=0; i<inputs.size(); i++)
        inputs[i] = ArrayArgs{
            _inputs[i]->ptr,
            _inputs[i]->shape,
            _inputs[i]->dtype(),
        };
    for (int i=0; i<outputs.size(); i++)
        outputs[i] = ArrayArgs{
            _outputs[i]->ptr,
            _outputs[i]->shape,
            _outputs[i]->dtype(),
        };
    result.varrays["inputs"] = move(inputs);
    result.varrays["outputs"] = move(outputs);
	forward.callback(&results);
}

} // jittor

#endif // JIT