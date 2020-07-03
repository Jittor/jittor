// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"
#include "numpy_func.h"

namespace jittor {

struct NumpyCodeOp : Op {
    vector<Var*> _inputs;
    vector<Var*> _outputs;
    NumpyFunc forward;
    vector<NumpyFunc> backward;
    NumpyResult _results;

    NumpyCodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, NumpyFunc&& forward, vector<NumpyFunc>&& backward);

    // @attrs(multiple_outputs)
    NumpyCodeOp(vector<NanoVector>&& shapes, vector<NanoString>&& dtypes, vector<Var*>&& inputs, NumpyFunc&& forward, vector<NumpyFunc>&& backward);
	
	// @pybind(None)
	NumpyCodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, NumpyFunc&& forward, NumpyResult&& results);

    const char* name() const override { return "numpy_code"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    
    void run() override;
};

} // jittor