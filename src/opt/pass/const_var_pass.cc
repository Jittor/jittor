// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "opt/expr.h"
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/const_var_pass.h"
#include "ops/array_op.h"
#include "jit_key.h"

namespace jittor {

using namespace expr;

void ConstVarPass::run() {
    int changed = 0;
    for (int i=0; i<op->ops.size(); i++) {
        auto opi = op->ops[i];
        if (opi->name() != string("array"))
            continue;
        string s;
        auto* v = opi->output(0);
        if (v->num != 1)
            continue;
        auto array_op = (ArrayOp*)opi;
        jk.clear();
        array_op->jit_prepare(jk);
        if (jk.to_string().find("[o:") == string::npos)
            continue;
        if (v->dtype() == ns_int32) {
            s = S(array_op->ptr<int32>()[0]);
        } else
        if (v->dtype() == ns_float32) {
            s = S(array_op->ptr<float32>()[0]);
        } else
            continue;
        auto def = ir->find_define("op"+S(i)+"_outputd");
        ASSERT(def);
        def->attrs["dtype"] = v->dtype().to_cstring();
        def->attrs["rvalue"] = s;
        changed ++;
        LOGvvvv << def->to_string();
    }
    if (changed) {
        ir->remove_all_unused();
    }
}

} // jittor