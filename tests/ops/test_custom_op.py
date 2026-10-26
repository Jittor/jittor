# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import os
import jittor as jt
from _helpers.assertions import expect_error

header ="""
#pragma once
#include "core/op.h"

namespace jittor {

struct CustomOp : Op {
    Var* output;
    CustomOp(NanoVector shape, NanoString dtype=ns_float32);
    
    const char* name() const override { return "custom"; }
    DECLARE_jit_run;
};

} // jittor
"""

src = """
#include "core/var.h"
#include "custom_op.h"

namespace jittor {
#ifndef JIT
CustomOp::CustomOp(NanoVector shape, NanoString dtype) {
    output = create_output(shape, dtype);
}

void CustomOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", output->dtype());
}

#else // JIT
#ifdef JIT_cpu
void CustomOp::jit_run() {
    index_t num = output->num;
    auto* __restrict__ x = output->ptr<T>();
    for (index_t i=0; i<num; i++)
        x[i] = (T)i;
}
#else
void CustomOp::jit_run() {
}
#endif // JIT_cpu
#endif // JIT

} // jittor
"""

class TestCustomOpErrorCategory(unittest.TestCase):
    """A custom operator's ``jit_run`` body is caller input.

    ``jt.compile_custom_op`` takes that text as a public argument, and the JIT
    half of the source is not compiled by g++ until after Jittor's fused-kernel
    parser has read it. A loop that parser cannot understand is therefore the
    author's error and must be catchable with a sentence naming the construct,
    not "internal Jittor invariant -- please report this issue".
    """

    def test_unsupported_jit_loop_is_a_catchable_user_error(self):
        header = """
struct JitLoopErrorOp : Op {
    Var* output;
    JitLoopErrorOp(NanoVector shape, NanoString dtype=ns_float32);
    const char* name() const override { return "jit_loop_error"; }
    DECLARE_jit_run;
};
"""
        source = """
#ifndef JIT
JitLoopErrorOp::JitLoopErrorOp(NanoVector shape, NanoString dtype) {
    set_type(OpType::element);
    output = create_output(shape, dtype);
}
void JitLoopErrorOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", output->dtype());
}
#else // JIT
void JitLoopErrorOp::jit_run() {
    index_t num = output->num;
    auto* __restrict__ xx = output->ptr<T>();
    for (index_t i=0; i<num; i+=1) xx[i] = (T)i;
}
#endif // JIT
"""
        with self.assertRaises(Exception) as caught:
            op = jt.compile_custom_op(header, source, "jit_loop_error")
            (op([4], "float32") + 1.0).sync()
        message = str(caught.exception)
        self.assertNotIn("internal Jittor invariant", message)
        self.assertNotIn("report this issue", message)
        self.assertIn("fused-kernel parser", message)
        self.assertIn("not this loop", message)


class TestCustomOp(unittest.TestCase):
    def test_compile_custom_ops(self):
        tmp_path = jt.introspection.policy.startup.cache_path
        hname = tmp_path+"/custom_op.h"
        ccname = tmp_path+"/custom_op.cc"
        with open(hname, "w") as f:
            f.write(header)
        with open(ccname, "w") as f:
            f.write(src)
        cops = jt.compile_custom_ops([hname, ccname])
        a = cops.custom([3,4,5], 'float')
        na = a.numpy()
        assert a.shape == [3,4,5] and str(a.dtype) == "float32"
        assert (na.flatten() == range(3*4*5)).all(), na

    def test_compile_custom_op(self):
        my_op = jt.compile_custom_op("""
        struct MyOp : Op {
            Var* output;
            MyOp(NanoVector shape, NanoString dtype=ns_float32);
            
            const char* name() const override { return "my"; }
            DECLARE_jit_run;
        };
        """, """
        #ifndef JIT
        MyOp::MyOp(NanoVector shape, NanoString dtype) {
            output = create_output(shape, dtype);
        }

        void MyOp::jit_prepare(JK& jk) {
            add_jit_define(jk, "T", output->dtype());
        }

        #else // JIT
        void MyOp::jit_run() {
            index_t num = output->num;
            auto* __restrict__ x = output->ptr<T>();
            for (index_t i=0; i<num; i++)
                x[i] = (T)-i;
        }
        #endif // JIT
        """,
        "my")
        a = my_op([3,4,5], 'float')
        na = a.numpy()
        assert a.shape == [3,4,5] and str(a.dtype) == "float32"
        assert (-na.flatten() == range(3*4*5)).all(), na

if __name__ == "__main__":
    unittest.main()
