# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import os
import jittor as jt
from .test_core import expect_error

header ="""
#pragma once
#include "op.h"

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
#include "var.h"
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

class TestCustomOp(unittest.TestCase):
    def test_compile_custom_ops(self):
        tmp_path = jt.flags.cache_path
        hname = tmp_path+"/custom_op.h"
        ccname = tmp_path+"/custom_op.cc"
        with open(hname, "w") as f:
            f.write(header)
        with open(ccname, "w") as f:
            f.write(src)
        cops = jt.compile_custom_ops([hname, ccname])
        a = cops.custom([3,4,5], 'float')
        na = a.data
        assert a.shape == [3,4,5] and a.dtype == 'float'
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
        na = a.data
        assert a.shape == [3,4,5] and a.dtype == 'float'
        assert (-na.flatten() == range(3*4*5)).all(), na

if __name__ == "__main__":
    unittest.main()
