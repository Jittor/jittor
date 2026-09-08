# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os
import unittest

import numpy as np

import jittor as jt


HEADER = r"""
#pragma once
#include "op.h"
namespace jittor {
struct JitsourceOp : Op {
    Var* x, * y;
    JitsourceOp(Var* x);
    const char* name() const override { return "jitsource"; }
    DECLARE_jit_run;
};
} // jittor
"""

SOURCE = r"""
#include "var.h"
#include "jitsource_op.h"
namespace jittor {
#ifndef JIT
JitsourceOp::JitsourceOp(Var* x) : x(x) {
    y = create_output(x->shape, x->dtype());
    set_type(OpType::element);
}
void JitsourceOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", x->dtype());
}
#else
void JitsourceOp::jit_run() {
    auto* xp = x->ptr<T>();
    auto* yp = y->ptr<T>();
    index_t num = y->num;
    const char* format = "value=%d {literal}";
#ifdef JIT_cuda
    _Pragma("unroll 1")
#else
    _Pragma("omp simd")
#endif
    for (index_t i=0; i<num; i++) {
        yp[i] = xp[i] + (T)(format[0] == 'v');
    }
}
#endif
} // jittor
"""


def _build_op():
    path = jt.flags.cache_path
    header = os.path.join(path, "jitsource_op.h")
    source = os.path.join(path, "jitsource_op.cc")
    with open(header, "w", encoding="utf-8") as f:
        f.write(HEADER)
    with open(source, "w", encoding="utf-8") as f:
        f.write(SOURCE)
    return jt.compile_custom_ops([header, source]).jitsource


class TestJitSourceContract(unittest.TestCase):
    def test_pragma_compiles_and_format_literal_survives_fusion(self):
        op = _build_op()
        x = jt.array(np.arange(16, dtype=np.float32))
        x.sync()
        with jt.profile_scope(enable_tuner=0) as report:
            got = (op(x) * 2).numpy()
        np.testing.assert_array_equal(got, (np.arange(16) + 1) * 2)

        generated = ""
        for row in report[1:]:
            for cell in row:
                if isinstance(cell, str) and cell.endswith(".cc") and os.path.exists(cell):
                    with open(cell, encoding="utf-8") as f:
                        generated += f.read()
        self.assertTrue(generated, "no fused source captured")
        self.assertIn('"value=%d {literal}"', generated)
        self.assertIn('#line 14 "', generated)
        self.assertIn("jitsource_op.cc", generated)


if __name__ == "__main__":
    unittest.main()
