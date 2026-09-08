# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Identifier renaming when several ops are spliced into one fused kernel.

``OpCompiler::__get_fused_src`` prefixes every op-local name with ``op{i}_`` so
the ops cannot collide.  Names that are *not* op-local -- C++ keywords, standard
types, members of the op struct -- have to be left alone.  They used to be an
implicit whitelist of 24 entries, so an op that wrote ``size_t`` or ``nullptr``
got ``op0_size_t`` and a C++ error naming an identifier nobody wrote.
"""
import os
import unittest

import numpy as np

import jittor as jt

HEADER = """
#pragma once
#include "op.h"
namespace jittor {
%(extra)s
struct %(cls)sOp : Op {
    Var* x, * y;
    %(cls)sOp(Var* x);
    const char* name() const override { return "%(name)s"; }
    DECLARE_jit_run;
};
} // jittor
"""

SRC = """
#include "var.h"
#include "%(name)s_op.h"
namespace jittor {
#ifndef JIT
%(cls)sOp::%(cls)sOp(Var* x) : x(x) {
    y = create_output(x->shape, x->dtype());
    set_type(OpType::element);
}
void %(cls)sOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", x->dtype());
}
#else // JIT
void %(cls)sOp::jit_run() {
    auto* __restrict__ xp = x->ptr<T>();
    auto* __restrict__ yp = y->ptr<T>();
    index_t num = y->num;
    for (index_t i=0; i<num; i++) {
%(body)s
    }
}
#endif // JIT
} // jittor
"""


def build_op(name, body, extra=""):
    """Compile an element-wise custom op whose loop body is ``body``."""
    cls = name.capitalize()
    subst = dict(name=name, cls=cls, body=body, extra=extra)
    path = jt.flags.cache_path
    hname = os.path.join(path, name + "_op.h")
    ccname = os.path.join(path, name + "_op.cc")
    with open(hname, "w") as f:
        f.write(HEADER % subst)
    with open(ccname, "w") as f:
        f.write(SRC % subst)
    return getattr(jt.compile_custom_ops([hname, ccname]), name)


def fused_source(op, a, tag):
    """Run ``op(a) * 2`` as one fused kernel; return (value, generated source)."""
    with jt.profile_scope(compile_options={"_ident_rename": tag},
                          enable_tuner=0) as rep:
        got = (op(a) * 2).data
    src = ""
    for row in rep[1:]:
        for cell in row:
            if isinstance(cell, str) and cell.endswith(".cc") and os.path.exists(cell):
                with open(cell) as f:
                    src += f.read()
    return got, src


# The op body only has to be *fusable*; what it exercises is the identifier
# table.  Every name used here is global, so none of them may be renamed.
RESERVED_BODY = """
        size_t k = (size_t)i;
        unsigned long long m = (unsigned long long)k;
        double d = static_cast<double>(xp[k]) + (double)(m - m);
        yp[k] = (T)((xp == nullptr) ? 0.0 : d);
"""

RESERVED_NAMES = ["size_t", "nullptr", "static_cast", "unsigned", "double"]


class TestFusedIdentifierRename(unittest.TestCase):
    def test_reserved_identifiers_are_not_renamed(self):
        """An element op written in ordinary C++ can be fused.

        Before the identifier table was made explicit this raised
        ``'op1_size_t' was not declared in this scope`` -- an error about a name
        that appears in nobody's source.
        """
        with jt.flag_scope(use_cuda=0):
            op = build_op("kwident", RESERVED_BODY)
            a = jt.random([32, 32])
            a.sync()
            got, src = fused_source(op, a, 1)
            np.testing.assert_allclose(got, a.numpy() * 2, rtol=1e-4, atol=1e-4)
        assert src, "no fused source captured"
        # the op really was fused, not run on its own
        assert "kwident" in src and "binary" in src, src[:400]
        for name in RESERVED_NAMES:
            assert name in src, f"{name} vanished from the fused kernel"
            for oi in range(4):
                bad = f"op{oi}_{name}"
                assert bad not in src, f"{bad} in the fused kernel: {name} was renamed"

    def test_unknown_type_names_the_identifier(self):
        """A type jittor does not know is reported by jittor, not by g++.

        Renaming it produces ``op1_my_index_t op1_k``, which g++ reports as an
        undeclared identifier that appears in no source file.  The compiler
        recognises the shape (two renamed identifiers with only whitespace
        between them) and says which name it did not know.
        """
        with jt.flag_scope(use_cuda=0):
            op = build_op("unkident",
                          "        my_index_t k = (my_index_t)i;\n"
                          "        yp[k] = xp[k];\n",
                          extra="typedef int64 my_index_t;")
            a = jt.random([32, 32])
            a.sync()
            with self.assertRaises(Exception) as cm:
                fused_source(op, a, 2)
        msg = str(cm.exception)
        assert "my_index_t" in msg, msg
        assert "unkident" in msg, msg
        # the give-away that jittor reported this, not g++: the message points
        # at the table to extend.  Without the check g++ reports an undeclared
        # "op1_my_index_t" and says nothing about where that name came from.
        assert "jit_reserved_identifiers" in msg, msg


if __name__ == "__main__":
    unittest.main()
