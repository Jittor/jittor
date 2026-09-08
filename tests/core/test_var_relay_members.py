# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""How a relayed op's Var* members are addressed from a generated kernel.

A relay splices a call to a real op (mkl_matmul, cudnn_conv, ...) into a fused
kernel, binding that op's Var* members to the fused op's vars first.  The kernel
source is written into the JIT cache and reused, so it must not encode anything
about the op's struct layout: byte offsets move when an op gains a member or the
compiler ABI changes, and the jit key covers neither.
"""
import os
import re
import unittest

import numpy as np

import jittor as jt


def relay_sources(build, tag):
    """Run ``build`` and return the generated sources that contain a relay."""
    with jt.profile_scope(compile_options={"_relay_members": tag}) as rep:
        got = build()
    srcs = []
    for row in rep[1:]:
        for cell in row:
            if isinstance(cell, str) and cell.endswith(".cc") and os.path.exists(cell):
                with open(cell) as f:
                    src = f.read()
                if "relay_groups[" in src:
                    srcs.append(src)
    return got, srcs


class TestVarRelayMembers(unittest.TestCase):
    def _check_relay_source(self, tag, use_cuda):
        a = jt.random([32, 48])
        b = jt.random([48, 64])
        a.sync()
        b.sync()
        with jt.flag_scope(use_cuda=use_cuda):
            got, srcs = relay_sources(lambda: jt.matmul(a, b).data, tag)
        np.testing.assert_allclose(got, a.numpy() @ b.numpy(),
                                   rtol=1e-4, atol=1e-4)
        if not srcs:
            self.skipTest("no relay backend for matmul here")
        for src in srcs:
            # the byte offset used to be written straight into the kernel:
            #   GET_VAR_MEMBER(rop_0_0, 120) = vars[2].var;
            assert not re.search(r"GET_VAR_MEMBER\s*\(", src), (
                "relay kernel encodes a struct byte offset:\n" +
                "\n".join(l for l in src.splitlines() if "GET_VAR_MEMBER" in l))
            names = re.findall(r'set_var_member\("(\w+)"', src)
            assert names, "relay kernel binds no member:\n" + src

    def test_relay_kernel_names_members_instead_of_offsets_cpu(self):
        self._check_relay_source(1, 0)

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def test_relay_kernel_names_members_instead_of_offsets_cuda(self):
        self._check_relay_source(2, 1)


class TestParseVarMembers(unittest.TestCase):
    """The header scan that feeds the op registry.

    A member it fails to read is left out of the registry, and relaying then
    never binds it -- the relayed op runs with whatever the member pointed at
    last.  Nothing downstream notices, so the scan itself has to complain.
    """

    def setUp(self):
        # imported here so that a build without the helper still runs the relay
        # test above instead of failing collection
        from jittor.compiler import parse_var_members
        self.parse = parse_var_members

    def test_canonical_spellings(self):
        self.assertEqual(self.parse("struct A : Op {\n    Var* x;\n};"), ["x"])
        self.assertEqual(self.parse("struct A : Op {\n    Var *x;\n};"), ["x"])
        self.assertEqual(
            self.parse("struct A : Op {\n    Var* x, * y, * z;\n};"),
            ["x", "y", "z"])
        self.assertEqual(
            self.parse("struct A : Op {\n    Var* x;\n    Var* y;\n};"),
            ["x", "y"])

    def test_non_members_are_not_picked_up(self):
        # function declarations, VarPtr members, and unrelated types
        src = ("struct A : Op {\n"
               "    VarPtr owned;\n"
               "    void f(Var* v);\n"
               "    Var* make(Var* a, Var* b);\n"
               "    Var* x;\n"
               "};")
        self.assertEqual(self.parse(src), ["x"])

    def test_unreadable_declaration_fails_the_build(self):
        for src in ["struct A : Op {\n    jittor::Var* x;\n};",
                    "struct A : Op {\n    const Var* x;\n};"]:
            with self.assertRaises(AssertionError) as cm:
                self.parse(src, "a_op.h")
            assert "a_op.h" in str(cm.exception), cm.exception

    def test_every_op_header_is_readable(self):
        import glob
        from jittor import compiler
        headers = (glob.glob(os.path.join(compiler.jittor_path, "src/ops/*.h")) +
                   glob.glob(os.path.join(compiler.jittor_path, "extern/**/*_op.h"),
                             recursive=True))
        assert len(headers) > 20, headers
        for h in headers:
            with open(h, encoding="utf8") as f:
                self.parse(f.read(), h)


if __name__ == "__main__":
    unittest.main()
