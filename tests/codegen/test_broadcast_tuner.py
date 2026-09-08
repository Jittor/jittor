# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import sys
import os
import pathlib
import shutil
import subprocess
import jittor as jt
import unittest
import time
import numpy as np
from _helpers.logs import find_log_with_re
from _helpers.tuner_parser import simple_parser

class TestBroadcastTuner(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        return

    def check(self, h, w, cs, rs, pa, rtp, dim):
        a = jt.random([h,w])
        a.data
        

        with jt.log_capture_scope(
            log_v=0, log_vprefix="tuner_manager=100",
            # this value is used for force compile
            compile_options={"test_broadcast_tuner":1}
        ) as logs:
            amean=jt.mean(a, dims=[dim], keepdims=1)
            a2mean=jt.mean(a*a, dims=[dim], keepdims=1)
            norm_aa=(a-amean.broadcast_var(a))/(jt.sqrt(a2mean-amean*amean).broadcast_var(a))
            norm_aa.data
        logs = find_log_with_re(logs, 
            "Run tuner broadcast: confidence\\((20)\\) candidates\\((.*)\\)$")
        assert len(logs) == 1, logs
        assert logs[0][0] == "20", "confidence of reorder should be 20"
        candidates = simple_parser(logs[0][1])
        assert candidates == {"order0": [0,], "order1": [1,], "order2": [0,], "split1": [2048,],}, candidates
        
    def test_broadcast_tuner(self):
        self.check(8192,8192, 0, 0, 0, 5, 0)


class TestOutputStoresStayOrdinary(unittest.TestCase):
    """No CPU kernel may store its output non-temporally, one element at a time.

    A non-temporal output store is worth real bandwidth, but only when the
    whole loop streams. Measured on this hardware, writing 256 MiB that is
    never read back:

        ordinary store                        15.0 GB/s   (g++ and clang)
        per-element non-temporal store         9.1 GB/s   g++ (_mm_stream_si32)
        per-element non-temporal store        14.6 GB/s   clang (__builtin_
                                                          nontemporal_store)
        _mm256_stream_ps over the whole loop  25.3 GB/s

    The middle two are what a pass that rewrites one store *statement* can
    emit, and neither compiler will widen such a store back into a vector one
    -- so the rewrite buys nothing on clang and costs 40% on g++. That is why
    UseMovntPass and the `use_movnt` tuner candidate were deleted rather than
    ported to g++.

    This test disassembles the kernel that actually got built, because a
    non-temporal store computes exactly the same numbers as an ordinary one:
    checking the result cannot tell the two apart.
    """

    def _cpu_kernels(self, tag):
        """Every kernel the broadcast subtract runs, compiled now or cached.

        Keyed on "Opening jit lib" rather than on "Generate", because a kernel
        already in the on-disk cache is never regenerated and a test watching
        for compilation would quietly check nothing from its second run on.
        Opening is only logged the first time a key is used in a process, so
        each test passes its own ``tag`` to keep its key to itself.
        """
        values = np.arange(4096, dtype=np.float32)
        lhs = jt.array(values)
        rhs = jt.ones((32, 4096), dtype="float32")
        with jt.log_capture_scope(
                log_v=0,
                log_vprefix="jit_compiler.cc=1000",
                compile_options={"test_output_store_form": tag}) as logs:
            result = (rhs-lhs.broadcast_var(rhs)).numpy()
        # Correct numbers are a precondition, not the evidence.
        np.testing.assert_array_equal(result[7], 1-values)

        opened = [entry["msg"].split("Opening jit lib:", 1)[1].strip()
                  for entry in logs if "Opening jit lib:" in entry["msg"]]
        self.assertTrue(opened, "no kernel was loaded, so nothing was checked")
        return opened

    def test_the_generated_source_assigns_the_output_normally(self):
        # Kernel file names are truncated to stay within the filesystem limit,
        # so the fused kernel is found by what its source defines, not by name.
        sources = []
        for library in self._cpu_kernels(1):
            source = library.replace("_op.so", "_op.cc")
            if not os.path.exists(source):
                continue
            text = open(source).read()
            if "#define op2_OP subtract" in text:
                sources.append((source, text))
        self.assertTrue(sources, "no fused subtract kernel, so nothing was checked")
        for path, source in sources:
            with self.subTest(kernel=os.path.basename(path)):
                self.assertIn("op2_zp[op2_i] =", source)
                self.assertNotIn("__builtin_nontemporal_store", source)
                self.assertNotIn("_mm_stream", source)
                # Markers the deleted asm_tuner chain keyed its rewriting on.
                self.assertNotIn("//@begin", source)
                self.assertNotIn("//@end", source)

    def test_the_built_shared_object_holds_no_non_temporal_store(self):
        if shutil.which("objdump") is None:
            self.skipTest("objdump is needed to read back what was compiled")
        checked = 0
        for library in self._cpu_kernels(2):
            if not os.path.exists(library):
                continue
            dump = subprocess.run(["objdump", "-d", library],
                                  stdout=subprocess.PIPE, text=True).stdout
            found = [line.strip() for line in dump.splitlines() if "movnt" in line]
            self.assertEqual(
                found, [],
                "%s stores its output non-temporally. Per-element movnt was "
                "measured at 9.1 GB/s (g++) and 14.6 GB/s (clang) against "
                "15.0 GB/s for an ordinary store; see this class's docstring "
                "before putting it back." % os.path.basename(library))
            checked += 1
        self.assertTrue(checked, "no .so was disassembled, so nothing was checked")


class TestUseMovntIsGone(unittest.TestCase):
    def test_the_pass_and_its_sources_are_removed(self):
        passes = pathlib.Path(jt.flags.jittor_path) / "src" / "opt" / "pass"
        self.assertFalse((passes / "use_movnt_pass.h").exists())
        self.assertFalse((passes / "use_movnt_pass.cc").exists())
        manager = (passes.parent / "pass_manager.cc").read_text()
        self.assertNotIn("UseMovntPass", manager)
        self.assertNotIn("use_movnt_pass.h", manager)

    def test_the_broadcast_tuner_no_longer_offers_the_candidate(self):
        tuner = (pathlib.Path(jt.flags.jittor_path) / "src" / "opt" / "tuner"
                 / "broadcast_tuner.cc").read_text()
        self.assertNotIn('add_candidate("use_movnt"', tuner)


if __name__ == "__main__":
    unittest.main()
