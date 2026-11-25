# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Regression tests for CPU JIT PCH, relocatable $ORIGIN rpath,
JITTOR_SHARED_CACHE, JT_FAST_DEV, JT_CCACHE, and runtime norm eps."""

import glob
import json
import os
import platform
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

import jittor as jt
import jittor.compiler as compiler
import jittor.nn as nn
import jittor_utils as jit_utils

from _helpers import capability as _test_capability
from _helpers.child_process import run_child_script


def _has_cuda():
    return _test_capability.check_accelerator("cuda", backend=jt).enabled


def _jit_so_set(cache_path):
    return set(glob.glob(os.path.join(cache_path, "jit", "*.so")))


def _readelf_rpaths(so_path):
    readelf = shutil.which("readelf")
    if not readelf:
        return None
    out = subprocess.check_output([readelf, "-d", so_path], text=True)
    lines = [line for line in out.splitlines() if "RPATH" in line or "RUNPATH" in line]
    return "\n".join(lines)


class TestCpuJitPchAndRelocatableRpath(unittest.TestCase):
    @unittest.skipUnless(platform.system() == "Linux", "CPU JIT PCH and $ORIGIN rpath are Linux-specific")
    def test_cpu_pch_matches_non_pch_and_handles_extra_flags(self):
        rs = np.random.RandomState(1234)
        a_np = rs.uniform(0.5, 2.0, (17, 19)).astype(np.float32)
        b_np = rs.uniform(0.5, 2.0, (17, 19)).astype(np.float32)
        c_np = rs.uniform(0.5, 2.0, (1, 19)).astype(np.float32)

        def run_ops():
            a, b, c = jt.array(a_np), jt.array(b_np), jt.array(c_np)
            y1 = ((a * b + c).sin() - (a + c).cos()).sum(dim=1)
            y2 = jt.where(a > b, (a * c).tanh(), (b + c).sigmoid())
            y3 = jt.maximum(a - b * 0.5, c).sqrt()
            jt.sync([y1, y2, y3])
            return y1.numpy(), y2.numpy(), y3.numpy()

        # Run with PCH disabled first (using a unique code op too)
        with jt.flag_scope(use_cuda=0, use_jit_pch=0):
            ref = run_ops()
            code_ref = jt.code(
                [4], "float32", [],
                cpu_src="""
                    for (int i = 0; i < 4; i++)
                        @out0(i) = float(i * JIT_PCH_EXTRA_FACTOR + 1);
                """,
            )
            code_ref.compile_options = {"FLAGS: -DJIT_PCH_EXTRA_FACTOR=5 ": 1}
            code_ref.sync()
            code_ref_np = code_ref.numpy().copy()

        # Run with PCH enabled: verify PCH builds, extra_flags works cleanly,
        # and outputs are numerically identical.
        with jt.flag_scope(use_cuda=0, use_jit_pch=1):
            # Trigger a fresh CPU kernel so PCH is definitely built & exercised.
            d = jt.array(rs.uniform(0.5, 2.0, (13, 23)).astype(np.float32))
            e = jt.array(rs.uniform(0.5, 2.0, (13, 23)).astype(np.float32))
            so_before = _jit_so_set(compiler.cache_path)
            z = ((d * 2.5 - e).sin() + (d + e * 0.5).tanh()).sum(dim=0)
            z.sync()
            so_after = _jit_so_set(compiler.cache_path)

            got = run_ops()
            for g, r in zip(got, ref):
                np.testing.assert_allclose(g, r, rtol=1e-6, atol=1e-6)

            code_pch = jt.code(
                [4], "float32", [],
                cpu_src="""
                    for (int i = 0; i < 4; i++)
                        @out0(i) = float(i * JIT_PCH_EXTRA_FACTOR + 2);
                """,
            )
            code_pch.compile_options = {"FLAGS: -DJIT_PCH_EXTRA_FACTOR=7 ": 1}
            code_pch.sync()
            np.testing.assert_allclose(
                code_pch.numpy(), np.array([2.0, 9.0, 16.0, 23.0], dtype=np.float32))
            np.testing.assert_allclose(
                code_ref_np, np.array([1.0, 6.0, 11.0, 16.0], dtype=np.float32))

        pch_gch = os.path.join(compiler.cache_path, "gen", "jit_cpu_pch.h.gch")
        self.assertTrue(os.path.isfile(pch_gch), f"missing CPU JIT PCH: {pch_gch}")
        self.assertTrue(os.path.isfile(pch_gch + ".key"), f"missing CPU JIT PCH key: {pch_gch}.key")

        # Verify newly compiled .so uses relocatable $ORIGIN/.. rpath
        sample_sos = sorted(so_after - so_before) or sorted(so_after)
        self.assertTrue(sample_sos)
        rpaths = _readelf_rpaths(sample_sos[0])
        if rpaths is not None:
            self.assertIn("$ORIGIN/..", rpaths)
            self.assertNotIn(compiler.cache_path, rpaths)


class TestSharedJitCache(unittest.TestCase):
    @unittest.skipUnless(platform.system() == "Linux", "JITTOR_SHARED_CACHE is Linux-specific")
    def test_shared_cache_populates_serves_warm_hits_and_isolates_changes(self):
        script_populate = r"""
import glob, json, os
import numpy as np
import jittor as jt

tag = os.environ["JT_TEST_TAG"]
val = int(os.environ["JT_TEST_VAL"])
use_cuda = int(os.environ["JT_TEST_CUDA"])

with jt.flag_scope(use_cuda=use_cuda, use_parallel_op_compiler=0):
    x = jt.array(np.arange(16, dtype=np.float32).reshape(4, 4))
    if use_cuda:
        y = jt.code(
            x.shape, x.dtype, [x],
            cuda_src=f'''
            __global__ static void k_{tag}(in0_type* x, out0_type* y, int n) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) y[i] = x[i] * {val}.0f + 3.0f;
            }}
            k_{tag}<<<1, 32>>>(in0_p, out0_p, (int)in0->num);
            ''',
        )
    else:
        y = jt.code(
            x.shape, x.dtype, [x],
            cpu_src=f'''
            int k_{tag} = {val};
            for (int i = 0; i < in0->num; i++)
                out0_p[i] = in0_p[i] * float(k_{tag}) + 3.0f;
            ''',
        )
    y.sync()
    out = y.numpy().tolist()

print("SHARED_RESULT " + json.dumps({
    "cache_path": jt.compiler.cache_path,
    "out": out,
}))
"""
        with tempfile.TemporaryDirectory(prefix="jt-shared-cache-") as tmp_dir:
            shared_dir = os.path.join(tmp_dir, "shared")
            os.makedirs(shared_dir, exist_ok=True)
            scripts_dir = os.path.join(tmp_dir, "scripts")
            os.makedirs(scripts_dir, exist_ok=True)

            backends = [0, 1] if _has_cuda() else [0]
            for use_cuda in backends:
                tag = f"shared_probe_{'cuda' if use_cuda else 'cpu'}_{os.getpid()}"
                # 1. Populate shared cache
                r1 = run_child_script(
                    script_populate,
                    env={
                        "JITTOR_SHARED_CACHE": shared_dir,
                        "JT_TEST_TAG": tag,
                        "JT_TEST_VAL": "2",
                        "JT_TEST_CUDA": str(use_cuda),
                        "JT_LOG_SILENT": "0",
                        "JT_LOG_V": "10",
                    },
                    directory=scripts_dir,
                    name=f"pop_{use_cuda}",
                    text=True,
                    merge_stderr=True,
                )
                self.assertEqual(r1.returncode, 0, r1.stdout[-4000:])
                line1 = next(l for l in r1.stdout.splitlines() if l.startswith("SHARED_RESULT "))
                res1 = json.loads(line1[len("SHARED_RESULT "):])
                local_cache = res1["cache_path"]

                shared_sos = list(Path(shared_dir).rglob("*.so"))
                self.assertTrue(shared_sos, f"shared cache was not populated for cuda={use_cuda}")
                # Check ELF rpath of published shared .so is relocatable
                rpaths = _readelf_rpaths(str(shared_sos[-1]))
                if rpaths is not None:
                    self.assertIn("$ORIGIN/..", rpaths)
                    self.assertNotIn(local_cache, rpaths)

                # 2. Remove local jit artifacts for this op so the next child
                # experiences a local cache miss and must fetch from shared cache.
                jit_dir = Path(local_cache) / "jit"
                removed = 0
                for cc_file in jit_dir.glob("*.cc"):
                    try:
                        if tag in cc_file.read_text(encoding="utf-8", errors="ignore"):
                            stem = cc_file.stem
                            for companion in jit_dir.glob(stem + "*"):
                                companion.unlink(missing_ok=True)
                                removed += 1
                    except OSError:
                        pass
                self.assertGreater(removed, 0, f"did not locate local artifacts for {tag}")

                # 3. Re-run: must hit JITTOR_SHARED_CACHE (zero compile) & match output
                r2 = run_child_script(
                    script_populate,
                    env={
                        "JITTOR_SHARED_CACHE": shared_dir,
                        "JT_TEST_TAG": tag,
                        "JT_TEST_VAL": "2",
                        "JT_TEST_CUDA": str(use_cuda),
                        "JT_LOG_SILENT": "0",
                        "JT_LOG_V": "10",
                    },
                    directory=scripts_dir,
                    name=f"hit_{use_cuda}",
                    text=True,
                    merge_stderr=True,
                )
                self.assertEqual(r2.returncode, 0, r2.stdout[-4000:])
                self.assertIn("Shared JIT cache hit:", r2.stdout)
                line2 = next(l for l in r2.stdout.splitlines() if l.startswith("SHARED_RESULT "))
                res2 = json.loads(line2[len("SHARED_RESULT "):])
                self.assertEqual(res2["out"], res1["out"])

                # 3b. Run again WITHOUT JITTOR_SHARED_CACHE: must now hit local
                # unlocked cache cleanly using the .key written on shared hit.
                r2b = run_child_script(
                    script_populate,
                    env={
                        "JITTOR_SHARED_CACHE": "",
                        "JT_TEST_TAG": tag,
                        "JT_TEST_VAL": "2",
                        "JT_TEST_CUDA": str(use_cuda),
                        "JT_LOG_SILENT": "0",
                        "JT_LOG_V": "10",
                    },
                    directory=scripts_dir,
                    name=f"local_{use_cuda}",
                    text=True,
                    merge_stderr=True,
                )
                self.assertEqual(r2b.returncode, 0, r2b.stdout[-4000:])
                self.assertIn("Cached op, no build lock taken:", r2b.stdout)
                line2b = next(l for l in r2b.stdout.splitlines() if l.startswith("SHARED_RESULT "))
                res2b = json.loads(line2b[len("SHARED_RESULT "):])
                self.assertEqual(res2b["out"], res1["out"])

                # 4. Change source value (2 -> 5): MUST miss shared cache and produce new output
                r3 = run_child_script(
                    script_populate,
                    env={
                        "JITTOR_SHARED_CACHE": shared_dir,
                        "JT_TEST_TAG": tag,
                        "JT_TEST_VAL": "5",
                        "JT_TEST_CUDA": str(use_cuda),
                        "JT_LOG_SILENT": "0",
                        "JT_LOG_V": "10",
                    },
                    directory=scripts_dir,
                    name=f"miss_{use_cuda}",
                    text=True,
                    merge_stderr=True,
                )
                self.assertEqual(r3.returncode, 0, r3.stdout[-4000:])
                self.assertNotIn("Shared JIT cache hit:", r3.stdout)
                line3 = next(l for l in r3.stdout.splitlines() if l.startswith("SHARED_RESULT "))
                res3 = json.loads(line3[len("SHARED_RESULT "):])
                expected_new = (np.arange(16, dtype=np.float32).reshape(4, 4) * 5.0 + 3.0).tolist()
                self.assertEqual(res3["out"], expected_new)

                # Clean up our synthetic probe files from the session jit dir
                for cc_file in jit_dir.glob("*.cc"):
                    try:
                        if tag in cc_file.read_text(encoding="utf-8", errors="ignore"):
                            for companion in jit_dir.glob(cc_file.stem + "*"):
                                companion.unlink(missing_ok=True)
                    except OSError:
                        pass

    @unittest.skipUnless(platform.system() == "Linux", "JITTOR_SHARED_CACHE is Linux-specific")
    def test_shared_cache_isolates_extra_flags_changes(self):
        script = r"""
import json, os
import numpy as np
import jittor as jt

tag = os.environ["JT_TEST_TAG"]
factor = int(os.environ["JT_TEST_FACTOR"])
with jt.flag_scope(use_cuda=0, use_parallel_op_compiler=0):
    x = jt.array(np.arange(8, dtype=np.float32))
    y = jt.code(
        x.shape, x.dtype, [x],
        cpu_src=f'''
        int k_{tag} = JT_CAS_FLAG_FACTOR;
        for (int i = 0; i < in0->num; i++)
            out0_p[i] = in0_p[i] * float(k_{tag});
        ''',
    )
    y.compile_options = {f"FLAGS: -DJT_CAS_FLAG_FACTOR={factor} ": 1}
    y.sync()
    print("FLAG_RESULT " + json.dumps({"cache_path": jt.compiler.cache_path, "out": y.numpy().tolist()}))
"""
        with tempfile.TemporaryDirectory(prefix="jt-shared-flags-") as tmp_dir:
            shared_dir = os.path.join(tmp_dir, "shared")
            scripts_dir = os.path.join(tmp_dir, "scripts")
            os.makedirs(shared_dir, exist_ok=True)
            os.makedirs(scripts_dir, exist_ok=True)
            tag = f"flag_iso_{os.getpid()}"

            r1 = run_child_script(
                script,
                env={"JITTOR_SHARED_CACHE": shared_dir, "JT_TEST_TAG": tag, "JT_TEST_FACTOR": "3", "JT_LOG_SILENT": "1"},
                directory=scripts_dir, name="f3", text=True, merge_stderr=True,
            )
            self.assertEqual(r1.returncode, 0, r1.stdout[-4000:])
            res1 = json.loads(next(l for l in r1.stdout.splitlines() if l.startswith("FLAG_RESULT "))[len("FLAG_RESULT "):])
            self.assertEqual(res1["out"], (np.arange(8, dtype=np.float32) * 3.0).tolist())

            # Same source, different compiler flag (-D): MUST NOT reuse the factor=3 binary.
            r2 = run_child_script(
                script,
                env={"JITTOR_SHARED_CACHE": shared_dir, "JT_TEST_TAG": tag, "JT_TEST_FACTOR": "9", "JT_LOG_SILENT": "1"},
                directory=scripts_dir, name="f9", text=True, merge_stderr=True,
            )
            self.assertEqual(r2.returncode, 0, r2.stdout[-4000:])
            res2 = json.loads(next(l for l in r2.stdout.splitlines() if l.startswith("FLAG_RESULT "))[len("FLAG_RESULT "):])
            self.assertEqual(res2["out"], (np.arange(8, dtype=np.float32) * 9.0).tolist())

            jit_dir = Path(res1["cache_path"]) / "jit"
            for cc_file in jit_dir.glob("*.cc"):
                try:
                    if tag in cc_file.read_text(encoding="utf-8", errors="ignore"):
                        for companion in jit_dir.glob(cc_file.stem + "*"):
                            companion.unlink(missing_ok=True)
                except OSError:
                    pass

    @unittest.skipUnless(platform.system() == "Linux", "JITTOR_SHARED_CACHE is Linux-specific")
    def test_shared_cache_skips_external_mutable_header(self):
        script = r"""
import json, os
import numpy as np
import jittor as jt

tag = os.environ["JT_TEST_TAG"]
hdr = os.environ["JT_TEST_EXT_HDR"]
with jt.flag_scope(use_cuda=0, use_parallel_op_compiler=0):
    x = jt.array(np.arange(8, dtype=np.float32))
    y = jt.code(
        x.shape, x.dtype, [x],
        cpu_header=f'#include "{hdr}"\n',
        cpu_src=f'''
        int k_{tag} = JT_EXT_MUTABLE_VAL;
        for (int i = 0; i < in0->num; i++)
            out0_p[i] = in0_p[i] + float(k_{tag});
        ''',
    )
    y.sync()
    print("EXT_RESULT " + json.dumps({"cache_path": jt.compiler.cache_path, "out": y.numpy().tolist()}))
"""
        with tempfile.TemporaryDirectory(prefix="jt-shared-ext-hdr-") as tmp_dir:
            shared_dir = os.path.join(tmp_dir, "shared")
            scripts_dir = os.path.join(tmp_dir, "scripts")
            os.makedirs(shared_dir, exist_ok=True)
            os.makedirs(scripts_dir, exist_ok=True)
            hdr_path = os.path.join(tmp_dir, "ext_mutable_val.h")
            Path(hdr_path).write_text("#define JT_EXT_MUTABLE_VAL 11\n", encoding="utf-8")
            tag = f"ext_hdr_{os.getpid()}"
            jit_dir = None
            try:
                r1 = run_child_script(
                    script,
                    env={
                        "JITTOR_SHARED_CACHE": shared_dir,
                        "JT_TEST_TAG": tag,
                        "JT_TEST_EXT_HDR": hdr_path,
                        "JT_LOG_SILENT": "1",
                    },
                    directory=scripts_dir, name="ext1", text=True, merge_stderr=True,
                )
                self.assertEqual(r1.returncode, 0, r1.stdout[-4000:])
                res1 = json.loads(next(l for l in r1.stdout.splitlines() if l.startswith("EXT_RESULT "))[len("EXT_RESULT "):])
                self.assertEqual(res1["out"], (np.arange(8, dtype=np.float32) + 11.0).tolist())

                # Remove local artifacts for this op and mutate the external header.
                jit_dir = Path(res1["cache_path"]) / "jit"
                for cc_file in jit_dir.glob("*.cc"):
                    try:
                        if tag in cc_file.read_text(encoding="utf-8", errors="ignore"):
                            for companion in jit_dir.glob(cc_file.stem + "*"):
                                companion.unlink(missing_ok=True)
                    except OSError:
                        pass
                Path(hdr_path).write_text("#define JT_EXT_MUTABLE_VAL 99\n", encoding="utf-8")

                r2 = run_child_script(
                    script,
                    env={
                        "JITTOR_SHARED_CACHE": shared_dir,
                        "JT_TEST_TAG": tag,
                        "JT_TEST_EXT_HDR": hdr_path,
                        "JT_LOG_SILENT": "0",
                        "JT_LOG_V": "10",
                    },
                    directory=scripts_dir, name="ext2", text=True, merge_stderr=True,
                )
                self.assertEqual(r2.returncode, 0, r2.stdout[-4000:])
                self.assertNotIn("Shared JIT cache hit:", r2.stdout)
                res2 = json.loads(next(l for l in r2.stdout.splitlines() if l.startswith("EXT_RESULT "))[len("EXT_RESULT "):])
                self.assertEqual(res2["out"], (np.arange(8, dtype=np.float32) + 99.0).tolist())
            finally:
                if jit_dir is not None:
                    for cc_file in jit_dir.glob("*.cc"):
                        try:
                            if tag in cc_file.read_text(encoding="utf-8", errors="ignore"):
                                for companion in jit_dir.glob(cc_file.stem + "*"):
                                    companion.unlink(missing_ok=True)
                        except OSError:
                            pass

    @unittest.skipUnless(platform.system() == "Linux", "JITTOR_SHARED_CACHE is Linux-specific")
    def test_shared_cache_corrupt_so_recovers_and_copies_independent_inode(self):
        script = r"""
import json, os
from pathlib import Path
import numpy as np
import jittor as jt

tag = os.environ["JT_TEST_TAG"]
shared_dir = os.environ["JITTOR_SHARED_CACHE"]
with jt.flag_scope(use_cuda=0, use_jit_pch=0, use_parallel_op_compiler=0):
    x = jt.array(np.arange(8, dtype=np.float32))
    x.sync()
    shared_before = set(str(p) for p in Path(shared_dir).rglob("*.so"))
    y = jt.code(
        x.shape, x.dtype, [x],
        cpu_src=f'''
        int k_{tag} = 4;
        for (int i = 0; i < in0->num; i++)
            out0_p[i] = in0_p[i] * float(k_{tag}) + 1.0f;
        ''',
    )
    y.sync()
    shared_after = set(str(p) for p in Path(shared_dir).rglob("*.so"))
    print("CORRUPT_RESULT " + json.dumps({
        "cache_path": jt.compiler.cache_path,
        "out": y.numpy().tolist(),
        "new_shared_sos": sorted(shared_after - shared_before),
    }))
"""
        with tempfile.TemporaryDirectory(prefix="jt-shared-corrupt-") as tmp_dir:
            shared_dir = os.path.join(tmp_dir, "shared")
            scripts_dir = os.path.join(tmp_dir, "scripts")
            os.makedirs(shared_dir, exist_ok=True)
            os.makedirs(scripts_dir, exist_ok=True)
            tag = f"corrupt_so_{os.getpid()}"
            expected = (np.arange(8, dtype=np.float32) * 4.0 + 1.0).tolist()
            jit_dir = None
            try:
                # 1. Populate shared cache
                r1 = run_child_script(
                    script,
                    env={"JITTOR_SHARED_CACHE": shared_dir, "JT_TEST_TAG": tag, "JT_LOG_SILENT": "1"},
                    directory=scripts_dir, name="c1", text=True, merge_stderr=True,
                )
                self.assertEqual(r1.returncode, 0, r1.stdout[-4000:])
                res1 = json.loads(next(l for l in r1.stdout.splitlines() if l.startswith("CORRUPT_RESULT "))[len("CORRUPT_RESULT "):])
                self.assertEqual(res1["out"], expected)
                self.assertEqual(len(res1["new_shared_sos"]), 1, res1["new_shared_sos"])
                shared_so = res1["new_shared_sos"][0]

                # Verify atomic copy created an independent inode (not a hardlink)
                jit_dir = Path(res1["cache_path"]) / "jit"
                local_so = None
                for cc_file in jit_dir.glob("*.cc"):
                    if tag in cc_file.read_text(encoding="utf-8", errors="ignore"):
                        cand = jit_dir / (cc_file.stem + ".so")
                        if cand.is_file():
                            local_so = cand
                        for companion in jit_dir.glob(cc_file.stem + "*"):
                            if companion != local_so:
                                companion.unlink(missing_ok=True)
                self.assertIsNotNone(local_so, f"local .so not found for {tag}")
                self.assertNotEqual(
                    os.stat(shared_so).st_ino, os.stat(local_so).st_ino,
                    "shared cache .so must not share a mutable inode with local JITTOR_HOME")
                local_so.unlink(missing_ok=True)

                # 2. Overwrite shared .so with valid ELF magic but invalid body so
                # is_valid_elf_file() passes and dlopen() fails.
                Path(shared_so).write_bytes(b"\x7fELF" + b"\x00" * 128)

                # 3. Run with use_jit_pch=0 (cmd == base_cmd): must detect load failure,
                # remove stale local files, recompile locally, succeed, and self-heal shared .so.
                r2 = run_child_script(
                    script,
                    env={
                        "JITTOR_SHARED_CACHE": shared_dir,
                        "JT_TEST_TAG": tag,
                        "JT_LOG_SILENT": "0",
                        "JT_LOG_V": "1",
                    },
                    directory=scripts_dir, name="c2", text=True, merge_stderr=True,
                )
                self.assertEqual(r2.returncode, 0, r2.stdout[-4000:])
                self.assertIn("Shared JIT cache library failed to load, recompiling locally:", r2.stdout)
                res2 = json.loads(next(l for l in r2.stdout.splitlines() if l.startswith("CORRUPT_RESULT "))[len("CORRUPT_RESULT "):])
                self.assertEqual(res2["out"], expected)
                self.assertGreater(os.path.getsize(shared_so), 1024, "corrupt shared .so was not self-healed")

                # 4. Delete local artifacts again and confirm the self-healed shared .so now hits!
                for cc_file in jit_dir.glob("*.cc"):
                    if tag in cc_file.read_text(encoding="utf-8", errors="ignore"):
                        for companion in jit_dir.glob(cc_file.stem + "*"):
                            companion.unlink(missing_ok=True)
                r3 = run_child_script(
                    script,
                    env={
                        "JITTOR_SHARED_CACHE": shared_dir,
                        "JT_TEST_TAG": tag,
                        "JT_LOG_SILENT": "0",
                        "JT_LOG_V": "10",
                    },
                    directory=scripts_dir, name="c3", text=True, merge_stderr=True,
                )
                self.assertEqual(r3.returncode, 0, r3.stdout[-4000:])
                self.assertIn("Shared JIT cache hit:", r3.stdout)
                res3 = json.loads(next(l for l in r3.stdout.splitlines() if l.startswith("CORRUPT_RESULT "))[len("CORRUPT_RESULT "):])
                self.assertEqual(res3["out"], expected)
            finally:
                if jit_dir is not None:
                    for cc_file in jit_dir.glob("*.cc"):
                        try:
                            if tag in cc_file.read_text(encoding="utf-8", errors="ignore"):
                                for companion in jit_dir.glob(cc_file.stem + "*"):
                                    companion.unlink(missing_ok=True)
                        except OSError:
                            pass

    @unittest.skipUnless(platform.system() == "Linux", "JITTOR_SHARED_CACHE is Linux-specific")
    def test_rewrite_op_zero_respects_on_disk_edit_with_shared_cache(self):
        script = r"""
import json, os
import numpy as np
import jittor as jt

tag = os.environ["JT_TEST_TAG"]
rewrite = int(os.environ["JT_TEST_REWRITE"])
with jt.flag_scope(use_cuda=0, rewrite_op=rewrite, use_parallel_op_compiler=0):
    x = jt.array(np.arange(8, dtype=np.float32))
    y = jt.code(
        x.shape, x.dtype, [x],
        cpu_src=f'''
        int k_{tag} = 10;
        for (int i = 0; i < in0->num; i++)
            out0_p[i] = in0_p[i] + float(k_{tag});
        ''',
    )
    y.sync()
    print("RW_RESULT " + json.dumps({"cache_path": jt.compiler.cache_path, "out": y.numpy().tolist()}))
"""
        with tempfile.TemporaryDirectory(prefix="jt-shared-rw0-") as tmp_dir:
            shared_dir = os.path.join(tmp_dir, "shared")
            scripts_dir = os.path.join(tmp_dir, "scripts")
            os.makedirs(shared_dir, exist_ok=True)
            os.makedirs(scripts_dir, exist_ok=True)
            tag = f"rw0_edit_{os.getpid()}"
            jit_dir = None
            try:
                r1 = run_child_script(
                    script,
                    env={
                        "JITTOR_SHARED_CACHE": shared_dir,
                        "JT_TEST_TAG": tag,
                        "JT_TEST_REWRITE": "1",
                        "JT_LOG_SILENT": "1",
                    },
                    directory=scripts_dir, name="rw1", text=True, merge_stderr=True,
                )
                self.assertEqual(r1.returncode, 0, r1.stdout[-4000:])
                res1 = json.loads(next(l for l in r1.stdout.splitlines() if l.startswith("RW_RESULT "))[len("RW_RESULT "):])
                self.assertEqual(res1["out"], (np.arange(8, dtype=np.float32) + 10.0).tolist())

                # Edit the generated .cc on disk (10 -> 77) and delete the local .so/.key/.d
                # while keeping the edited .cc so rewrite_op=0 compiles the edited file and
                # MUST NOT hit the shared cache entry for the unedited 10-version.
                jit_dir = Path(res1["cache_path"]) / "jit"
                edited = False
                for cc_file in jit_dir.glob("*.cc"):
                    text = cc_file.read_text(encoding="utf-8", errors="ignore")
                    if tag in text:
                        self.assertIn(f"int k_{tag} = 10;", text)
                        cc_file.write_text(text.replace(f"int k_{tag} = 10;", f"int k_{tag} = 77;"), encoding="utf-8")
                        for ext in (".so", ".so.key", ".so.d"):
                            (jit_dir / (cc_file.stem + ext)).unlink(missing_ok=True)
                        edited = True
                self.assertTrue(edited, f"failed to locate and edit .cc for {tag}")

                r2 = run_child_script(
                    script,
                    env={
                        "JITTOR_SHARED_CACHE": shared_dir,
                        "JT_TEST_TAG": tag,
                        "JT_TEST_REWRITE": "0",
                        "JT_LOG_SILENT": "0",
                        "JT_LOG_V": "10",
                    },
                    directory=scripts_dir, name="rw2", text=True, merge_stderr=True,
                )
                self.assertEqual(r2.returncode, 0, r2.stdout[-4000:])
                self.assertNotIn("Shared JIT cache hit:", r2.stdout)
                res2 = json.loads(next(l for l in r2.stdout.splitlines() if l.startswith("RW_RESULT "))[len("RW_RESULT "):])
                self.assertEqual(res2["out"], (np.arange(8, dtype=np.float32) + 77.0).tolist())
            finally:
                if jit_dir is not None:
                    for cc_file in jit_dir.glob("*.cc"):
                        try:
                            if tag in cc_file.read_text(encoding="utf-8", errors="ignore"):
                                for companion in jit_dir.glob(cc_file.stem + "*"):
                                    companion.unlink(missing_ok=True)
                        except OSError:
                            pass



class TestFastDevAndCcache(unittest.TestCase):
    def test_cuda_arch_flags_fast_dev_omits_ptx(self):
        with_ptx = compiler.cuda_arch_flags([89], include_ptx=True)
        no_ptx = compiler.cuda_arch_flags([89], include_ptx=False)
        self.assertIn("--generate-code=arch=compute_89,code=sm_89", with_ptx)
        self.assertIn("--generate-code=arch=compute_89,code=compute_89", with_ptx)
        self.assertIn("--generate-code=arch=compute_89,code=sm_89", no_ptx)
        self.assertNotIn("code=compute_89", no_ptx)

    def test_fast_dev_env_helper_and_cache_config_isolation(self):
        self.assertFalse(compiler.fast_dev_enabled({}))
        self.assertFalse(compiler.fast_dev_enabled({"JT_FAST_DEV": "0"}))
        self.assertTrue(compiler.fast_dev_enabled({"JT_FAST_DEV": "1"}))
        base_fp = jit_utils.build_config_fingerprint({"nvcc_path": "/bin/nvcc"})
        fast_fp = jit_utils.build_config_fingerprint(
            {"nvcc_path": "/bin/nvcc", "JT_FAST_DEV": "1"})
        self.assertNotEqual(base_fp, fast_fp)

    @unittest.skipIf(os.name == "nt", "POSIX PATH launcher test")
    def test_ccache_prefix_opt_in_and_fallback(self):
        self.assertEqual(compiler.ccache_compile_prefix(compiler.cc_path, environ={}), ("", ""))
        self.assertEqual(
            compiler.ccache_compile_prefix(compiler.cc_path, environ={"JT_CCACHE": "0"}),
            ("", ""),
        )
        with tempfile.TemporaryDirectory(prefix="jt-fake-ccache-") as tmp_dir:
            fake = Path(tmp_dir) / "ccache"
            fake.write_text("#!/bin/sh\nexec \"$@\"\n", encoding="utf-8")
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            old_path = os.environ.get("PATH", "")
            try:
                os.environ["PATH"] = tmp_dir + os.pathsep + old_path
                prefix, extra = compiler.ccache_compile_prefix(
                    compiler.cc_path, compiler.jittor_path, environ={"JT_CCACHE": "1"})
                self.assertIn(str(fake), prefix)
                if extra:
                    self.assertIn("-ffile-prefix-map=", extra)
            finally:
                os.environ["PATH"] = old_path

    @unittest.skipUnless(_has_cuda(), "no usable CUDA in this build")
    def test_fast_dev_cuda_kernel_compiles_and_executes(self):
        script = r"""
import json
import numpy as np
import jittor as jt

with jt.flag_scope(use_cuda=1):
    x = jt.array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    y = (x * 3.0 + 1.0).sin()
    y.sync()
    print("FASTDEV_RESULT " + json.dumps({
        "nvcc_flags": jt.compiler.nvcc_flags,
        "out": y.numpy().tolist(),
    }))
"""
        with tempfile.TemporaryDirectory(prefix="jt-fastdev-") as tmp_dir:
            r = run_child_script(
                script,
                env={"JT_FAST_DEV": "1", "JT_LOG_SILENT": "1"},
                directory=tmp_dir,
                name="fastdev_probe",
                text=True,
                merge_stderr=True,
            )
            self.assertEqual(r.returncode, 0, r.stdout[-4000:])
            line = next(l for l in r.stdout.splitlines() if l.startswith("FASTDEV_RESULT "))
            res = json.loads(line[len("FASTDEV_RESULT "):])
            self.assertIn("code=sm_", res["nvcc_flags"])
            self.assertNotIn("code=compute_", res["nvcc_flags"])
            expected = np.sin(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) * 3.0 + 1.0)
            np.testing.assert_allclose(res["out"], expected, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(_has_cuda(), "no usable CUDA in this build")
class TestNormEpsNoRecompileChurn(unittest.TestCase):
    def _ref_layer_norm(self, x, w, b, eps):
        x64 = x.astype(np.float64)
        mean = x64.mean(axis=-1, keepdims=True)
        var = x64.var(axis=-1, keepdims=True)
        return ((x64 - mean) / np.sqrt(var + eps) * w + b).astype(np.float32)

    def _ref_rms_norm(self, x, w, eps):
        x64 = x.astype(np.float64)
        inv = 1.0 / np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + eps)
        return (x64 * inv * w).astype(np.float32)

    def test_layer_norm_and_rms_norm_reuse_single_kernel_across_eps(self):
        rs = np.random.RandomState(2026)
        # Use a spread of eps values including a large one (1e-1) so outputs
        # differ noticeably, proving eps is passed at runtime while reusing
        # the exact same compiled .so with 0 extra compilations.
        eps_values = [1e-5, 1e-6, 2e-5, 3e-5, 1e-4, 1e-1]
        x_np = rs.randn(128, 192).astype(np.float32)
        w_np = rs.uniform(0.5, 1.5, (192,)).astype(np.float32)
        b_np = rs.uniform(-0.5, 0.5, (192,)).astype(np.float32)

        with jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0):
            # 1. LayerNorm inference
            with jt.no_grad():
                y0 = nn.layer_norm(jt.array(x_np), (192,), jt.array(w_np), jt.array(b_np), eps=eps_values[0])
                y0.sync()
                np.testing.assert_allclose(y0.numpy(), self._ref_layer_norm(x_np, w_np, b_np, eps_values[0]), rtol=1e-4, atol=1e-4)
                so_after_first = _jit_so_set(compiler.cache_path)
                for eps in eps_values[1:]:
                    y = nn.layer_norm(jt.array(x_np), (192,), jt.array(w_np), jt.array(b_np), eps=eps)
                    y.sync()
                    np.testing.assert_allclose(y.numpy(), self._ref_layer_norm(x_np, w_np, b_np, eps), rtol=1e-4, atol=1e-4)
                self.assertEqual(
                    _jit_so_set(compiler.cache_path) - so_after_first, set(),
                    "layer_norm inference recompiled for new eps values")

            # 2. LayerNorm training (fwd + bwd)
            def run_ln_train(eps):
                x = jt.array(x_np); x.requires_grad = True
                w = jt.array(w_np); w.requires_grad = True
                b = jt.array(b_np); b.requires_grad = True
                y = nn.layer_norm(x, (192,), w, b, eps=eps)
                gx, gw, gb = jt.grad((y * y).sum(), [x, w, b])
                jt.sync([y, gx, gw, gb])
                return y.numpy(), gx.numpy(), gw.numpy(), gb.numpy()

            y0, _, _, _ = run_ln_train(eps_values[0])
            np.testing.assert_allclose(y0, self._ref_layer_norm(x_np, w_np, b_np, eps_values[0]), rtol=1e-4, atol=1e-4)
            so_after_first = _jit_so_set(compiler.cache_path)
            for eps in eps_values[1:]:
                y, _, _, _ = run_ln_train(eps)
                np.testing.assert_allclose(y, self._ref_layer_norm(x_np, w_np, b_np, eps), rtol=1e-4, atol=1e-4)
            self.assertEqual(
                _jit_so_set(compiler.cache_path) - so_after_first, set(),
                "layer_norm training recompiled for new eps values")

            # 3. RMSNorm inference
            with jt.no_grad():
                y0 = nn.rms_norm(jt.array(x_np), jt.array(w_np), eps=eps_values[0])
                y0.sync()
                np.testing.assert_allclose(y0.numpy(), self._ref_rms_norm(x_np, w_np, eps_values[0]), rtol=1e-4, atol=1e-4)
                so_after_first = _jit_so_set(compiler.cache_path)
                for eps in eps_values[1:]:
                    y = nn.rms_norm(jt.array(x_np), jt.array(w_np), eps=eps)
                    y.sync()
                    np.testing.assert_allclose(y.numpy(), self._ref_rms_norm(x_np, w_np, eps), rtol=1e-4, atol=1e-4)
                self.assertEqual(
                    _jit_so_set(compiler.cache_path) - so_after_first, set(),
                    "rms_norm inference recompiled for new eps values")

            # 4. RMSNorm training (fwd + bwd)
            def run_rms_train(eps):
                x = jt.array(x_np); x.requires_grad = True
                w = jt.array(w_np); w.requires_grad = True
                y = nn.rms_norm(x, w, eps=eps)
                gx, gw = jt.grad((y * y).sum(), [x, w])
                jt.sync([y, gx, gw])
                return y.numpy(), gx.numpy(), gw.numpy()

            y0, _, _ = run_rms_train(eps_values[0])
            np.testing.assert_allclose(y0, self._ref_rms_norm(x_np, w_np, eps_values[0]), rtol=1e-4, atol=1e-4)
            so_after_first = _jit_so_set(compiler.cache_path)
            for eps in eps_values[1:]:
                y, _, _ = run_rms_train(eps)
                np.testing.assert_allclose(y, self._ref_rms_norm(x_np, w_np, eps), rtol=1e-4, atol=1e-4)
            self.assertEqual(
                _jit_so_set(compiler.cache_path) - so_after_first, set(),
                "rms_norm training recompiled for new eps values")


if __name__ == "__main__":
    unittest.main()
