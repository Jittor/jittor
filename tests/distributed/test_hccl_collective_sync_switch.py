# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""HCCL's four-syncs-per-collective, now behind one switch (8.02).

Every HCCL collective used to issue ``aclrtSynchronizeDevice()`` +
``aclrtSynchronizeStream(aclstream)`` both before and after the call -- four
full synchronisations per collective, written out four times, once per
operator. That serialises the whole NPU pipeline around every gradient
exchange.

NOTE ON VERIFICATION: there is no Ascend hardware here, so none of the HCCL
code can be compiled against real CANN headers, let alone run, and the syncs
were therefore *not* deleted. They moved behind
``JT_HCCL_COLLECTIVE_SYNC``, default ``full`` (bit-for-bit the old behaviour),
with ``stream-order`` dropping them. The on-device checklist that has to pass
before the default flips is ``agent/manuals/hccl-on-device-verification.md``.

What is checkable anywhere is what this test does, following
``test_hccl_check_macros.py``: lift the two bracket helpers verbatim out of
``hccl_wrapper.h``, compile them against minimal stubs, and assert that the
switch actually controls the synchronisations -- and that no operator writes
them out inline any more, since that is how "one place to change" gets
silently undone.
"""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import jittor as jt

_HCCL = (Path(__file__).resolve().parents[2] / "python" / "jittor" / "extern"
         / "acl" / "hccl")
_HEADER = _HCCL / "inc" / "hccl_wrapper.h"
_OPS = ["hccl_all_reduce_op.cc", "hccl_all_gather_op.cc", "hccl_reduce_op.cc",
        "hccl_broadcast_op.cc"]

_HARNESS = r"""
#include <cstdio>
#include <cstring>

// --- minimal stand-ins for the ACL runtime ---------------------------------
enum { ACL_SUCCESS = 0 };
static int g_device_syncs = 0;
static int g_stream_syncs = 0;
typedef void* aclrtStream;
static aclrtStream aclstream = nullptr;
static int aclrtSynchronizeDevice() { g_device_syncs++; return ACL_SUCCESS; }
static int aclrtSynchronizeStream(aclrtStream) { g_stream_syncs++; return ACL_SUCCESS; }
#define ACLCHECK(x) do { (void)(x); } while (0)

// The switch itself lives in hccl_wrapper.cc and reads the environment; here it
// is driven directly so both branches are exercised in one binary.
static bool g_full = true;
static bool hccl_collective_full_sync() { return g_full; }

// --- the bracket helpers, lifted verbatim from hccl_wrapper.h --------------
%(helpers)s

static int failures = 0;
static void expect(bool ok, const char* what) {
    if (!ok) { printf("FAIL %%s\n", what); failures++; }
    else printf("ok   %%s\n", what);
}

int main() {
    g_full = true;
    g_device_syncs = g_stream_syncs = 0;
    hccl_collective_begin();
    hccl_collective_end();
    // The historical behaviour, unchanged: two device syncs and two stream
    // syncs around one collective.
    expect(g_device_syncs == 2, "full: two aclrtSynchronizeDevice");
    expect(g_stream_syncs == 2, "full: two aclrtSynchronizeStream");

    g_full = false;
    g_device_syncs = g_stream_syncs = 0;
    hccl_collective_begin();
    hccl_collective_end();
    expect(g_device_syncs == 0, "stream-order: no aclrtSynchronizeDevice");
    expect(g_stream_syncs == 0, "stream-order: no aclrtSynchronizeStream");

    printf("RESULT %%s\n", failures ? "FAIL" : "ALL PASS");
    return failures ? 1 : 0;
}
"""


def _extract(name):
    """Lift `inline void NAME() { ... }` verbatim, braces balanced."""
    text = _HEADER.read_text()
    start = text.index("inline void " + name + "()")
    depth = 0
    for i in range(text.index("{", start), len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise AssertionError("unbalanced braces around " + name)


class TestHcclCollectiveSyncSwitch(unittest.TestCase):

    def test_the_switch_controls_the_synchronisations(self):
        helpers = "\n\n".join(_extract(n) for n in
                              ("hccl_collective_begin", "hccl_collective_end"))
        self.assertIn("hccl_collective_full_sync()", helpers)

        cc = getattr(jt.compiler, "cc_path", None) or "g++"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "hccl_sync.cc"
            src.write_text(_HARNESS % {"helpers": helpers})
            binary = tmp / "hccl_sync"
            build = subprocess.run(
                [cc, "-O0", "-std=c++14", "-o", os.fspath(binary),
                 os.fspath(src)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                timeout=600)
            self.assertEqual(build.returncode, 0,
                             "the bracket helpers do not compile:\n"
                             + build.stdout)
            run = subprocess.run([os.fspath(binary)], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 timeout=600)
            self.assertEqual(run.returncode, 0, run.stdout)
            self.assertIn("RESULT ALL PASS", run.stdout)

    def test_no_operator_writes_the_synchronisations_inline(self):
        # Four copies of the same two lines is how the audit found this in the
        # first place, and re-adding one would put an operator outside the
        # switch -- green everywhere, wrong only on hardware.
        offenders = []
        for name in _OPS:
            body = (_HCCL / "ops" / name).read_text()
            body = body[body.index("#else // JIT"):]
            self.assertIn("hccl_collective_begin();", body, name)
            self.assertIn("hccl_collective_end();", body, name)
            for match in re.finditer(
                    r"^[ \t]*ACLCHECK\(aclrtSynchronize(Device|Stream)",
                    body, re.M):
                offenders.append("{}: {}".format(name, match.group(0).strip()))
        # hccl_broadcast_op keeps exactly one, after its blocking aclrtMemcpy
        # on the root branch. It is a different sync from the four this task is
        # about and is equally unverifiable here; the manual says so.
        self.assertEqual(
            offenders,
            ["hccl_broadcast_op.cc: ACLCHECK(aclrtSynchronizeDevice"],
            "an HCCL operator synchronises outside hccl_collective_begin/end")

    def test_the_switch_rejects_an_unknown_value(self):
        source = (_HCCL / "src" / "hccl_wrapper.cc").read_text()
        body = source[source.index("bool hccl_collective_full_sync()"):]
        body = body[:body.index("\n}")]
        # Falling back to a default on a typo is the failure mode that makes an
        # on-device A/B meaningless: the "stream-order" run silently measures
        # "full" again and the two look identical.
        self.assertIn("LOGf", body)
        self.assertIn("stream-order", body)


if __name__ == "__main__":
    unittest.main()
