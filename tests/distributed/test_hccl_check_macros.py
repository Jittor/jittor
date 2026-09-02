# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""HCCL's error macros must throw, not return (6.B03).

``ACLCHECK`` and ``HCCLCHECK`` used to log at LOGe and then ``return``. Inside a
collective operator's ``jit_run()`` -- which returns void -- that meant a failed
collective simply did not happen: the output var kept whatever was in it, the
rank continued, and nothing above a log line said anything was wrong. Every rank
then carried on with garbage.

NOTE ON VERIFICATION: there is no Ascend hardware available, so none of the HCCL
code can be built against the real CANN headers, let alone run. What this test
does check is the part that is checkable anywhere: it lifts the macro
definitions verbatim out of hccl_wrapper.h, compiles them against minimal stubs,
and asserts their control flow -- throw on failure, nothing on success, single
evaluation of the argument, and no throw from the shutdown-only variant (a throw
there would be std::terminate, since it runs in a destructor). The surrounding
HCCL code remains unverified on hardware.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import jittor as jt

_HCCL_HEADER = (Path(__file__).resolve().parents[2] / "python" / "jittor"
                / "extern" / "acl" / "hccl" / "inc" / "hccl_wrapper.h")

_HARNESS = r"""
#include <sstream>
#include <stdexcept>
#include <string>
#include <cstdio>

// --- minimal stand-ins for the jittor logging macros -----------------------
// jittor's LOGf is `LogFatalVoidify() && Log(...)`; `<<` binds tighter than
// `&&`, so the whole message is built before the fatal handler sees it. Same
// shape here, so the macros under test are exercised exactly as written.
static std::string g_last_error;

struct Msg {
    std::ostringstream o;
    template <class T> Msg& operator<<(const T& v) { o << v << ' '; return *this; }
};
static Msg& msg_() { static Msg m; m.o.str(""); m.o.clear(); return m; }
struct Fatal {
    bool operator&&(Msg& m) const { throw std::runtime_error(m.o.str()); return false; }
};
struct Err {
    bool operator&&(Msg& m) const { g_last_error = m.o.str(); return true; }
};
// No parentheses, exactly like jittor's own definition: `<<` binds tighter than
// `&&`, so the message is fully built before the handler runs.
#define LOGf Fatal() && msg_()
#define LOGe Err() && msg_()

// --- minimal stand-ins for the ACL / HCCL types ----------------------------
enum { ACL_SUCCESS = 0, HCCL_SUCCESS = 0 };
static const char* HcclGetErrorString(int code) { (void)code; return "stub-error"; }

// --- the macros under test, lifted verbatim from hccl_wrapper.h ------------
%(macros)s

// --- the checks ------------------------------------------------------------
static int g_eval_count = 0;
static int counted(int value) { g_eval_count++; return value; }

static int failures = 0;
static void expect(bool ok, const char* what) {
    if (!ok) { printf("FAIL %%s\n", what); failures++; }
    else printf("ok   %%s\n", what);
}

int main() {
    bool threw;

    threw = false;
    try { ACLCHECK(1); } catch (const std::runtime_error&) { threw = true; }
    expect(threw, "ACLCHECK throws on failure");

    threw = false;
    try { ACLCHECK(ACL_SUCCESS); } catch (const std::runtime_error&) { threw = true; }
    expect(!threw, "ACLCHECK is silent on success");

    threw = false;
    try { HCCLCHECK(2); } catch (const std::runtime_error&) { threw = true; }
    expect(threw, "HCCLCHECK throws on failure");

    threw = false;
    try { HCCLCHECK(HCCL_SUCCESS); } catch (const std::runtime_error&) { threw = true; }
    expect(!threw, "HCCLCHECK is silent on success");

    // A throw out of a destructor is std::terminate, so the shutdown variant
    // must stay non-throwing.
    threw = false;
    try { HCCLCHECK_PEEK(3); } catch (const std::runtime_error&) { threw = true; }
    expect(!threw, "HCCLCHECK_PEEK does not throw");
    expect(!g_last_error.empty(), "HCCLCHECK_PEEK still reports the error");

    // The old ACLCHECK evaluated its argument twice on the failure path.
    g_eval_count = 0;
    try { ACLCHECK(counted(1)); } catch (const std::runtime_error&) {}
    expect(g_eval_count == 1, "ACLCHECK evaluates its argument exactly once");

    g_eval_count = 0;
    try { HCCLCHECK(counted(1)); } catch (const std::runtime_error&) {}
    expect(g_eval_count == 1, "HCCLCHECK evaluates its argument exactly once");

    // Usable as a plain statement inside an if/else with no braces.
    if (failures == 0) ACLCHECK(ACL_SUCCESS); else ACLCHECK(ACL_SUCCESS);

    printf("RESULT %%s\n", failures ? "FAIL" : "ALL PASS");
    return failures ? 1 : 0;
}
"""


def _extract_macros():
    """Lift the macro definitions verbatim: from `#define NAME(` up to the first
    line that does not end in a backslash continuation."""
    lines = _HCCL_HEADER.read_text().splitlines()
    out = []
    for name in ("ACLCHECK", "HCCLCHECK", "HCCLCHECK_PEEK"):
        start = next((i for i, line in enumerate(lines)
                      if line.startswith("#define " + name + "(")), None)
        assert start is not None, "macro %s not found in hccl_wrapper.h" % name
        end = start
        while lines[end].rstrip().endswith("\\"):
            end += 1
        out.append("\n".join(lines[start:end + 1]))
    return "\n\n".join(out)


class TestHcclCheckMacros(unittest.TestCase):

    def test_macros_throw_instead_of_returning(self):
        macros = _extract_macros()
        # The old shape is what this whole task is about; make sure it is gone.
        self.assertNotIn("return;", macros,
                         "an HCCL check macro still swallows the error with return")

        cc = getattr(jt.compiler, "cc_path", None) or "g++"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "hccl_macros.cc"
            src.write_text(_HARNESS % {"macros": macros})
            binary = tmp / "hccl_macros"
            build = subprocess.run(
                [cc, "-O0", "-std=c++14", "-o", os.fspath(binary), os.fspath(src)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
            self.assertEqual(build.returncode, 0,
                             "the macros do not compile:\n" + build.stdout)
            run = subprocess.run([os.fspath(binary)], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, timeout=600)
            self.assertEqual(run.returncode, 0, run.stdout)
            self.assertIn("RESULT ALL PASS", run.stdout)

    def test_finalizer_does_not_use_the_throwing_macro(self):
        # ~hccl_finalizer() runs at process teardown; a throw there terminates.
        source = (_HCCL_HEADER.parent.parent / "src" / "hccl_wrapper.cc").read_text()
        body = source[source.index("struct hccl_finalizer"):]
        body = body[:body.index("};")]
        self.assertIn("HCCLCHECK_PEEK", body)
        self.assertNotIn("HCCLCHECK(", body)


if __name__ == "__main__":
    unittest.main()
