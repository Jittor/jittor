#!/usr/bin/env python3
"""Syntax-check ACL backend sources on a host that has no CANN SDK.

Usage:

    python syntax_check.py --repo <worktree> --jittor-cache <cfg dir> \\
        python/jittor/extern/acl/aclops/where_op_acl.cc ...

What it does and does not prove
-------------------------------
It runs ``g++ -fsyntax-only`` over the real translation unit, with the real
Jittor core headers and a generated stub CANN tree (``make_cann_stub.py``). So
it does catch: parse errors, unknown identifiers, wrong argument counts to
Jittor-side helpers, and -- the reason this exists for 8.06 -- a launcher that
does not convert to ``BaseOpRunner::launch``'s ``AclExecuteLauncher``, because
the stub declares every ``aclnn`` execute entry point with its real
``(void*, uint64_t, aclOpExecutor*, aclrtStream)`` ABI.

It does not prove the operator is called correctly. The per-operator
``aclnnXxxGetWorkspaceSize`` signatures are not knowable without the SDK, so
they are stubbed variadic. A consequence is that the ``AclOpFunctions``
constructor overload set in ``acl_jittor.h`` becomes ambiguous against those
variadic stubs, and that one file always produces diagnostics under the stub.
Those are filtered by file, and any diagnostic pointing at a requested source
-- or at any other header -- fails the check. Argument-level mistakes in a
workspace query still need a real Ascend 910B3 build.

``--check-launchers`` adds a separate generated translation unit that asserts
every ``launch(ret, <symbol>, ...)`` site names a function with the exact
execute ABI. This is needed because ``-fsyntax-only`` alone cannot catch a
workspace query passed where a launcher belongs: ``std::function`` accepts a
variadic stub, so the mistake compiles. Comparing raw function pointer types
does reject it.
"""

import argparse
import pathlib
import platform
import re
import subprocess
import sys

import make_cann_stub

# acl_jittor.h holds the aclOpFuncMap table whose overload set cannot be
# resolved against variadic stubs. Nothing else may report a diagnostic.
STUB_AMBIGUITY_ONLY = "acl_jittor.h"

DIAGNOSTIC = re.compile(r"^(?P<file>[^\s:][^:]*):(?P<line>\d+):(?P<col>\d+): "
                        r"(?P<kind>error|warning|fatal error):")

LAUNCH_SITE = re.compile(r"\blaunch\(\s*\w+\s*,\s*(aclnn[A-Za-z0-9_]*)\s*,")

LAUNCHER_TU = """#include "acl/acl.h"
#include "acl/aclnn_entry_points.h"
#include <type_traits>

typedef aclnnStatus (*AclExecuteAbi)(void *, uint64_t, aclOpExecutor *, aclrtStream);
{asserts}
"""


def compile_flags(repo, cache, stub):
    acl = repo / "python" / "jittor" / "extern" / "acl"
    extra = []
    if platform.machine() != "aarch64":
        # __fp16 is an aarch64 builtin and CANN hosts are aarch64. On an x86_64
        # review host the equivalent storage type is _Float16; without this the
        # fp16 branch of binary_op_acl.cc cannot be parsed at all.
        extra.append("-D__fp16=_Float16")
    return extra + [
        # IS_CUDA is what makes extern/cuda/inc/helper_cuda.h self-consistent:
        # without it the header still parses findCudaDevice but not the
        # helper_string.h it calls into.
        "-fsyntax-only", "-std=c++14", "-fPIC",
        "-DHAS_CUDA", "-DIS_CUDA", "-DIS_ACL", "-w",
        "-I", str(stub), "-I", str(stub / "acl"),
        "-I", str(repo / "python" / "jittor" / "src"),
        "-I", str(repo / "python" / "jittor" / "extern"),
        "-I", str(acl), "-I", str(acl / "aclnn"), "-I", str(acl / "aclops"),
        "-I", str(repo / "python" / "jittor" / "extern" / "cuda" / "inc"),
        "-I", "/usr/local/cuda/include",
        "-I", str(cache),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="worktree root")
    parser.add_argument("--jittor-cache", required=True,
                        help="jittor.compiler cache_path (holds generated headers)")
    parser.add_argument("--python-include", required=True)
    parser.add_argument("--stub", required=True, help="where to build the CANN stub tree")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--check-launchers", action="store_true",
                        help="also assert every launch() site names an execute ABI")
    parser.add_argument("sources", nargs="+")
    args = parser.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    stub = pathlib.Path(args.stub).resolve()
    acl_root = repo / "python" / "jittor" / "extern" / "acl"

    make_cann_stub.build(acl_root, stub)

    flags = compile_flags(repo, pathlib.Path(args.jittor_cache), stub)
    flags += ["-I", args.python_include]

    failed = False
    for source in args.sources:
        result = subprocess.run([args.cxx] + flags + [source],
                                cwd=str(repo), capture_output=True, text=True)
        unexpected = []
        for line in result.stderr.splitlines():
            match = DIAGNOSTIC.match(line)
            if match and pathlib.Path(match.group("file")).name != STUB_AMBIGUITY_ONLY:
                unexpected.append(line)
        if unexpected:
            failed = True
            print("FAIL {}".format(source))
            for line in unexpected:
                print("  " + line)
        else:
            print("ok   {}".format(source))

    if args.check_launchers:
        asserts = []
        for source in args.sources:
            text = (repo / source).read_text(encoding="utf-8")
            for symbol in sorted(set(LAUNCH_SITE.findall(text))):
                asserts.append(
                    'static_assert(std::is_same<decltype(&{0}), AclExecuteAbi>::value,\n'
                    '    "{0} is passed to BaseOpRunner::launch but does not have the '
                    'aclnn execute ABI");'.format(symbol))
        if asserts:
            tu = stub / "launcher_abi_check.cc"
            tu.write_text(LAUNCHER_TU.format(asserts="\n".join(asserts)), encoding="utf-8")
            result = subprocess.run(
                [args.cxx, "-fsyntax-only", "-std=c++14",
                 "-I", str(stub), "-I", str(stub / "acl"), str(tu)],
                capture_output=True, text=True)
            if result.returncode != 0:
                failed = True
                print("FAIL launcher ABI check")
                print(result.stderr.strip())
            else:
                print("ok   launcher ABI ({} symbols)".format(len(asserts)))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
