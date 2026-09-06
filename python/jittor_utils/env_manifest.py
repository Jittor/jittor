# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The list of every environment variable Jittor reads, generated from source.

    python -m jittor_utils.env_manifest            # markdown table
    python -m jittor_utils.env_manifest --json     # the same, for scripts

Nothing hand-maintained: the native flags come from the ``DEFINE_FLAG`` sites in
``src/`` (the same scan ``compiler.py:gen_jit_flags`` does, so the manifest
cannot list a flag the core does not have or miss one it does) and the
Python-only settings come from the tables in :mod:`jittor_utils.env_config`.

There was no such list. Names were spread over four conventions -- unprefixed
lower case, unprefixed upper case, ``JITTOR_``, ``JT_`` -- across ``log.h``,
``compiler.py``, ``compile_extern.py``, ``jittor_utils/`` and the compat shim,
and no single place in the repository or the documentation had all of them. The
practical consequence was that ``export debug=1`` was documented and
``export name=x`` was believed to be a flag (it never was: the audit read
``DEFINE_FLAG(type, name, default, doc)`` and took the macro's own parameter for
a flag name), and nobody could check either claim.

This module is also the gate's source of truth:
``tests/structure/test_env_var_manifest.py`` fails when a setting is read
under an unprefixed name anywhere outside :mod:`jittor_utils.env_config`.
"""

import argparse
import json
import os
from typing import Dict, Tuple
import re
import runpy

from . import env_config

#: The `DEFINE_FLAG` sites, as `gen_jit_flags` reads them.
_DEFINE_FLAG = re.compile(r"DEFINE_(RUNTIME_)?FLAG(_WITH_SETTER)?\((.*?)\);",
                          re.DOTALL)

#: Macros that guard definitions compiled only into the C++ self-test binaries,
#: which are therefore not settings of a real jittor process. ``nthread`` is the
#: only flag behind one, and it is why a whole-tree grep for DEFINE_FLAG reports
#: one flag more than ``jt.flags`` has.
_TEST_ONLY_MACROS = ("TEST_LOG", "TEST_OP", "TEST")


def _drop_test_only_regions(src):
    """``src`` with every ``#ifdef TEST*`` block removed.

    Tracking the nesting matters: ``log.cc`` has an ``#ifdef TEST_LOG`` inside a
    function 17 lines *above* the definitions of ``log_v``, ``log_silent``,
    ``log_sync``, ``log_file`` and ``log_vprefix``. Cutting the file at the first
    occurrence dropped the five most commonly set flags in the framework from the
    manifest -- and the generated list looked complete either way.
    """
    kept = []
    depth = 0          # nesting depth of any conditional
    skipping_from = None
    for line in src.splitlines(True):
        stripped = line.lstrip()
        if stripped.startswith(("#if", "#ifdef", "#ifndef")):
            depth += 1
            if skipping_from is None and any(
                    macro in stripped for macro in _TEST_ONLY_MACROS):
                skipping_from = depth
        elif stripped.startswith("#endif"):
            if skipping_from is not None and depth == skipping_from:
                skipping_from = None
                depth -= 1
                continue
            depth -= 1
        if skipping_from is None:
            kept.append(line)
    return "".join(kept)


def _strip_cxx_comments(src):
    # A commented-out DEFINE_FLAG must not become a manifest entry; see
    # compiler.py's copy of this for the bug that motivated it.
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in '"\'':
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == '\\' else 1
            out.append(src[i:min(j + 1, n)])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def native_flags(source_root):
    """``{flag: (type, default, doc)}`` for every flag the core defines."""
    found: Dict[str, Tuple[str, str, str]] = {}
    for base, _dirs, names in os.walk(source_root):
        for name in sorted(names):
            if not name.endswith((".cc", ".h", ".cu")):
                continue
            path = os.path.join(base, name)
            with open(path, "r", encoding="utf8", errors="ignore") as handle:
                text = handle.read()
            # log.h defines the macros themselves; its `DEFINE_FLAG(type, name,
            # default, doc)` parameter list is not a flag called "name".
            if os.path.basename(path) == "log.h":
                continue
            live = _drop_test_only_regions(_strip_cxx_comments(text))
            for _runtime, _setter, args in _DEFINE_FLAG.findall(live):
                parts = args.split(",")
                if len(parts) < 4:
                    continue
                flag = parts[1].strip()
                found.setdefault(flag, (parts[0].strip(), parts[2].strip(),
                                        ",".join(parts[3:]).strip()))
    return found


def default_source_root():
    """The ``src/`` tree :func:`native_flags` scans.

    Named rather than computed inline so the gate can assert the root it checks
    is the root the manifest actually reads: a scan root that stops matching the
    tree yields no flags, and a manifest generated from nothing looks exactly
    like a complete one.
    """
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "jittor", "src")


def entries(source_root=None):
    """Every setting, as dicts sorted by canonical variable name."""
    if source_root is None:
        source_root = default_source_root()
    rows = []
    for flag, (ctype, default, doc) in native_flags(source_root).items():
        namespace = "build" if flag in _startup_flags() else "runtime"
        rows.append({
            "setting": flag,
            "variable": env_config.prefixed_name(flag, namespace),
            "namespace": namespace,
            "deprecated_name": None if "_" not in flag else flag,
            "kind": "native flag",
            "type": ctype,
            "default": default,
            "doc": doc.strip('" '),
        })
    for table, namespace, kind in ((env_config.PYTHON_BUILD_VARS, "build",
                                    "python build variable"),
                                   (env_config.PYTHON_RUNTIME_VARS, "runtime",
                                    "python runtime variable")):
        for name, doc in table.items():
            rows.append({
                "setting": name,
                "variable": env_config.prefixed_name(name, namespace),
                "namespace": namespace,
                "deprecated_name": None if "_" not in name else name,
                "kind": kind,
                "type": "string",
                "default": "",
                "doc": doc,
            })
    return sorted(rows, key=lambda row: row["variable"])


def flag_policy_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "jittor", "_runtime", "flag_policy.py")


def _startup_flags():
    """``STARTUP_FLAGS`` read from its source file.

    ``import jittor._runtime.flag_policy`` would import the ``jittor`` package,
    i.e. build the core -- and this module has to be usable to *find out* how to
    configure that build.
    """
    namespace = runpy.run_path(flag_policy_path())
    return namespace["STARTUP_FLAGS"]


def markdown(rows=None):
    rows = entries() if rows is None else rows
    lines = ["| variable | namespace | deprecated name | type | default | what it does |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append("| `%s` | %s | %s | %s | `%s` | %s |" % (
            row["variable"], row["namespace"],
            ("`%s`" % row["deprecated_name"]) if row["deprecated_name"]
            else "*(none: an ordinary word is never read unprefixed)*",
            row["type"], row["default"], row["doc"]))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output")
    args = parser.parse_args(argv)
    rows = entries()
    print(json.dumps(rows, indent=2, sort_keys=True) if args.json
          else markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
