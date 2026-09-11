# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The environment-variable list is generated, complete, and stays that way. [2.22]

``python -m jittor_utils.env_manifest`` is the answer to "which variables does
jittor read", a question that had no answer: the names lived in ``log.h``,
``compiler.py``, ``compile_extern.py``, ``jittor_utils/`` and the compat shim
under four different conventions, and no place in the repository or the
documentation listed them. A generated list only stays honest if something fails
when a new name appears outside it, so this module is the teeth:

* the manifest is generated from the ``DEFINE_FLAG`` sites and from
  ``env_config``'s tables, so it cannot list a flag the core lacks or miss one it
  has;
* the C++ and Python halves of the build/runtime partition are the same set --
  ``log.cc`` has to name the build flags itself because it cannot import Python,
  and two copies of a list drift;
* the flags whose value ``compiler.py`` assigns are exactly the ones the core
  refuses to read from the environment, which is what stops one name from
  meaning "append" in one reader and "replace" in another;
* no module reads a setting under its unprefixed name except ``env_config``;
* every root these rules scan is non-empty. A manifest gate's characteristic
  failure is not a wrong answer but an empty question -- the scan root stops
  matching the tree, the parametrised rule expands to zero cases, and checking
  nothing reports the same green as checking everything.

Run::  python -m pytest tests/structure/build/test_env_var_manifest.py
"""

import ast
import re
import runpy
from pathlib import Path

import pytest

from jittor_utils import env_config, env_manifest


REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "python"


def _flag_policy():
    return runpy.run_path(str(env_manifest.flag_policy_path()))


def _cxx_string_list(name):
    """The C++ ``static const char* <name>[] = { ... }`` initializer, as a set."""
    text = (REPO / "src/utils/log.cc").read_text()
    match = re.search(r"static const char\* " + name + r"\[\] = \{(.*?)\};",
                      text, re.DOTALL)
    assert match, name + " is no longer a string list in log.cc"
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def test_manifest_covers_every_native_flag_and_nothing_else():
    flags = {row["setting"] for row in env_manifest.entries()
             if row["kind"] == "native flag"}
    policy = _flag_policy()
    classified = (policy["STARTUP_FLAGS"] | policy["READONLY_FLAGS"]
                  | policy["RUNTIME_FLAGS"])
    assert flags == set(classified), {
        "in manifest, unclassified": sorted(flags - set(classified)),
        "classified, not in manifest": sorted(set(classified) - flags),
    }


def test_the_two_halves_of_the_build_partition_are_the_same_set():
    # log.cc cannot import flag_policy.py, so it repeats the list. This is the
    # only thing keeping the repetition honest.
    assert _cxx_string_list("startup_flag_names") == set(
        _flag_policy()["STARTUP_FLAGS"])


def test_the_core_refuses_exactly_the_flags_compiler_py_assigns():
    """``flags.cc_path = ...`` and "not readable from the environment" agree.

    This pair *is* the fix for the contradiction 2.22 is named after: while both
    readers existed, ``cc_flags`` in the environment was appended to the
    compiler's flags by ``compiler.py`` and replaced the native flag wholesale in
    ``log.h``, and which one you believed depended on which file you read.
    """
    tree = ast.parse((SOURCE / "jittor/build/compiler.py").read_text())
    assigned = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "flags"):
                assigned.add(target.attr)
    startup = set(_flag_policy()["STARTUP_FLAGS"])
    assert _cxx_string_list("compiler_owned_flag_names") == assigned & startup, {
        "refused by log.cc": sorted(_cxx_string_list("compiler_owned_flag_names")),
        "assigned by compiler.py": sorted(assigned & startup),
    }


def test_every_setting_has_a_namespace_and_a_documented_purpose():
    for row in env_manifest.entries():
        assert row["variable"].startswith(("JT_", "JT_BUILD_")), row
        assert row["namespace"] in ("build", "runtime"), row
        assert row["doc"].strip(), row["variable"] + " has no documented purpose"


def test_a_setting_whose_name_is_an_ordinary_word_has_no_deprecated_form():
    for row in env_manifest.entries():
        if "_" in row["setting"]:
            assert row["deprecated_name"] == row["setting"], row
        else:
            # This is the rule that makes `export debug=1` safe. A new setting
            # named with a single bare word would silently reintroduce the
            # collision, so it has to fail here.
            assert row["deprecated_name"] is None, row


def test_the_manifest_renders():
    table = env_manifest.markdown()
    assert table.startswith("| variable |")
    assert "`JT_BUILD_DEBUG`" in table
    assert "`JT_LOG_V`" in table


#: Files allowed to name an unprefixed setting: the resolver itself and the
#: generator.
_RESOLVER_FILES = {
    "jittor/build/utils/env_config.py",
    "jittor/build/utils/env_manifest.py",
}

#: The one module that cannot go through the resolver, and the exact set of
#: deprecated names it is allowed to know about.
#:
#: It decides whether ``import jittor`` can proceed at all, so it runs before
#: ``jittor_utils`` is importable. It holds the names in constants, which means
#: the regex below cannot see them -- so the allowance is bounded here instead,
#: and a third name appearing in that module fails the gate.
_SHIM_PREFLIGHT = REPO / "compat/shim/preflight.py"
_SHIM_ALLOWED_DEPRECATED = {"tikcc_path", "nvcc_flags"}


def test_the_shim_preflight_exception_does_not_grow():
    tree = ast.parse(_SHIM_PREFLIGHT.read_text())
    declared = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith("_DEPRECATED_")
                and isinstance(node.value, ast.Constant)):
            declared.add(node.value.value)
    assert declared == _SHIM_ALLOWED_DEPRECATED, declared

#: ``os.environ`` reads, as ``.get("x")``, ``["x"]`` or ``"x" in os.environ``.
_ENV_READ = re.compile(
    r"""environ\s*(?:\.get\(|\.pop\(|\[)\s*["']([A-Za-z_][A-Za-z_0-9]*)["']"""
    r"""|["']([A-Za-z_][A-Za-z_0-9]*)["']\s+in\s+(?:os\.)?environ""")


def _unprefixed_settings():
    names = set()
    for row in env_manifest.entries():
        if row["deprecated_name"]:
            names.add(row["deprecated_name"])
        names.add(row["setting"])
    return names


#: Stands in for the module list when the scan root matched nothing, so that the
#: rule below reports one failure instead of expanding to zero cases.
_EMPTY_SCAN = "<no python module matched the scan root>"


def _module_paths(root):
    """Every ``*.py`` under ``root``, relative to it, in name order."""
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.py"))


def _scanned_modules():
    return _module_paths(SOURCE) or [_EMPTY_SCAN]


def test_every_scan_root_matches_the_tree():
    """The roots the rules below read are non-empty, one assertion each.

    This is the failure this whole file is most likely to have, and it fails
    towards green: a root that stops matching the tree -- a directory renamed,
    ``parents[2]`` off by one after a move -- makes ``rglob`` yield nothing, the
    parametrised rule expand to zero cases, and ``native_flags`` return no
    flags. Nothing then reports a problem, because no rule ran.

    Deliberately not written as "more than N settings in total": a total still
    passes when one of two roots has gone empty and the other is large enough to
    carry it. Each root is asserted separately, by a file it has to contain.
    """
    assert SOURCE.is_dir(), SOURCE
    modules = _module_paths(SOURCE)
    assert "jittor/build/utils/env_config.py" in modules, SOURCE
    assert "jittor/build/compiler.py" in modules, SOURCE

    native_root = Path(env_manifest.default_source_root())
    assert native_root.is_dir(), native_root
    assert (native_root / "utils/log.cc").is_file(), native_root
    assert env_manifest.native_flags(str(native_root)), native_root


def test_the_non_empty_rule_would_notice_a_root_that_stopped_matching():
    """The rule above, against a root that does not match the tree.

    Both directions, because "it found nothing" and "there is nothing to find"
    are the two readings of an empty result and only one of them is a bug.
    """
    missing = SOURCE / "no_such_directory"
    assert _module_paths(missing) == []
    assert env_manifest.native_flags(str(missing)) == {}
    # ...and the substitution that keeps the parametrised rule from vanishing.
    assert _module_paths(SOURCE)
    assert _scanned_modules() != [_EMPTY_SCAN]


@pytest.mark.parametrize(
    "path", [path for path in _scanned_modules() if path not in _RESOLVER_FILES]
)
def test_no_module_reads_a_setting_under_its_unprefixed_name(path):
    """Every read goes through ``env_config``, which is what makes it reportable.

    A direct ``os.environ.get("nvcc_path")`` still *works* -- that is the point
    of keeping the old names -- but it is invisible to the startup report and to
    the cache fingerprint, and the second of those is how ``JT_BUILD_CC_FLAGS``
    would compile different object code into an ordinary build's directory.
    """
    assert path != _EMPTY_SCAN, (
        "the scan root %s matched no module; this rule checked nothing"
        % SOURCE)
    settings = _unprefixed_settings()
    found = sorted({(match.group(1) or match.group(2))
                    for match in _ENV_READ.finditer((SOURCE / path).read_text())}
                   & settings)
    assert not found, (
        "%s reads %s directly; use jittor_utils.env_config.build_env / "
        "runtime_env so the value reaches the startup report and the build "
        "fingerprint" % (path, found))


def test_resolver_files_are_the_only_allowlisted_unprefixed_readers():
    scanned = set(_scanned_modules())
    assert _RESOLVER_FILES <= scanned


def test_the_gate_would_notice_an_unprefixed_read():
    # A rule nobody has seen fail is a rule nobody knows works.
    settings = _unprefixed_settings()
    assert "nvcc_path" in settings and "log_v" in settings
    offending = 'value = os.environ.get("nvcc_path")'
    found = {(m.group(1) or m.group(2)) for m in _ENV_READ.finditer(offending)}
    assert found & settings == {"nvcc_path"}


def test_prefixed_name_rejects_an_unknown_namespace():
    with pytest.raises(ValueError):
        env_config.prefixed_name("log_v", "somewhere_else")
