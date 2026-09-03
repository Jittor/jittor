# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A node's flag word is a union, and the union has to be declared, not assumed.

``Node::flags`` is one 32-bit word.  Bits 0..5 mean the same thing on every
node; everything above that means one thing on a ``Var`` and a different thing
on an ``Op``.  That is a reasonable layout.  What was not reasonable is how it
was written down: both meanings lived in a single enum of hand-picked numbers,
with the disjointness recorded in comments ("bit 23 is free in both layouts",
"bits 6..22 are shared").  A comment cannot fail.

It had already failed once: ``_th_require_grad`` and ``_is_scalar`` were both
bit 11, so every ``requires_grad_(True)`` parameter read back as a Python
scalar -- it dropped out of dtype promotion and mixed precision skipped every
operator that touched it.  That was fixed by moving one of them to a number
somebody checked by hand, which is the same mechanism that produced the bug.

So the rules below are mechanical:

* the kind-private bits are *generated* by their enums, never written as
  numbers, so inserting one cannot land on another;
* the shared bits that have to sit above both private layouts are computed
  from those layouts rather than picked;
* the invariants that used to be comments are ``static_assert``\\ s;
* nothing outside ``node.h`` can name a kind-private bit through
  ``NodeFlags::``.  The compiler already enforces this -- ``Node`` has no
  ``flag()`` -- but the raw word (``flags.flags``) is still reachable, and this
  test is what keeps that escape hatch down to the two places that need it.

The last rule is what the type change bought: a kind-private flag now takes a
``Var*`` or an ``Op*`` to read, so code holding a bare ``Node*`` has to ask
``is_var()`` first.  Enforcing that turned up a real one in ``executor.cc``,
which read the Op-only ``_has_gopt`` off nodes that may be vars and then called
``n->op()->graph_optimize()`` on whatever answered yes; it was safe only
because the bit number ``_has_gopt`` happened to occupy was unused in the Var
layout.

Run::  JITTOR_TORCH_SHIM=1 python -m pytest tests/structure/test_node_flag_kinds.py
"""

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_H = REPO_ROOT / "python" / "jittor" / "src" / "node.h"

#: Where C++ that names these flags lives.
#:
#: ``tests/`` is in the list because **C++ source is not only in .cc files**.
#: 54 Python files in this repository embed ``cuda_src``/``cpu_src``/``cpu_header``
#: strings -- about 15k lines of C++ that the JIT compiles at run time and that
#: a grep for ``NodeFlags::`` over ``*.cc`` does not see. ``[2.01]`` renamed the
#: Op-only bits and missed three such strings (``tests/backends/cuda/test_cuda.py``
#: twice, ``tests/backends/rocm/test_rocm.py`` once); they only surfaced as
#: ``'_cpu' is not a member of 'jittor::NodeFlags'`` when those tests ran.
SOURCE_ROOTS = (REPO_ROOT / "python" / "jittor", REPO_ROOT / "tests")

#: Files that quote the *old* spelling on purpose, to explain the rule.
QUOTES_THE_OLD_SPELLING = {
    "tests/structure/test_node_flag_kinds.py",
    "tests/core/test_node_flag_layout.py",
}

#: ``flags.flags`` hands out the whole word with no kind attached, so every use
#: is an escape from the rule this file exists to keep. These two are allowed
#: because neither reads a *bit*: one hands the opaque word to Python, the other
#: writes it back verbatim into generated code.
RAW_WORD_ALLOWED = {
    "python/jittor/src/var_holder.h": "exposes the whole word to Python as an opaque int",
    "python/jittor/src/opt/pass/fake_main_pass.cc": "round-trips the whole word into generated source",
}


def _enum_body(source, struct_name):
    """The text between ``struct <name> {`` and its closing brace."""
    start = source.index("struct %s {" % struct_name)
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i]
    raise AssertionError("unterminated struct %s" % struct_name)


def _enumerators(body):
    """``(name, initialiser or None)`` for each enumerator in an enum block.

    Comments come out *before* the split on commas, not after: the prose in
    this header contains commas, and stripping second turns every comment into
    a pile of nameless enumerators.
    """
    enum = body[body.index("enum Flags {"):]
    enum = enum[:enum.index("}")]
    enum = enum.split("{", 1)[1]
    enum = "\n".join(re.sub(r"//.*$", "", line) for line in enum.splitlines())
    found = []
    for entry in enum.split(","):
        entry = " ".join(entry.split())
        if not entry:
            continue
        name, _, value = entry.partition("=")
        found.append((name.strip(), value.strip() or None))
    return found


def test_kind_private_bits_are_generated_not_numbered():
    """Only the first enumerator names where a private layout starts."""
    source = NODE_H.read_text(encoding="utf-8")
    offenders = []
    for struct_name, first in (("VarFlags", "_force_fuse"), ("OpFlags", "_cpu")):
        entries = _enumerators(_enum_body(source, struct_name))
        assert entries, "%s has no enumerators" % struct_name
        assert entries[0][0] == first, (
            "%s should start at %s, starts at %s" % (struct_name, first, entries[0][0]))
        assert entries[0][1] == "NodeBits::_kind_private", (
            "%s must start where the shared range ends, not at %r"
            % (struct_name, entries[0][1]))
        assert entries[-1][0] == "_end", (
            "%s must end with _end so the layout has a computable size" % struct_name)
        for name, value in entries[1:]:
            if value is not None:
                offenders.append(
                    "%s::%s is written as %r; let the enum generate it, "
                    "otherwise inserting a flag can land on another one"
                    % (struct_name, name, value))
    assert offenders == [], "\n".join(offenders)


def test_shared_high_bits_are_computed_from_both_layouts():
    """``_lived_tracked`` sits above whichever private layout is taller."""
    source = NODE_H.read_text(encoding="utf-8")
    entries = dict(_enumerators(_enum_body(source, "NodeFlags")))
    shared_high = entries.get("_shared_high")
    assert shared_high is not None, "NodeFlags no longer derives _shared_high"
    assert "VarFlags::_end" in shared_high and "OpFlags::_end" in shared_high, (
        "_shared_high must be computed from both private layouts, not written "
        "as %r -- 'bit 28 is free in both layouts' was a comment and comments "
        "do not fail" % shared_high)
    assert entries.get("_lived_tracked") == "_shared_high", (
        "a shared bit must be placed at _shared_high, got %r"
        % entries.get("_lived_tracked"))


def test_the_layout_invariants_are_static_asserts():
    source = NODE_H.read_text(encoding="utf-8")
    required = [
        "a shared flag bit aliases a Var-only bit",
        "a shared flag bit aliases an Op-only bit",
        "the private layouts must both start where the shared range ends",
        "the six amp bits op.cc and grad.cc move as one field are no longer contiguous",
        "OpType is read as two bits and they are no longer adjacent",
        "node_order is read as two bits and they are no longer adjacent",
    ]
    missing = [text for text in required if text not in source]
    assert missing == [], (
        "these invariants are not checked at compile time:\n  " + "\n  ".join(missing))


def _shared_flag_names():
    """The bits ``NodeFlags::`` is allowed to name, read out of node.h itself.

    Derived rather than listed, and derived by *subtracting* the two private
    enums: a name that appears in both lists (``_requires_grad_disabled`` does,
    with unrelated meanings) is private by definition. Reading ``NodeBits``
    also makes this fail outright on a layout that has no kind split, which is
    the state this test exists to rule out.
    """
    source = NODE_H.read_text(encoding="utf-8")
    shared = {name for name, _ in _enumerators(_enum_body(source, "NodeBits"))}
    shared |= {name for name, _ in _enumerators(_enum_body(source, "NodeFlags"))}
    private = {name for name, _ in _enumerators(_enum_body(source, "VarFlags"))}
    private |= {name for name, _ in _enumerators(_enum_body(source, "OpFlags"))}
    return shared - private


def _scanned_files():
    for root in SOURCE_ROOTS:
        for path in sorted(root.rglob("*")):
            if path.suffix not in (".cc", ".h", ".py") or path == NODE_H:
                continue
            if path.relative_to(REPO_ROOT).as_posix() in QUOTES_THE_OLD_SPELLING:
                continue
            yield path


def test_no_source_names_a_kind_private_bit_through_nodeflags():
    """Including the C++ that lives inside Python strings.

    A ``cuda_src`` string is compiled by the JIT exactly like a file in
    ``src/``, and it is invisible to a grep over ``*.cc``. That is how
    ``[2.01]`` left three of them behind, and how a rename of a core C++ name
    will do it again unless the rule looks where the source actually is.
    """
    shared = _shared_flag_names()
    offenders = []
    for path in _scanned_files():
        for number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for name in re.findall(r"NodeFlags::(_[a-z_0-9]+)", line):
                if name in shared:
                    continue
                offenders.append(
                    "%s:%d names NodeFlags::%s; that bit belongs to VarFlags or "
                    "OpFlags and has to be read through a Var* or an Op*"
                    % (path.relative_to(REPO_ROOT).as_posix(), number, name))
    assert offenders == [], "\n".join(offenders)


def test_the_raw_flag_word_stays_out_of_reach():
    offenders = []
    for path in _scanned_files():
        if path.suffix == ".py":
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in RAW_WORD_ALLOWED:
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if re.search(r"\bflags\s*\.\s*flags\b", line):
                offenders.append(
                    "%s:%d reaches the raw flag word; a bit read off it carries "
                    "no kind, which is the mistake this layout is arranged to "
                    "prevent" % (relative, number))
    assert offenders == [], "\n".join(offenders)
