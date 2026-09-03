# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``Node::custom_data`` has exactly one owner, and the rule is a test.

It used to have six.  ``Executor::run_sync`` kept each op's and var's index in
it, ``count_fuse`` read those, ``grad()`` kept gradient-var indices there, the
two topological sorts used it for in-degrees, ``dump_all_graphs`` numbered nodes
with it, and ``FusedOp::update_ops`` bit-packed a var index, a "visited" bit and
the caller's "cannot fuse" bit into the same 32 bits.  Nothing recorded whose
turn it was, so correctness was a scheduling convention -- and where the
convention could not be kept it was patched by hand:
``MemoryProfiler::check()``, which runs from *inside* ``run_sync``'s op loop,
copied the whole field out and put it back around its own traversal.

Delete those six lines and a fused graph dies with
``Check failed: outputs().size()`` in ``FusedOp::update_ops``: the "cannot fuse"
bit the executor left behind comes back as an in-degree.  That is the shape of
the problem -- the failure lands in a third file, long after the two traversals
that collided.

Five of the six are gone (``[2.02]``): each keeps its own storage, either a
local table (``misc/node_index.h``) or ``Node::batch_index``, which is stamped
with the batch that wrote it and read through ``batch_index_at(stamp)`` so a
reader that names the wrong batch gets an assertion instead of somebody else's
number.

The sixth stays, and is why this file exists rather than a simple "the field is
gone" assertion.  ``FusedOp``'s packing is not a traversal marker: it is a
mapping that has to stay valid across the whole JIT pipeline, so removing it
means giving ``FusedOp`` an explicit map and changing three readers, one of them
tied to the generated code's struct offsets -- task ``2.24``, sequenced after
``3.11`` because it lands in the same code.  Until then the field is owned, and
*owned* has to mean something a person cannot quietly undo.  A comment is what
let the previous five in.

Run::  JITTOR_TORCH_SHIM=1 python -m pytest tests/structure/test_node_scratch_state.py
"""

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "python" / "jittor" / "src"

#: The only places allowed to name ``custom_data``, and why.
#:
#: All three are the one owner: ``FusedOp::update_ops`` builds the packing,
#: ``load_fused_op`` reads it back while turning a fusion group into a FusedOp,
#: and the op-relay unit test drives both directly.
CUSTOM_DATA_OWNERS = {
    "python/jittor/src/node.h": "declares it, and names its owner",
    "python/jittor/src/fused_op.cc": "FusedOp::update_ops builds the packing",
    "python/jittor/src/executor.cc": "load_fused_op reads the packing back",
    "python/jittor/src/test/test_op_relay.cc": "drives FusedOp directly",
}


def _sources():
    for path in sorted(SRC.rglob("*")):
        if path.suffix in (".cc", ".h"):
            yield path


def _code_lines(path):
    """Lines with comments and string literals removed.

    Prose about the rule mentions the names the rule is about, and so does the
    error message the rule's own guard prints, so a check that only strips
    ``//`` reports its own documentation.
    """
    in_block = False
    for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        code = line
        if in_block:
            end = code.find("*/")
            if end < 0:
                continue
            code = code[end+2:]
            in_block = False
        while True:
            start = code.find("/*")
            if start < 0:
                break
            end = code.find("*/", start+2)
            if end < 0:
                code = code[:start]
                in_block = True
                break
            code = code[:start] + code[end+2:]
        code = code.split("//", 1)[0]
        # ... and string literals, which is where the diagnostics that name
        # these fields live.
        yield number, re.sub(r'"(?:[^"\\]|\\.)*"', '""', code)


def test_custom_data_is_named_as_belonging_to_one_owner():
    node_h = (SRC / "node.h").read_text(encoding="utf-8")
    struct = node_h[node_h.index("struct Node {"):]
    declaration = struct[:struct.index("int custom_data;")]
    tail = declaration[declaration.rindex("\n\n"):]
    assert "FUSED-OP ONLY" in tail, (
        "the declaration of Node::custom_data no longer says who owns it. It "
        "is one int with one owner and five former ones; a reader who cannot "
        "see that from the declaration is the way the other five got in.")
    assert "2.24" in tail, (
        "the declaration should point at the task that removes it, so the "
        "field's continued existence stays a decision rather than an oversight.")


def test_only_the_fused_op_pipeline_touches_custom_data():
    offenders = []
    for path in _sources():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in CUSTOM_DATA_OWNERS:
            continue
        for number, code in _code_lines(path):
            if "custom_data" in code:
                offenders.append(
                    "%s:%d uses Node::custom_data. It belongs to FusedOp's "
                    "cross-stage var index; a traversal that borrows it will "
                    "corrupt a fusion decision, and the failure will surface "
                    "somewhere else entirely." % (relative, number))
    assert offenders == [], "\n".join(offenders)


def test_the_executor_batch_index_is_stamped_and_read_checked():
    """It may live on the node, but not as a bare int anybody can read."""
    node_h = (SRC / "node.h").read_text(encoding="utf-8")
    struct = node_h[node_h.index("struct Node {"):]
    for required in ("int64 batch_stamp", "set_batch_index", "batch_index_at"):
        assert required in struct, (
            "Node no longer offers %s. The executor's batch numbering is "
            "allowed to live on the node only because it carries the stamp of "
            "the batch that wrote it and every read names that batch."
            % required)

    offenders = []
    for path in _sources():
        if path.name == "node.h":
            continue
        for number, code in _code_lines(path):
            if "set_batch_index" in code:
                continue
            if re.search(r"(?<![\w:])batch_index\b(?!_at)", code):
                offenders.append(
                    "%s:%d reads Node::batch_index without naming a batch; use "
                    "batch_index_at(stamp)"
                    % (path.relative_to(REPO_ROOT).as_posix(), number))
    assert offenders == [], "\n".join(offenders)


def test_no_traversal_hand_saves_node_state_around_another():
    """The workaround is the tell, not the bug.

    A call site that copies a node field out and back around a call it makes is
    saying the callee will overwrite state the caller still needs. That is the
    collision this rule removes, so the workaround must not come back either --
    silently, on a field somebody adds later.
    """
    offenders = []
    for path in _sources():
        for number, code in _code_lines(path):
            if re.search(r"\bbackup_\w*(custom_data|node|flag)\w*\b", code):
                offenders.append(
                    "%s:%d saves per-node state around a call; the callee "
                    "should keep its own table instead"
                    % (path.relative_to(REPO_ROOT).as_posix(), number))
    assert offenders == [], "\n".join(offenders)
