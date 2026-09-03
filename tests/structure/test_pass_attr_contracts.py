"""Every pass declares which KernelIR attributes it reads and writes.

The contract between the passes was 14 string literals spread over 13 pass
files, documented only by a comment in ``kernel_ir.h``: nothing said which pass
has to run before which for an attribute to be there, and nothing checked it.
Each pass now declares ``reads`` and ``writes`` in its constructor, and
``PassManager::check_attr_contract`` walks the pipeline refusing a pass that
reads something nothing before it produces.

A declaration that drifts from the code is worse than none, so this test checks
the direction that can go stale on its own: every ``kir::x`` a pass's .cc
mentions must appear in that pass's declaration.  (The other direction is
deliberately loose -- a pass may touch an attribute through a KernelIR helper,
as RenameLoopIndexPass does with ``loop_id``.)
"""
import re
import unittest
from pathlib import Path

import jittor


_OPT_DIR = Path(jittor.__file__).resolve().parent / "src" / "opt"
_PASS_DIR = _OPT_DIR / "pass"
_PASS_MANAGER = _OPT_DIR / "pass_manager.cc"

#: attributes the KernelIR parser sets, so they are there before any pass runs;
#: kept in step with `parsed_attrs` in pass_manager.cc
_PARSED = {"lvalue", "rvalue", "code", "dtype", "loop_id", "raw",
           "void_discard", "has_bc", "used"}

_DECL = re.compile(r"\b(reads|writes)\s*=\s*\{([^}]*)\}")
_RUN_PASS = re.compile(r"^\s*run_pass<(\w+)>\(", re.M)
_CLASS = re.compile(r"struct (\w+) : Pass \{")
_USE = re.compile(r"\bkir::(\w+)")


def _declared(header):
    text = header.read_text(encoding="utf8")
    names = set()
    for _, body in _DECL.findall(text):
        names.update(_USE.findall(body))
    return names


def _contracts():
    """{pass class name: (reads, writes)} from the pass headers."""
    out = {}
    for header in sorted(_PASS_DIR.glob("*_pass.h")):
        text = header.read_text(encoding="utf8")
        klass = _CLASS.search(text)
        if not klass:
            continue
        sets = {"reads": set(), "writes": set()}
        for kind, body in _DECL.findall(text):
            sets[kind].update(_USE.findall(body))
        out[klass.group(1)] = (sets["reads"], sets["writes"])
    return out


class TestPassAttrContracts(unittest.TestCase):
    def test_every_pass_declares_a_contract(self):
        headers = sorted(p for p in _PASS_DIR.glob("*_pass.h"))
        assert len(headers) >= 25, headers
        missing = [h.name for h in headers
                   if "reads = {" not in h.read_text(encoding="utf8")]
        assert not missing, (
            "these passes declare no attribute contract (Pass::reads/writes): "
            + ", ".join(missing))

    def test_declarations_cover_what_the_pass_touches(self):
        violations = []
        for source in sorted(_PASS_DIR.glob("*_pass.cc")):
            header = source.with_suffix(".h")
            if not header.exists():
                continue
            declared = _declared(header)
            used = set(_USE.findall(source.read_text(encoding="utf8")))
            for name in sorted(used - declared):
                violations.append(
                    "{} uses kir::{} but {} does not declare it".format(
                        source.name, name, header.name))
        assert not violations, "\n  ".join([""] + violations)


    def test_the_declared_pipeline_order_satisfies_the_contracts(self):
        """The same walk PassManager::check_attr_contract does, over the source.

        Reading an attribute a pass also writes is that pass checking its own
        marker, not a dependency.
        """
        contracts = _contracts()
        order = _RUN_PASS.findall(_PASS_MANAGER.read_text(encoding="utf8"))
        assert len(order) >= 25, order
        produced = set(_PARSED)
        violations = []
        for name in order:
            assert name in contracts, "%s has no contract" % name
            reads, writes = contracts[name]
            for attr in sorted(reads - produced - writes):
                violations.append(
                    "%s reads kir::%s but nothing before it in "
                    "PassManager::run_passes writes it" % (name, attr))
            produced |= writes
        assert not violations, "\n  ".join([""] + violations)


if __name__ == "__main__":
    unittest.main()
