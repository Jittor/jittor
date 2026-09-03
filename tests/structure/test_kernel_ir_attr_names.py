"""KernelIR attribute names are declared, not spelled out at the call site.

``KernelIR::attrs`` is a ``map<string,string>`` and ``get_attr`` returns
``attrs[s]``, so a key reached with a string literal has no spelling check
anywhere: a typo inserts an empty value and reads back as "this node does not
have that attribute".  Nothing fails; the pass that asked just quietly does
nothing.  The names live in ``namespace kir`` in ``opt/kernel_ir.h``, where a
typo is a compile error -- this test is what stops literals coming back.
"""
import re
import unittest
from pathlib import Path

import jittor


_SRC = Path(jittor.__file__).resolve().parent / "src"
_KERNEL_IR_H = _SRC / "opt" / "kernel_ir.h"

#: where a KernelIR attribute name can appear
_LITERAL = re.compile(
    r'(?:attrs\s*(?:\.at|\.count|\.find)?\s*[\[(]\s*|'
    r'\b(?:get_attr|has_attr|check_attr)\s*\(\s*)"(\w+)"')

#: the files that reach into KernelIR::attrs
_SCANNED = ["opt", "test", "ops/getitem_op.cc"]


def _declared_names():
    text = _KERNEL_IR_H.read_text(encoding="utf8")
    body = text[text.index("namespace kir {"):text.index("} // kir")]
    return set(re.findall(r'constexpr const char\* (\w+) = "(?:\w+)";', body))


def _scanned_files():
    files = []
    for entry in _SCANNED:
        path = _SRC / entry
        if path.is_dir():
            files += sorted(path.rglob("*.cc")) + sorted(path.rglob("*.h"))
        elif path.exists():
            files.append(path)
    return files


class TestKernelIRAttrNames(unittest.TestCase):
    def test_names_are_declared(self):
        declared = _declared_names()
        assert len(declared) >= 10, declared
        # the ones a reader will look for first
        for name in ("lvalue", "rvalue", "code", "dtype", "loop_id"):
            assert name in declared, declared

    def test_no_attribute_name_is_a_string_literal(self):
        violations = []
        for path in _scanned_files():
            text = path.read_text(encoding="utf8")
            for match in _LITERAL.finditer(text):
                line = text[:match.start()].count("\n") + 1
                violations.append(
                    "{}:{} uses \"{}\" instead of kir::{}".format(
                        path.relative_to(_SRC), line,
                        match.group(1), match.group(1)))
        assert not violations, (
            "KernelIR attribute names must come from namespace kir in "
            "opt/kernel_ir.h, so that a typo is a compile error:\n  " +
            "\n  ".join(violations))


if __name__ == "__main__":
    unittest.main()
