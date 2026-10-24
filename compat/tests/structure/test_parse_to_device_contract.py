"""Execute production parser/device routing without importing Jittor or JIT.

The device class and integer validation are the real source definitions. Only
the backend-presence flag and dtype/tensor identities are host stand-ins.
"""
import ast
import enum
from pathlib import Path
import types
import typing
import unittest


ROOT = Path(__file__).resolve().parents[3]


class _Dtype:
    _registry = {}

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "torch." + self.name


class _Ordinal(enum.IntEnum):
    ZERO = 0
    TWO = 2


class TestParseToDeviceContract(unittest.TestCase):
    def load(self, has_acl):
        jt = types.SimpleNamespace(Var=type("Tensor", (), {}),
                                   compiler=types.SimpleNamespace(has_acl=has_acl))
        namespace = dict(jt=jt, dtype=_Dtype, typing=typing,
                         kind_of=lambda value: type(value).__name__,
                         EXPECTED=(AttributeError,))
        tree = ast.parse((ROOT / "compat/torch/types.py").read_text())
        names = {"device", "_is_index", "_check_index", "_check_type", "_accelerator_type"}
        tree.body = [node for node in tree.body
                     if (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names)
                     or (isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id == "_DEVICE_TYPES"
                         for target in node.targets))]
        exec(compile(tree, "device_contract", "exec"), namespace)
        tree = ast.parse((ROOT / "compat/torch/installers/cuda/api.py").read_text())
        tree.body = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "_parse_to"]
        exec(compile(tree, "parse_to_contract", "exec"), namespace)
        return namespace["_parse_to"], namespace["device"]

    def test_integer_routes_to_actual_configured_accelerator(self):
        for has_acl, kind in ((False, "cuda"), (True, "npu")):
            parse, _ = self.load(has_acl)
            for value in (0, 2, _Ordinal.ZERO, _Ordinal.TWO):
                for positional in (True, False):
                    with self.subTest(acl=has_acl, value=value, positional=positional):
                        result = parse(value) if positional else parse(device=value)
                        self.assertEqual((result[0].type, result[0].index), (kind, int(value)))
                        self.assertEqual(result[1:], (None, False, None))

    def test_negative_and_boolean_keyword_devices_fail_closed(self):
        parse, _ = self.load(True)
        for value in (-1, -3):
            with self.assertRaisesRegex(RuntimeError, "negative"):
                parse(value)
            with self.assertRaisesRegex(RuntimeError, "negative"):
                parse(device=value)
        for value in (False, True):
            with self.assertRaises(TypeError):
                parse(device=value)

    def test_existing_strings_dtype_and_keyword_flags_are_unchanged(self):
        parse, device = self.load(True)
        dtype = _Dtype("float32")
        for argument in ("cpu", device("cpu")):
            target, actual_dtype, non_blocking, memory_format = parse(
                argument, dtype, non_blocking=True, memory_format="fixture-format")
            self.assertEqual((target.type, target.index), ("cpu", None))
            self.assertIs(actual_dtype, dtype)
            self.assertIs(non_blocking, True)
            self.assertEqual(memory_format, "fixture-format")
        self.assertEqual(parse(dtype), (None, dtype, False, None))


if __name__ == "__main__":
    unittest.main()
