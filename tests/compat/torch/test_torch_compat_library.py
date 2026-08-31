"""Contracts for the dynamically published ``torch.library`` surface."""

from __future__ import annotations

import importlib
import unittest
from typing import get_args, List, Optional, Sequence, Tuple

import jittor as torch


class TestInferSchema(unittest.TestCase):
    def test_unknown_ops_namespace_is_lazy_and_stable(self):
        namespace = torch.ops.jittor_missing_namespace
        self.assertIs(namespace, torch.ops.jittor_missing_namespace)
        self.assertFalse(hasattr(namespace, "missing_op"))

    def test_torch_func_is_an_importable_module(self):
        func = importlib.import_module("torch.func")
        self.assertIs(func, torch.func)
        self.assertIs(func.functional_call, torch.functional_call)

    def test_library_registers_executable_default_overload(self):
        library = torch.library.Library("jittor_test", "DEF")
        self.assertEqual(
            library.define(
                "scale(Tensor value, float factor=2.0) -> Tensor",
                tags=(torch.Tag.pointwise,),
            ),
            "scale",
        )

        def scale(value, factor=2.0):
            return value * factor

        self.assertIsNone(library.impl("scale", scale, dispatch_key="CompositeExplicitAutograd"))
        packet = torch.ops.jittor_test.scale
        self.assertIs(packet.default, packet)
        self.assertEqual(packet.overloads(), ["default"])
        self.assertIsInstance(packet, torch._ops.OpOverload)
        self.assertIsInstance(packet, torch._ops.OpOverloadPacket)
        self.assertEqual(packet._tags, (torch.Tag.pointwise,))
        self.assertEqual(packet(torch.array([3.0]), 4.0).numpy().tolist(), [12.0])

        def fake(value, factor=2.0):
            return value

        self.assertIsNone(library._register_fake("scale", fake))
        self.assertIs(packet._fake_impl, fake)
        self.assertEqual(packet(torch.array([3.0]), 4.0).numpy().tolist(), [12.0])

        def backward(_context, gradient):
            return gradient

        self.assertIsNone(torch.library.register_autograd("jittor_test::scale", backward))
        self.assertIs(packet._backward, backward)

    def test_torch_types_module_replaces_python_types_attribute(self):
        self.assertEqual(torch.types.__name__, "torch.types")
        self.assertEqual(set(get_args(torch.types.Number)), {int, float, bool})
        self.assertEqual(
            set(get_args(torch.types.Device)),
            {torch.device, str, int, type(None)},
        )
        self.assertEqual(
            torch.types.__all__,
            ["Number", "Device", "FileLike", "Storage"],
        )
        self.assertIs(torch.SymInt, int)
        self.assertIs(torch.SymFloat, float)
        self.assertIs(torch.SymBool, bool)
        self.assertEqual(torch.types.py_sym_types, (int, float, bool))

    def test_tag_enum_matches_public_torch_contract(self):
        expected = {
            "core": 0,
            "cudagraph_unsafe": 1,
            "data_dependent_output": 2,
            "dynamic_output_shape": 3,
            "flexible_layout": 4,
            "generated": 5,
            "inplace_view": 6,
            "maybe_aliasing_or_mutating": 7,
            "needs_contiguous_strides": 8,
            "needs_exact_strides": 9,
            "needs_fixed_stride_order": 10,
            "nondeterministic_bitwise": 11,
            "nondeterministic_seeded": 12,
            "pointwise": 13,
            "pt2_compliant_tag": 14,
            "reduction": 15,
            "view_copy": 16,
        }
        self.assertEqual({tag.name: tag.value for tag in torch.Tag}, expected)
        self.assertIs(torch._C.Tag, torch.Tag)
        self.assertIs(
            torch.Tag.needs_fixed_stride_order,
            torch.Tag.needs_fixed_stride_order,
        )

    def test_public_surface_and_basic_schema(self):
        def op(
            value: torch.Tensor, count: int = 2, *, scale: float = 1.5, enabled: bool = True
        ) -> torch.Tensor:
            return value

        self.assertEqual(
            torch.library.infer_schema(op, mutates_args=[]),
            "(Tensor value, SymInt count=2, *, float scale=1.5, bool enabled=True) -> Tensor",
        )
        self.assertEqual(
            torch.library.infer_schema(op, mutates_args=[], op_name="scale"),
            "scale(Tensor value, SymInt count=2, *, float scale=1.5, bool enabled=True) -> Tensor",
        )
        self.assertEqual(
            torch.library.infer_schema.__module__,
            "jittor.compat.torch.library",
        )

    def test_optional_collections_and_tuple_return(self):
        def op(
            value: Optional[torch.Tensor],
            tensors: List[torch.Tensor],  # noqa: UP006 - Python 3.7 eval
            dims: Sequence[int],
        ) -> Tuple[torch.Tensor, int]:  # noqa: UP006 - Python 3.7 eval
            return tensors[0], len(dims)

        self.assertEqual(
            torch.library.infer_schema(op, mutates_args=[]),
            "(Tensor? value, Tensor[] tensors, SymInt[] dims) -> (Tensor, SymInt)",
        )

    def test_mutation_aliases_match_argument_positions(self):
        def op(
            value: torch.Tensor,
            other: torch.Tensor,
            tensors: List[torch.Tensor],  # noqa: UP006 - Python 3.7 eval
        ) -> torch.Tensor:
            return value

        self.assertEqual(
            torch.library.infer_schema(op, mutates_args=["other", "tensors"]),
            "(Tensor value, Tensor(a1!) other, Tensor(a2!)[] tensors) -> Tensor",
        )
        self.assertEqual(
            torch.library.infer_schema(op, mutates_args="unknown"),
            "(Tensor(a0!) value, Tensor(a1!) other, Tensor(a2!)[] tensors) -> Tensor",
        )

    def test_defaults_for_dtype_device_and_string(self):
        def op(
            value: torch.Tensor,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cpu"),
            label: str = "x",
        ) -> torch.Tensor:
            return value

        self.assertEqual(
            torch.library.infer_schema(op, mutates_args=[]),
            '(Tensor value, ScalarType dtype=float32, Device device="cpu", '
            'str label="x") -> Tensor',
        )

    def test_invalid_contracts_fail_closed(self):
        def missing(value) -> torch.Tensor:
            return value

        def bad_mutation(count: int) -> int:
            return count

        def variadic(*values: torch.Tensor) -> torch.Tensor:
            return values[0]

        with self.assertRaisesRegex(ValueError, "must have a type annotation"):
            torch.library.infer_schema(missing, mutates_args=[])
        with self.assertRaisesRegex(ValueError, "only Tensors"):
            torch.library.infer_schema(bad_mutation, mutates_args=["count"])
        with self.assertRaisesRegex(ValueError, "were not found"):
            torch.library.infer_schema(bad_mutation, mutates_args=["missing"])
        with self.assertRaisesRegex(ValueError, "varargs"):
            torch.library.infer_schema(variadic, mutates_args=[])
        with self.assertRaisesRegex(ValueError, "sequence of argument names"):
            torch.library.infer_schema(bad_mutation, mutates_args="all")


if __name__ == "__main__":
    unittest.main()
