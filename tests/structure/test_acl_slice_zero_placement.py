"""Execute slice-cache allocation routing on host, without Jittor/JIT.

These doubles test placement/cache identity, not kernel/device correctness.
The separate tensor regression exercises real forward and backward execution.
"""

import ast
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]


class Tensor:
    def __init__(self, shape, backend="acl", device_id=0, dtype="float32"):
        self.shape, self.ndim = tuple(shape), len(shape)
        self.backend, self.device_id, self.dtype = backend, device_id, dtype

    def stop_grad(self):
        return self


class TestAclSliceZeroPlacement(unittest.TestCase):
    def setUp(self):
        self.placement = ("cpu", 0)
        self.allocations, self.calls, self.programs = [], [], []
        self.raise_allocation = False

        @contextmanager
        def scope(tensor):
            # The real *_scope_like is a DEFAULT: it must preserve an already
            # active CPU placement requested by tensor_frontend. Treating it
            # as a force-override would hide the exact FSDP regression.
            yield

        def set_placement(backend, device):
            self.assertEqual(backend, 2)  # Native BackendId::Acl.
            previous = self.placement
            self.placement = ("acl", device)
            return previous

        def reset_placement(previous):
            self.placement = previous

        def zeros(shape, dtype):
            if self.raise_allocation:
                raise RuntimeError("synthetic allocation failure")
            value = Tensor(shape, *self.placement, dtype=dtype)
            self.allocations.append(value)
            return value

        def program(*args, **kwargs):
            value = object()
            self.programs.append(value)
            return value

        def forward(program, inputs, output_dtypes, output_shapes):
            self.calls.append(inputs)
            self.assertTrue(all((value.backend, value.device_id) == (inputs[0].backend, inputs[0].device_id)
                                for value in inputs), "CodeOp received mixed CPU/device or cross-device inputs")
            return [Tensor(output_shapes[0], inputs[0].backend, inputs[0].device_id)]

        path = ROOT / "backends/acl/kernels/ops/getitem_op.py"
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("_cached_slice_zero", "basic_slice_acl", "_build_slice_program")]
        self.namespace = {
            "jt": SimpleNamespace(flags=SimpleNamespace(device_id=0), zeros=zeros,
                                  core=SimpleNamespace(_set_tensor_placement=set_placement,
                                                       _reset_tensor_placement=reset_placement)),
            "_slice_zero_cache": OrderedDict(), "_jittor_dtype_name": str,
            "_SLICE_PROGRAMS": {}, "_SLICE_PROGRAM_LIMIT": 4096,
            "canonical_dtype_name": str, "acl_program": program,
            "placement_scope_like": scope, "device_scope_like": scope,
            "dispatch_context": lambda tensor: SimpleNamespace(backend=tensor.backend, device_id=tensor.device_id),
            "code_program": lambda value: value, "attribute_program": lambda *a, **k: None,
            "acl_emit": forward,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), self.namespace)
        self.slice = self.namespace["basic_slice_acl"]

    def test_explicit_device_slice_with_default_cpu_factories(self):
        self.slice(Tensor((2, 8), "acl", 0), (slice(None), slice(1, 7)))
        self.assertEqual(self.placement, ("cpu", 0))
        self.assertEqual(len(self.allocations), 1)  # Same before/after padding shares one zero.
        self.assertIs(self.calls[-1][1], self.calls[-1][2])

    def test_cache_identity_includes_actual_backend_and_device(self):
        for backend, device in (("acl", 0), ("acl", 1), ("acl", 0)):
            self.slice(Tensor((2, 8), backend, device), (slice(None), slice(1, 7)))
        self.assertEqual(len(self.allocations), 2)
        self.assertIs(self.calls[0][1], self.calls[-1][1])
        self.assertIsNot(self.calls[0][1], self.calls[1][1])
        self.assertEqual(self.placement, ("cpu", 0))

    def test_non_acl_input_is_rejected_without_allocation(self):
        for backend in ("cpu", "cuda"):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "ACL"):
                self.slice(Tensor((2, 8), backend), (slice(None), slice(1, 7)))
        self.assertEqual(self.allocations, [])
        self.assertEqual(self.placement, ("cpu", 0))

    def test_allocation_failure_restores_caller_placement_and_does_not_cache(self):
        self.raise_allocation = True
        with self.assertRaisesRegex(RuntimeError, "synthetic allocation failure"):
            self.slice(Tensor((2, 8), "acl", 1), (slice(None), slice(1, 7)))
        self.assertEqual(self.placement, ("cpu", 0))
        self.assertEqual(len(self.namespace["_slice_zero_cache"]), 0)

    def test_shape_dtype_separation_and_lru_bound(self):
        for shape, dtype in (((2, 8), "float32"), ((3, 8), "float32"), ((2, 8), "float16")):
            self.slice(Tensor(shape, "acl", 0, dtype), (slice(None), slice(1, 7)))
        self.assertEqual(len(self.allocations), 3)
        cache = self.namespace["_slice_zero_cache"]
        self.assertIn(((2, 1), "float32"), {(key[1], key[2]) for key in cache})
        for width in range(1, 36):
            self.slice(Tensor((2, 40)), (slice(None), slice(width, 40)))
        self.assertEqual(len(cache), 32)
        self.assertEqual(self.placement, ("cpu", 0))

    def test_program_cache_reuses_metadata_but_not_cross_device_zero_storage(self):
        for device in (0, 1, 0):
            self.slice(Tensor((2, 8), "acl", device), (slice(None), slice(1, 7)))
        self.assertEqual(len(self.programs), 1)
        self.assertEqual(len(self.allocations), 2)
        self.assertEqual(len(self.namespace["_SLICE_PROGRAMS"]), 1)
        self.assertIs(self.calls[0][1], self.calls[2][1])
        self.assertIsNot(self.calls[0][1], self.calls[1][1])
        self.assertEqual(self.placement, ("cpu", 0))


if __name__ == "__main__":
    unittest.main()
