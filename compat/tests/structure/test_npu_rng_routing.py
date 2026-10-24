"""NPU RNG publication/routing only; host doubles prove no device behavior."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "compat/torch/installers/cuda"


class Tensor:
    def __init__(self, data, dtype="uint8", device="cpu"):
        self.data = np.asarray(data, dtype=dtype)
        self.dtype, self.ndim = dtype, self.data.ndim
        self.device = SimpleNamespace(type=device)

    def numpy(self):
        return self.data.copy()


class TestNpuRngRouting(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.blobs = ["JITTOR_ACL_RNG_V1\nnumel_v1\n7\n0", "JITTOR_ACL_RNG_V1\nnumel_v1\n8\n15"]
        def call(name, result=None):
            def function(*args):
                self.calls.append((name, args))
                return result
            return function
        native = SimpleNamespace(get_rng_state=call("get", self.blobs[1]),
            set_rng_state=call("set"), get_rng_state_all=call("get_all", self.blobs),
            set_rng_state_all=call("set_all"), manual_seed=call("seed"),
            manual_seed_all=call("seed_all"), initial_seed=call("initial", 8))
        self.torch = SimpleNamespace(Tensor=Tensor, uint8="uint8",
            tensor=lambda values, dtype, device: Tensor(values, dtype, device))
        def index(value=None):
            if value is None or value == "npu:1":
                return 1
            if type(value) is not int or value not in (0, 1):
                raise ValueError("invalid NPU device")
            return value
        self.api = {"_native_rng": lambda: native, "_device_index": index,
                    "_torch": lambda: self.torch, "_dtype_to_str": str}
        self.npu = SimpleNamespace()
        path = DIRECTORY / "npu_random.py"
        if path.exists():
            tree = ast.parse(path.read_text())
            tree.body = [node for node in tree.body
                         if not isinstance(node, ast.ImportFrom)
                         and not (isinstance(node, ast.Import) and any(a.name == "jittor" for a in node.names))
                         and not (isinstance(node, ast.FunctionDef) and node.name in {"_native_rng", "_torch"})]
            exec(compile(tree, str(path), "exec"), self.api)
            self.modules = {}
            self.api["install"](self.npu, self.modules)

    def test_get_cpu_byte_tensor_from_native_exact_state(self):
        result = self.npu.get_rng_state("npu:1")
        self.assertEqual(result.numpy().tobytes(), self.blobs[1].encode("ascii"))
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(self.calls, [("get", (1,))])

    def test_set_all_batches_single_native_transaction(self):
        states = [Tensor(list(value.encode("ascii"))) for value in self.blobs]
        self.npu.set_rng_state_all(iter(states))
        self.assertEqual(self.calls, [("set_all", (self.blobs,))])

    def test_invalid_second_state_never_partially_restores(self):
        states = [Tensor(list(self.blobs[0].encode("ascii"))), Tensor([7], "int64")]
        with self.assertRaises(TypeError):
            self.npu.set_rng_state_all(states)
        self.assertEqual(self.calls, [])

    def test_wrong_shape_device_type_and_nonascii_rejected(self):
        for value in ([7], Tensor([[7]]), Tensor([7], device="npu"), Tensor([255])):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    self.npu.set_rng_state(value)
                self.assertEqual(self.calls, [])

    def test_seed_routes_without_global_or_cuda_seed(self):
        self.npu.manual_seed(17)
        self.npu.manual_seed_all(23)
        self.assertEqual(self.npu.initial_seed(), 8)
        self.assertEqual(self.calls, [("seed", (1, 17)), ("seed_all", (23,)), ("initial", (1,))])

    def test_seed_no_silent_truncation(self):
        for value in (True, 1.5, Tensor([1.5], "float32"), -1, 1 << 63, 1 << 80):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError, OverflowError)):
                    self.npu.manual_seed(value)
                self.assertEqual(self.calls, [])

    def test_full_supported_signed_positive_range_is_forwarded_exactly(self):
        seed = (1 << 63) - 1
        self.npu.manual_seed(seed)
        self.assertEqual(self.calls, [("seed", (1, seed))])

    def test_installer_is_wired_into_actual_npu_publication(self):
        source = (DIRECTORY / "bindings.py").read_text()
        self.assertIn("from .npu_random import install as install_npu_random", source)
        self.assertIn("install_npu_random(g.npu, _modules)", source)

    def test_actual_native_rejection_propagates(self):
        def fail(*args):
            raise RuntimeError("native state rejected")
        self.api["_native_rng"] = lambda: SimpleNamespace(set_rng_state=fail)
        with self.assertRaisesRegex(RuntimeError, "native state rejected"):
            self.npu.set_rng_state(Tensor(list(b"foreign state")), 1)

    def test_random_alias_identity_and_all_state(self):
        for name in ("get_rng_state", "set_rng_state", "get_rng_state_all", "set_rng_state_all",
                     "initial_seed", "manual_seed", "manual_seed_all"):
            self.assertIs(getattr(self.npu, name), getattr(self.modules["torch.npu.random"], name))
        self.assertEqual([s.numpy().tobytes() for s in self.npu.get_rng_state_all()],
                         [s.encode("ascii") for s in self.blobs])


if __name__ == "__main__":
    unittest.main()
