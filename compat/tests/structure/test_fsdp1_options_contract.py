"""Host-only execution of FSDP1 option routing; no tensor/JIT/device claims."""
import ast
import contextlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]


class Device:
    def __init__(self, value):
        if isinstance(value, Device):
            self.type, self.index = value.type, value.index
        elif isinstance(value, bool):
            raise TypeError("bool device")
        elif isinstance(value, int):
            self.type, self.index = "npu", value
        else:
            fields = value.split(":")
            self.type = fields[0]
            self.index = int(fields[1]) if len(fields) == 2 else None


class Tensor:
    def __init__(self, dtype="float32"):
        self.dtype, self.device, self.is_meta = dtype, Device("cpu"), False


class NestedTensor:
    """Distinct from Tensor, matching the public jagged type contract."""
    def __init__(self, dtype):
        self.dtype = dtype


class Module:
    def __init__(self):
        self.parameter = Tensor()
        self.buffer = Tensor()
        self.integer_buffer = Tensor("int64")
        self.calls = []

    def parameters(self):
        return [self.parameter]

    def buffers(self):
        return [self.buffer, self.integer_buffer]

    def to(self, *, device):
        self.calls.append(device)
        for value in self.parameters() + self.buffers():
            value.device = device


class Mesh:
    def __init__(self, device_type, shape=(1,)):
        self.device_type, self.shape = device_type, shape


class TestFSDP1OptionsContract(unittest.TestCase):
    def setUp(self):
        config = types.ModuleType("config")
        exec(compile((ROOT / "compat/fsdp2/config.py").read_text(), "config.py", "exec"), vars(config))
        tree = ast.parse((ROOT / "compat/fsdp2/api.py").read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name.startswith("_fsdp1_")]
        self.world = types.SimpleNamespace(_world_size=lambda: 1)
        self.api = dict(
            __package__="_fsdp1_contract.fsdp2",
            config=config, common=self.world, _compat_device=Device,
            dtype_name=lambda value: str(value).replace("torch.", ""),
            _reject_unsupported_mesh=lambda mesh: None,
            dtensor=types.SimpleNamespace(init_device_mesh=lambda kind, shape: Mesh(kind, shape)),
            jt=types.SimpleNamespace(Var=Tensor, runtime=types.SimpleNamespace(
                use_cuda=0, scope=lambda **kwargs: contextlib.nullcontext())),
        )
        torch_owner = types.ModuleType("_fsdp1_contract.torch")
        torch_owner.nested_tensor_type = lambda: NestedTensor
        patches = mock.patch.dict(sys.modules, {torch_owner.__name__: torch_owner})
        patches.start()
        self.addCleanup(patches.stop)
        exec(compile(tree, "fsdp1_functions", "exec"), self.api)
        self.config = config
        self.module = Module()

    def prepare(self, **options):
        return self.api["_fsdp1_prepare_module"](self.module, (), options)

    def test_device_mesh_precision_identity_and_integer_buffer(self):
        parameter = self.module.parameter
        mesh = Mesh("npu")
        precision = self.config.MixedPrecision(param_dtype="float32", reduce_dtype="float32", buffer_dtype="float32")
        callback = lambda module: self.fail("materialized module must not call param_init_fn")
        mapped = self.prepare(device_id="npu:0", device_mesh=mesh, mixed_precision=precision,
                              use_orig_params=True, sync_module_states=True, param_init_fn=callback)
        self.assertIs(mapped["mesh"], mesh)
        self.assertIs(mapped["mp_policy"], precision)
        self.assertIs(self.module.parameter, parameter)
        self.assertEqual(len(self.module.calls), 1)
        for value in self.module.parameters() + self.module.buffers():
            self.assertEqual((value.device.type, value.device.index), ("npu", 0))
        self.assertEqual(self.module.integer_buffer.dtype, "int64")

    def test_integer_device_resolves_accelerator_and_builds_mesh(self):
        mapped = self.prepare(device_id=0, use_orig_params=True)
        self.assertEqual(mapped["mesh"].device_type, "npu")
        self.assertEqual(self.module.calls[0].index, 0)

    def test_legacy_no_options_has_no_move(self):
        self.prepare()
        self.assertEqual(self.module.calls, [])

    def test_unsupported_options_fail_before_moving(self):
        cases = [
            {"use_orig_params": False}, {"auto_wrap_policy": lambda **kw: True},
            {"cpu_offload": self.config.CPUOffload(True)}, {"forward_prefetch": True},
            {"ignored_modules": [Module()]}, {"ignored_states": [Tensor()]},
            {"process_group": object()}, {"unknown_option": True},
            {"sharding_strategy": self.config.ShardingStrategy.HYBRID_SHARD},
            {"device_mesh": Mesh("cpu")}, {"device_id": "meta"},
            {"sync_module_states": "yes"}, {"backward_prefetch": "anything"},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises((TypeError, ValueError, NotImplementedError)):
                self.prepare(**dict({"use_orig_params": True, "device_id": "npu:0"}, **case))
            self.assertEqual(self.module.calls, [])

    def test_multi_rank_and_meta_fail_before_moving(self):
        self.world._world_size = lambda: 2
        with self.assertRaisesRegex(NotImplementedError, "world_size=1"):
            self.prepare()
        with self.assertRaisesRegex(NotImplementedError, "world_size=1"):
            self.prepare(device_id="npu:0", use_orig_params=True)
        self.world._world_size = lambda: 1
        self.module.parameter.is_meta = True
        with self.assertRaisesRegex(NotImplementedError, "meta materialization"):
            self.prepare(device_id="npu:0", use_orig_params=True)
        self.assertEqual(self.module.calls, [])

    def test_precision_cast_and_input_cast_are_not_metadata_success(self):
        for field in ("param_dtype", "reduce_dtype", "buffer_dtype"):
            with self.subTest(field=field), self.assertRaisesRegex(NotImplementedError, field):
                self.prepare(device_id="npu:0", use_orig_params=True,
                             mixed_precision=self.config.MixedPrecision(**{field: "float16"}))
        self.assertEqual(self.module.calls, [])
        with self.assertRaisesRegex(NotImplementedError, "input casting"):
            self.api["_fsdp1_check_forward_inputs"]({"nested": [Tensor("float16")]}, "float32")
        self.api["_fsdp1_check_forward_inputs"]([Tensor("float32"), Tensor("int64")], "float32")

    def test_nested_precision_guard_checks_real_type_without_tensor_subclass(self):
        check = self.api["_fsdp1_check_forward_inputs"]
        self.assertNotIsInstance(NestedTensor("float16"), Tensor)
        for dtype in ("float16", "bfloat16", "float64"):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(NotImplementedError, "input casting"):
                check({"nested": (NestedTensor(dtype),)}, "float32")
        check({"nested": [NestedTensor("float32"), NestedTensor("int64")]}, "float32")
        # A lookalike flag must not give an unrelated object tensor semantics.
        check(types.SimpleNamespace(is_nested=True, dtype="float16"), "float32")


if __name__ == "__main__":
    unittest.main()
