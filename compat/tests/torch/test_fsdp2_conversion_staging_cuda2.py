"""Real CUDA2 Module.to staging, not whole-PPO or peak-memory acceptance.

CPU is only a storage destination. Values, gradients and storage-pointer
checks execute on CUDA. Run under the native NCCL2 launcher with separate
rank caches; an unavailable CUDA2 environment is a skip, never acceptance.
"""
from contextlib import ExitStack
import ctypes
from functools import partial
import importlib
import os
import unittest
from unittest.mock import patch

import pytest
import torch
import jittor as jt
from jittor.compat.diagnostics import counts
from jittor._runtime.fallback import forbid_backend_fallbacks

pytestmark = pytest.mark.cuda


class TestFSDP2ConversionStagingCUDA2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("real CUDA2 required; a skip is not acceptance")
        cls.dist = torch.distributed
        cls.created_group = not cls.dist.is_initialized()
        if cls.created_group:
            world = int(os.environ.get("WORLD_SIZE", jt.world_size))
            rank = int(os.environ.get("RANK", jt.rank))
            if world != 2:
                raise unittest.SkipTest("native NCCL world2 required")
            cls.dist.init_process_group("nccl", rank=rank, world_size=world)
        if cls.dist.get_world_size() != 2 or cls.dist.get_backend() != "nccl":
            raise AssertionError("actual NCCL2 group required")

    @classmethod
    def tearDownClass(cls):
        if cls.created_group:
            cls.dist.destroy_process_group()

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(forbid_backend_fallbacks())
        stack.enter_context(jt.runtime.scope(use_cuda=1, device_id=0,
                                             backend_fallback="error"))
        torch.cuda.set_device(0)
        self.initial_counts = counts()
        self.addCleanup(self.clean_diagnostics)
        self.driver = ctypes.CDLL("libcuda.so.1")
        self.driver.cuInit.argtypes = [ctypes.c_uint]
        self.driver.cuInit.restype = ctypes.c_int
        self.driver.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.driver.cuPointerGetAttribute.restype = ctypes.c_int
        self.assertEqual(self.driver.cuInit(0), 0)
        self.conversion = importlib.import_module("jittor.compat.fsdp2._module_conversion")
        self.methods = importlib.import_module("jittor.compat.torch.installers.nn.module_methods")
        self.shard = importlib.import_module("jittor.compat.fsdp2.shard")

    def clean_diagnostics(self):
        torch.cuda.synchronize(0)
        self.assertEqual(jt.core.backend_fallback_count(), 0)
        self.assertTrue(counts() == self.initial_counts,
                        "test body introduced swallowed diagnostics")

    def resident(self, value, kind, dtype):
        value.sync()
        self.assertEqual(value.dtype, dtype)
        self.assertEqual(value.device.type, kind)
        self.assertEqual((value.location(), int(value.placement_backend), int(value.device_id)),
                         ("device", 1, 0) if kind == "cuda" else ("cpu", 0, -1))
        if kind == "cuda" and int(value.numel()):
            self.assertEqual(tuple(jt.core.dispatch_context([value])), ("cuda", 0))
            pointer = int(value.device_raw_ptr)
            self.assertGreater(pointer, 0)
            ordinal, memory_type = ctypes.c_int(-1), ctypes.c_int(-1)
            self.assertEqual(self.driver.cuPointerGetAttribute(
                ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)), 0)
            self.assertEqual(self.driver.cuPointerGetAttribute(
                ctypes.byref(memory_type), 2, ctypes.c_void_p(pointer)), 0)
            self.assertEqual((ordinal.value, memory_type.value), (0, 2))

    def model(self, flat, reshard=True):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.full(
                    (4, 4), .5, device="cuda:0", dtype=torch.float32))
                self.alias = self.weight
                self.frozen = torch.nn.Parameter(torch.full(
                    (4,), .125, device="cuda:0", dtype=torch.float32), requires_grad=False)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight) + self.frozen

        with patch.dict(os.environ, {"JITTOR_FSDP2_FLAT": "1" if flat else "0"}):
            model = Model()
            fully_shard(model, mesh=init_device_mesh("cuda", (2,)),
                        reshard_after_forward=reshard)
        self.assertEqual(model._fsdp_state.true_fsdp_flat, flat)
        return model

    def sources(self, state):
        return ([state.true_fsdp_flat_shard] if state.true_fsdp_flat else
                [entry.shard for entry in state.true_fsdp_params])

    def exact(self, value, fill, dtype):
        self.resident(value, "cuda", dtype)
        expected = torch.full_like(value, fill, dtype=dtype, device="cuda:0")
        self.resident(expected, "cuda", dtype)
        self.assertTrue(bool(torch.equal(value, expected)))

    def aliases(self, model, originals, dtype, kind):
        state = model._fsdp_state
        self.assertIs(model.alias, model.weight)
        for entry in state.true_fsdp_params:
            self.assertIs(entry.shard, originals[entry.name])
            self.assertIs(getattr(model, entry.name), entry.shard)
            self.assertEqual(bool(entry.shard.requires_grad), entry.name != "frozen")
            self.assertEqual(bool(entry.shard.is_stop_grad()), entry.name == "frozen")
            self.assertTrue(entry.shard.is_leaf)
            self.assertIsNone(entry.shard.grad_fn)
            self.assertIsNone(entry.shard.grad)
            self.resident(entry.shard, kind, dtype)
        if state.true_fsdp_flat:
            owner = state.true_fsdp_flat_shard
            self.resident(owner, kind, dtype)
            for entry in state.true_fsdp_params:
                start, length = self.shard._flat_local_overlap(state, entry)
                if length:
                    self.assertEqual(int(entry.shard.data_ptr()),
                                     int(owner.data_ptr()) + start * owner.element_size())
                    self.assertEqual(entry.shard.storage_offset(), owner.storage_offset() + start)

    def exercise_roundtrip(self, flat):
        model = self.model(flat)
        state = model._fsdp_state
        originals = {entry.name: entry.shard for entry in state.true_fsdp_params}
        # Current FSDP backward discovers full-parameter targets/publication
        # through live optimizer groups. Keep that real owner throughout both
        # the unmoved control and the migration case; this test does not claim
        # support for optimizer-free FSDP backward.
        optimizer = torch.optim.SGD(model.parameters(), lr=.125)
        optimizer_params = tuple(optimizer.param_groups[0]["params"])
        self.assertEqual({id(p) for p in optimizer_params}, {id(p) for p in originals.values()})
        control_x = torch.full((2, 4), 1., device="cuda:0", requires_grad=True)
        model(control_x).sum().backward()
        self.exact(control_x.grad, 2., torch.float32)
        self.assertIsNotNone(model.weight.grad)
        self.exact(model.weight.grad.full_tensor(), 2., torch.float32)
        self.assertIsNone(model.frozen.grad)
        optimizer.zero_grad(set_to_none=True)
        self.assertTrue(all(entry.shard.grad is None for entry in state.true_fsdp_params))
        self.assertTrue(all(entry.full_param is None for entry in state.true_fsdp_params))
        self.assertFalse(state.true_fsdp_unsharded)
        del control_x
        old = {name: value.detach() for name, value in originals.items()}
        for value in self.sources(state):
            self.resident(value, "cuda", torch.float32)
        calls = []
        original_convert = self.methods._module_to_conversion

        def observe(ds, dev, copy, value):
            # The independent transport holder uses the genuine native detach,
            # not public DTensor.detach with stale snapshot/state methods.
            self.assertTrue(all(value is not source for source in self.sources(state)))
            self.assertIsNone(getattr(value, "_jittor_fsdp2_state", None))
            self.assertIsNone(getattr(value, "_jittor_fsdp2_detached_snapshot", None))
            self.assertTrue(value.is_stop_grad())
            calls.append(value.device.type)
            return original_convert(ds, dev, copy, value)

        with patch.object(self.methods, "_module_to_conversion", side_effect=observe), \
                patch.object(self.conversion, "copy_native_tensor",
                             side_effect=AssertionError("unexpected same-device staging copy")):
            for destination, dtype in (("cpu", torch.float32), ("cuda:0", torch.bfloat16),
                                       ("cpu", torch.bfloat16), ("cuda:0", torch.float32)):
                self.assertIs(model.to(device=destination, dtype=dtype), model)
                self.aliases(model, originals, dtype, "cpu" if destination == "cpu" else "cuda")
                self.assertTrue(all(a is b for a, b in zip(
                    optimizer_params, optimizer.param_groups[0]["params"])))
                if destination == "cuda:0":
                    for entry in state.true_fsdp_params:
                        if entry.name == "weight":
                            with torch.no_grad():
                                entry.shard.add_(.25)
                # A previously detached checkpoint still owns the old allocation
                # and dtype, even after writes to the reloaded parameters.
                for name, value in old.items():
                    self.exact(value.to_local(), .125 if name == "frozen" else .5, torch.float32)
        self.assertEqual(len(calls), 4 * len(self.sources(state)))
        self.exact(model.weight.full_tensor(), 1., torch.float32)
        self.exact(model.frozen.full_tensor(), .125, torch.float32)
        x = torch.full((2, 4), 1., device="cuda:0", requires_grad=True)
        model(x).sum().backward()
        self.exact(x.grad, 4., torch.float32)
        self.assertIsNotNone(model.weight.grad)
        self.exact(model.weight.grad.full_tensor(), 2., torch.float32)
        self.assertIsNone(model.frozen.grad)
        optimizer.step()
        self.exact(model.weight.full_tensor(), .75, torch.float32)
        self.exact(model.frozen.full_tensor(), .125, torch.float32)
        for name, value in old.items():
            self.exact(value.to_local(), .125 if name == "frozen" else .5, torch.float32)

    def test_flat_no_same_device_staging_copy_roundtrip(self):
        self.exercise_roundtrip(True)

    def test_nonflat_no_same_device_staging_copy_roundtrip(self):
        self.exercise_roundtrip(False)

    def test_failure_is_atomic_across_states(self):
        class FailedConversion(RuntimeError):
            pass
        for flat in (True, False):
            with self.subTest(flat=flat):
                parent = torch.nn.Module()
                parent.first = self.model(flat)
                parent.second = self.model(flat)
                states = [parent.first._fsdp_state, parent.second._fsdp_state]
                originals = [(value, int(value.device_raw_ptr))
                             for state in states for value in self.sources(state)]
                original_convert = self.methods._module_to_conversion
                calls = []

                def fail_later(ds, dev, copy, value):
                    calls.append(value)
                    if len(calls) == len(self.sources(states[0])) + 1:
                        raise FailedConversion("intentional second-state conversion failure")
                    return original_convert(ds, dev, copy, value)

                with patch.object(self.methods, "_module_to_conversion", side_effect=fail_later) as convert, \
                        patch.object(self.conversion, "copy_native_tensor",
                                     side_effect=AssertionError("unexpected staging copy")):
                    with self.assertRaises(FailedConversion):
                        self.conversion._convert_module_storage(
                            parent, partial(convert, None, torch.device("cpu"), False))
                for value, pointer in originals:
                    # Observe original storage before any restoring device move.
                    self.resident(value, "cuda", torch.float32)
                    self.assertEqual(int(value.device_raw_ptr), pointer)
                for child in (parent.first, parent.second):
                    self.exact(child.weight.full_tensor(), .5, torch.float32)
                    self.exact(child.frozen.full_tensor(), .125, torch.float32)

    def test_live_graph_and_pending_gradient_still_rejected(self):
        for flat in (True, False):
            with self.subTest(flat=flat):
                live = self.model(flat)
                x = torch.ones((2, 4), device="cuda:0", requires_grad=True)
                y = live(x)
                self.assertTrue(any(entry.full_param is not None
                                    for entry in live._fsdp_state.true_fsdp_params))
                with self.assertRaises(NotImplementedError):
                    live.to("cpu")
                self.exact(y, 2.125, torch.float32)
                pending = self.model(flat)
                entry = next(e for e in pending._fsdp_state.true_fsdp_params if int(e.shard.numel()))
                entry.shard.grad = torch.zeros_like(entry.shard)
                gradient = entry.shard.grad
                with self.assertRaises(NotImplementedError):
                    pending.to("cpu")
                self.assertIs(entry.shard.grad, gradient)
                self.resident(entry.shard, "cuda", torch.float32)
                entry.shard.grad = None
