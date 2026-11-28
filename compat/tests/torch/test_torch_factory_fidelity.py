import importlib
import unittest

import numpy as np
import pytest

import jittor as jt
import torch


FACTORY_NAMES = (
    "arange", "bernoulli", "empty", "empty_like", "full", "full_like",
    "linspace", "multinomial", "normal", "ones", "ones_like", "rand",
    "rand_like", "randint", "randn", "randn_like", "randperm", "tril",
    "triu", "zeros", "zeros_like",
)


class TestTorchFactoryFidelity(unittest.TestCase):

    def test_cuda_scalar_tensor_sequence_keeps_values_live_in_replay(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        with torch.no_grad(), jt.flag_scope(backend_fallback="error"):
            def apply(x):
                return torch.tensor([x.max(), x[0]], device=x.device)

            compiled = torch.compile(apply, mode="reduce-overhead")
            for offset in range(6):
                x = torch.tensor([2 + offset, 5 + offset], device="cuda")
                out = compiled(x)
                self.assertEqual(out.device.type, "cuda")
                self.assertEqual(out.tolist(), [5 + offset, 2 + offset])
            self.assertIsNone(compiled.refused)
            self.assertGreater(compiled.stats["graph"], 0)

        x = torch.tensor([3.], device="cuda", requires_grad=True)
        out = torch.tensor([x[0]], device=x.device, requires_grad=True)
        self.assertNotEqual(out.data_ptr(), x.data_ptr())
        self.assertTrue(out.is_leaf)
        out.sum().backward()
        self.assertIsNone(x.grad)
        self.assertEqual(out.grad.item(), 1.)

    def test_linspace_endpoint_can_feed_a_capture_time_list(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        with torch.no_grad(), jt.flag_scope(backend_fallback="error"):
            count = torch.tensor(4, device="cuda")
            values = torch.tensor([5.], device="cuda")

            def apply(x):
                points = torch.linspace(1., 0., count, device=x.device).tolist()
                return x + sum(points[:-1]), x + points[-1]

            compiled = torch.compile(apply, mode="reduce-overhead")
            for _ in range(5):
                total, last = compiled(values)
                self.assertAlmostEqual(total.item(), 7.)
                self.assertEqual(last.item(), 5.)
            self.assertIsNone(compiled.refused)
            self.assertGreater(compiled.stats["graph"], 0)

    def test_cuda_scalar_sequence_preserves_a_host_input(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        with torch.no_grad(), jt.flag_scope(backend_fallback="error"):
            def apply(x):
                return torch.tensor([x.sum()], device="cuda")

            compiled = torch.compile(apply, mode="reduce-overhead")
            for offset in range(6):
                x = torch.tensor([2 + offset, 5], device="cpu")
                out = compiled(x)
                self.assertEqual(out.device.type, "cuda")
                self.assertEqual(out.tolist(), [7 + offset])
            self.assertIsNone(compiled.refused)
            self.assertGreater(compiled.stats["replayed"], 0)

    def test_factory_objects_are_module_level_and_keep_public_identity(self):
        factories = importlib.import_module(
            "jittor.compat.torch.installers.factories")
        for name in FACTORY_NAMES:
            with self.subTest(name=name):
                implementation = getattr(factories, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch, name), implementation)
                self.assertEqual(implementation.__name__, name)

    def test_fidelity_report_is_complete_deterministic_and_conservative(self):
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        report = fidelity.fidelity_report(prefix="torch.")
        factory_records = tuple(
            record for record in report
            if record.api in {"torch." + name for name in FACTORY_NAMES}
        )
        self.assertEqual(
            tuple(record.api for record in factory_records),
            tuple("torch." + name for name in FACTORY_NAMES),
        )
        for record in factory_records:
            self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
            self.assertIs(
                fidelity.fidelity_of(record.api).implementation,
                record.implementation,
            )
            self.assertTrue(record.detail)

    def test_independently_imported_factory_executes_on_cpu(self):
        factories = importlib.import_module(
            "jittor.compat.torch.installers.factories")
        with jt.flag_scope(use_cuda=0):
            value = factories.zeros((2, 3), dtype=torch.float32)
        np.testing.assert_array_equal(value.numpy(), np.zeros((2, 3)))

    def test_randint_defaults_to_long_and_respects_explicit_dtype(self):
        with jt.flag_scope(use_cuda=0):
            plain = torch.randint(0, 128, (1, 8))
            seeded = torch.randint(
                0, 128, (1, 8),
                generator=torch.Generator(device="cpu").manual_seed(7))
            receive = torch.empty((1, 8), dtype=torch.long)
            explicit = torch.randint(0, 128, (1, 8), dtype=torch.int32)
            explicit_none = torch.randint(0, 128, (1, 8), dtype=None)
        for value in (plain, seeded, receive, explicit_none):
            self.assertEqual(value.dtype, torch.int64)
            self.assertEqual(tuple(value.shape), (1, 8))
        self.assertEqual(explicit.dtype, torch.int32)

    def test_randint_defaults_to_long_on_cuda(self):
        if not jt.has_cuda:
            self.skipTest("CUDA is unavailable")
        with jt.flag_scope(use_cuda=1):
            plain = torch.randint(0, 128, (1, 8), device="cuda")
            seeded = torch.randint(
                0, 128, (1, 8), device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(7))
            receive = torch.empty((1, 8), dtype=torch.long, device="cuda")
            for value in (plain, seeded, receive):
                self.assertEqual(value.dtype, torch.int64)
                self.assertEqual(value.device.type, "cuda")
                value.sync()

    def test_empty_like_implementation_is_family_owned_and_runs_on_cpu(self):
        factories = importlib.import_module(
            "jittor.compat.torch.installers.factories")
        self.assertEqual(factories.empty_like.__module__, factories.__name__)
        self.assertIs(getattr(torch, "empty_like"), factories.empty_like)
        record = importlib.import_module(
            "jittor.compat.torch.fidelity").fidelity_of("torch.empty_like")
        self.assertIn("device", record.detail)
        with jt.flag_scope(use_cuda=0):
            source = torch.ones((2, 3), dtype=torch.float64)
            value = factories.empty_like(source)
        self.assertEqual(tuple(value.shape), (2, 3))
        self.assertEqual(value.dtype, source.dtype)


@pytest.mark.parametrize('ambient_cuda', [0, 1])
def test_lightweight_cuda_empty_retains_layout_and_gradient_policy(monkeypatch, ambient_cuda):
    if not jt.has_cuda:
        pytest.skip("real CUDA required for lightweight allocation")
    from jittor.compat.torch.installers import factories

    original = factories._fresh_cuda_empty
    optimized = []

    def check(namespace, name, *args):
        result = original(namespace, name, *args)
        if result is not None:
            optimized.append(name)
        return result

    monkeypatch.setattr(factories, "_fresh_cuda_empty", check)
    with jt.flag_scope(use_cuda=ambient_cuda, auto_graph_replay=0, backend_fallback="error"):
        source = torch.ones((3, 5), dtype=torch.float32, device="cuda", requires_grad=True).t()
        allocated = torch.empty(2, 3, dtype=torch.bfloat16, device="cuda")
        grad_mode_empty = torch.empty_like(source)
        with torch.no_grad():
            strided = torch.empty_like(source)
        grad_leaf = torch.empty_like(source, requires_grad=True)
        host = torch.empty_like(source, device="cpu")
        assert optimized == ["empty", "empty_like"]
        assert allocated.dtype == torch.bfloat16 and allocated.device.type == "cuda"
        assert allocated.shape == (2, 3) and not allocated.requires_grad
        assert strided.is_leaf and not strided.requires_grad and strided.device.type == "cuda"
        assert strided.stride() == source.stride()
        assert grad_mode_empty.is_leaf and not grad_mode_empty.requires_grad
        assert grad_mode_empty.stride() == source.stride()
        assert grad_leaf.requires_grad and grad_leaf.is_leaf
        assert host.device.type == "cpu" and host.stride() == source.stride()
        strided.fill_(2)
        np.testing.assert_array_equal(strided.cpu().numpy(), np.full((5, 3), 2.0))
        np.testing.assert_array_equal(source.detach().cpu().numpy(), np.ones((5, 3)))


@pytest.mark.parametrize('ambient_cuda', [0, 1])
@pytest.mark.parametrize('host_allocator', [0, 1])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('shape', [(), (1,), (8,), (0, 3)])
@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float32', 'float64'])
def test_array_construction_honors_explicit_device_before_graph_edges_exist(
        ambient_cuda, host_allocator, device, shape, dtype):
    if not jt.has_cuda:
        pytest.skip('real CUDA build and device required for mixed-placement coverage')
    expected = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    if dtype == 'int64':
        expected += 2 ** 40
    elif dtype == 'float64':
        expected = np.nextafter(expected + 1, np.inf)
    source = expected.copy()
    with jt.flag_scope(use_cuda=ambient_cuda, use_cuda_host_allocator=host_allocator,
                       auto_graph_replay=0):
        value = torch.tensor(source, dtype=getattr(torch, dtype), device=device)
        source.fill(-9)
        value.sync(True, False)
        assert value.device.type == device
        assert value.dtype == getattr(torch, dtype)
        if expected.size:
            assert value.location() == ('cpu' if device == 'cpu' else 'device')
        np.testing.assert_array_equal(value.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
