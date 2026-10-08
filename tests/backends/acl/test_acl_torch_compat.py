
from _helpers import capability as _test_capability
import unittest

import numpy as np

import torch
import jittor as jt
from jittor.nn.backends import hooks as backend_hooks
from jittor._runtime.dispatch import override_kernel, registered_kernel


def _assert_acl_device(test_case, value):
    """Check executed placement before a host fetch changes tensor residency."""
    test_case.assertTrue(jt.compiler.has_acl)
    test_case.assertEqual(jt.runtime.use_cuda, 1)
    value.sync()
    test_case.assertEqual(value.location(), "device")
    test_case.assertGreaterEqual(value.device_id, 0)
    # -1 is native FollowRuntime, which selects ACL in this runtime;
    # 2 is explicit BackendId::Acl. Do not force native graph placement.
    test_case.assertIn(value.placement_backend, (-1, 2))
    return value


def _fetch_acl(test_case, values, *, as_float=False):
    values = list(values)
    for value in values:
        _assert_acl_device(test_case, value)
    if as_float:
        values = [value.float() for value in values]
    return jt.fetch_sync(values)


def _bfloat16_round(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32).copy()
    bits += np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))
    return (bits & np.uint32(0xffff0000)).view(np.float32)


def _sort_projection_records(torch_owner, device, check_device=None):
    """FP32 sort values and input gradients for distinct and stable tied keys."""
    shape = (3, 4)
    grid = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    unique = ((grid * 7) % 13 - 6) / 8
    tied = grid % 2
    projection = ((grid * 5) % 17 - 8) / 16
    records = []
    for tied_keys, data in ((False, unique), (True, tied)):
        for axis in (0, -1):
            for descending in (False, True):
                for stable in ((True,) if tied_keys else (False, True)):
                    source = torch_owner.tensor(
                        data, dtype=torch_owner.float32, device=device,
                        requires_grad=True)
                    weight = torch_owner.tensor(
                        projection, dtype=torch_owner.float32, device=device)
                    ordered = torch_owner.sort(
                        source, dim=axis, descending=descending, stable=stable)
                    loss = (ordered.values * weight).sum()
                    loss.backward()
                    for value in (source, ordered.values, source.grad):
                        assert value.dtype == torch_owner.float32
                        assert tuple(value.shape) == shape
                    assert ordered.indices.dtype == torch_owner.int64
                    assert tuple(ordered.indices.shape) == shape
                    assert loss.dtype == torch_owner.float32
                    assert loss.numel() == 1
                    for value in (source, weight, ordered.values,
                                  ordered.indices, loss, source.grad):
                        assert value.device.type == "npu"
                    if check_device is not None:
                        for value in (source, weight, ordered.values,
                                      ordered.indices, loss, source.grad):
                            check_device(value)
                    values = ordered.values.detach().cpu().numpy()
                    indices = ordered.indices.detach().cpu().numpy()
                    gradient = source.grad.detach().cpu().numpy()
                    assert np.isfinite(values).all()
                    assert np.isfinite(gradient).all()
                    assert np.isfinite(loss.detach().cpu().item())
                    records.append({
                        "tied": tied_keys, "axis": axis,
                        "descending": descending, "stable": stable,
                        "values": values.tolist(), "indices": indices.tolist(),
                        "gradient": gradient.tolist(),
                    })
    return records


@unittest.skipIf(not _test_capability.check_accelerator('acl', backend=jt).enabled, "No ACL found")
class TestACLTorchCompat(unittest.TestCase):
    def setUp(self):
        # Fail closed if the runner imports binary PyTorch or the old native alias.
        self.assertIsNot(torch, jt)
        self.assertIsNot(torch.Tensor, jt.Var)
        self.assertIs(torch.Tensor._frontend_backend, jt)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_npu_int64_roll_stays_on_acl(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            labels = torch.tensor([[-100, 10, 11, 2], [-100, 12, 13, 2]],
                                  dtype=torch.int64, device="npu:0")
            rolled = torch.roll(labels, -1, 1)
            _assert_acl_device(self, rolled)
            np.testing.assert_array_equal(
                rolled.detach().cpu().numpy(),
                [[10, 11, 2, -100], [12, 13, 2, -100]],
            )
        self.assertEqual(jt.core.backend_fallback_count() - before, 0)


    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_npu_bare_none_index_preserves_values_and_gradient(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            for data in (np.array(3.0, dtype=np.float32),
                         np.array([2.0, -3.0, 4.0], dtype=np.float32)):
                with self.subTest(shape=data.shape):
                    source = torch.tensor(data, device="npu", requires_grad=True)
                    expanded = source[None]
                    self.assertEqual(tuple(expanded.shape), (1,) + data.shape)
                    _assert_acl_device(self, expanded)
                    (expanded * 2.0).sum().backward()
                    self.assertIsNotNone(source.grad)
                    self.assertEqual(tuple(source.grad.shape), data.shape)
                    _assert_acl_device(self, source)
                    _assert_acl_device(self, source.grad)
                    actual, gradient = _fetch_acl(
                        self, [expanded.detach().clone(), source.grad.detach().clone()])
                    np.testing.assert_array_equal(actual, data[None])
                    np.testing.assert_array_equal(gradient, np.full(data.shape, 2.0, dtype=np.float32))
        self.assertEqual(jt.core.backend_fallback_count(), before)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_npu_copy_preserves_parameter_trainability(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            for factory in ("parameter", "linear"):
                for host_source in (False, True):
                    for trainable in (False, True):
                        with self.subTest(factory=factory, host_source=host_source, trainable=trainable):
                            if factory == "linear":
                                module = torch.nn.Linear(2, 2, bias=False)
                                parameter = dict(module.named_parameters())["weight"]
                                module.eval()
                            else:
                                parameter = torch.nn.Parameter(
                                    torch.tensor([[1.0, 2.0], [1.0, 2.0]], device="npu"))
                            parameter.requires_grad_(trainable)
                            values = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
                            source = torch.from_numpy(values) if host_source else torch.tensor(values, device="npu")
                            self.assertEqual(bool(parameter.requires_grad), trainable)
                            with torch.no_grad():
                                result = parameter.copy_(source)
                            self.assertIs(result, parameter)
                            self.assertEqual(bool(parameter.requires_grad), trainable)
                            _assert_acl_device(self, parameter)
                            np.testing.assert_array_equal(parameter.detach().cpu().numpy(), values)
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_npu_inplace_preserves_connected_gradient(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            source = torch.tensor([1.0, 2.0], device="npu", requires_grad=True)
            value = source * 2
            self.assertIs(value.mul_(3), value)
            value.sum().backward()
            _assert_acl_device(self, value)
            self.assertIsNotNone(source.grad)
            _assert_acl_device(self, source.grad)
            np.testing.assert_array_equal(source.grad.detach().cpu().numpy(), [6.0, 6.0])
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_npu_fetch_callback_lifecycle(self):
        """Async fetch delivers once, drains, and accepts later NPU work."""
        from jittor._runtime.fallback import forbid_backend_fallbacks

        seen = []

        def capture(tag, array):
            self.assertIsInstance(array, np.ndarray)
            seen.append((tag, array.copy()))

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            source = torch.tensor([1.0, 2.0, 3.0], device="npu")
            first = source * 2
            second = source + 4
            for value in (source, first, second):
                self.assertIs(type(value), torch.Tensor)
                _assert_acl_device(self, value)
                self.assertEqual(value.placement_backend, 2)
                self.assertEqual(value.device_id, 0)
            jt.fetch(first, lambda array: capture("first", array))
            jt.fetch(second, lambda array: capture("second", array))
            jt.sync_all(True)
            self.assertEqual([tag for tag, _ in seen], ["first", "second"])
            np.testing.assert_array_equal(seen[0][1], [2.0, 4.0, 6.0])
            np.testing.assert_array_equal(seen[1][1], [5.0, 6.0, 7.0])

            # Repeated draining must not replay either callback.
            jt.sync_all(True)
            jt.sync_all(True)
            self.assertEqual([tag for tag, _ in seen], ["first", "second"])

            third = source * 3
            _assert_acl_device(self, third)
            self.assertEqual(third.placement_backend, 2)
            self.assertEqual(third.device_id, 0)
            jt.fetch(third, lambda array: capture("third", array))
            jt.sync_all(True)
            self.assertEqual([tag for tag, _ in seen], ["first", "second", "third"])
            np.testing.assert_array_equal(seen[2][1], [3.0, 6.0, 9.0])
            jt.sync_all(True)
            self.assertEqual(len(seen), 3)
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_sort_axes_options_and_projection_backward_matches_torch_npu(self):
        import inspect
        import json
        import os
        import subprocess
        import tempfile
        from _helpers.child_process import child_env, default_timeout

        oracle = os.environ.get("REAL_TORCH_PYTHON", "")
        if not oracle:
            if os.environ.get("JITTOR_REQUIRE_REAL_TORCH") == "1":
                self.fail("REAL_TORCH_PYTHON is required for sort gradients")
            self.skipTest("independent PyTorch is not configured")
        source = (
            "import torch\n"
            "assert not hasattr(torch, '_torch_compat_install_context')\n"
            "import torch_npu, numpy as np, json\n"
            "assert torch.npu.is_available()\n"
            "torch.npu.set_device(0)\n"
        )
        source += inspect.getsource(_sort_projection_records)
        source += "\nprint(json.dumps(_sort_projection_records(torch, 'npu:0')))\n"
        env = child_env(without_torch_mode=True, repo_paths=False)
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [oracle, "-c", source], env=env, cwd=cwd,
                capture_output=True, text=True, timeout=default_timeout())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = json.loads(result.stdout.strip().splitlines()[-1])
        before = jt.core.backend_fallback_count()
        actual = _sort_projection_records(
            torch, "npu:0", lambda value: _assert_acl_device(self, value))
        self.assertEqual(len(actual), 12)
        self.assertEqual(actual, expected)
        self.assertEqual(jt.core.backend_fallback_count(), before)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_independent_frontend_tensor_executes_on_acl(self):
        source = torch.tensor([1.0, 2.0], device="npu", requires_grad=True)
        output = (source * source).sum()
        gradient, = torch.autograd.grad(output, source)
        self.assertIs(type(source), torch.Tensor)
        self.assertIs(type(output), torch.Tensor)
        self.assertIs(type(gradient), torch.Tensor)
        output.sync()
        gradient.sync()
        self.assertEqual(output.placement_backend, 2)
        self.assertEqual(gradient.placement_backend, 2)
        self.assertEqual(output.location(), "device")
        self.assertEqual(gradient.location(), "device")
        self.assertEqual(output.item(), 5.0)
        np.testing.assert_array_equal(gradient.detach().cpu().numpy(), [2.0, 4.0])

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_fused_adamw_bfloat16_matches_cann_two_steps(self):
        initial = [1.0, -2.0, 0.5, -0.25, 4.0, -8.0, 0.125, -0.0625]
        parameters = [
            torch.tensor(initial, dtype=torch.bfloat16).requires_grad_(True)
            for _ in range(2)
        ]
        optimizer = torch.optim.AdamW(
            parameters, lr=0.01, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0.1, fused=True)
        gradients = (
            [0.25, -0.5, 1.0, -2.0, 0.03125, -0.0625, 4.0, -8.0],
            [-0.125, 0.25, -0.5, 1.0, -0.015625, 0.03125, -2.0, 4.0],
        )
        expected_parameters = (
            [0.98828125, -1.984375, 0.490234375, -0.240234375,
             3.984375, -7.96875, 0.11474609375, -0.052490234375],
            [0.984375, -1.9765625, 0.486328125, -0.2373046875,
             3.984375, -7.96875, 0.11181640625, -0.0498046875],
        )

        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            for gradient, expected in zip(gradients, expected_parameters):
                for parameter in parameters:
                    parameter.grad = torch.tensor(
                        gradient, dtype=torch.bfloat16)
                optimizer.step()
                for parameter in parameters:
                    np.testing.assert_array_equal(
                        parameter.float().numpy(),
                        np.asarray(expected, dtype=np.float32),
                    )

        expected_moment = np.asarray(
            [0.010009765625, -0.02001953125, 0.0400390625, -0.080078125,
             0.001251220703125, -0.00250244140625, 0.16015625, -0.3203125],
            dtype=np.float32,
        )
        expected_variance = np.asarray(
            [7.82012939453125e-05, 0.00031280517578125, 0.001251220703125,
             0.0050048828125, 1.2218952178955078e-06,
             4.887580871582031e-06, 0.02001953125, 0.080078125],
            dtype=np.float32,
        )
        for parameter in parameters:
            state = optimizer.state[parameter]
            self.assertEqual(state["step"], 2.0)
            np.testing.assert_array_equal(
                state["exp_avg"].float().numpy(), expected_moment)
            np.testing.assert_array_equal(
                state["exp_avg_sq"].float().numpy(), expected_variance)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_adamw_bfloat16_state_scalar_stays_on_acl(self):
        parameter = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
        parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        before = parameter.float().numpy().copy()

        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            (parameter * parameter).sum().backward()
            optimizer.step()
            after = parameter.float().numpy()

        self.assertEqual(str(parameter.dtype).replace("torch.", ""), "bfloat16")
        self.assertEqual(
            str(optimizer.state[parameter]["exp_avg"].dtype).replace("torch.", ""),
            "bfloat16",
        )
        self.assertEqual(
            str(optimizer.state[parameter]["exp_avg_sq"].dtype).replace("torch.", ""),
            "bfloat16",
        )
        self.assertTrue(np.isfinite(after).all())
        self.assertFalse(np.array_equal(after, before))

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_named_rms_norm_executes_custom_forward_and_gradient(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        class CustomRMSNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.full((8,), 3.0, device="npu"))
                self.variance_epsilon = 1e-6
                self.calls = 0

            def forward(self, value):
                self.calls += 1
                return value * self.weight + 2.0

        def forbidden(*args, **kwargs):
            raise AssertionError("A class-name shortcut replaced custom forward")

        before = jt.core.backend_fallback_count()
        with forbid_backend_fallbacks(), override_kernel("nn.rms_norm.training", "cuda", forbidden), override_kernel("nn.rms_norm.inference", "cuda", forbidden):
            module = CustomRMSNorm()
            value = torch.ones((2, 8), device="npu", requires_grad=True)
            output = module(value)
            dx, dw = torch.autograd.grad(output.sum(), (value, module.weight))
            actual = _fetch_acl(self, [output, dx, dw])
        self.assertEqual(module.calls, 1)
        np.testing.assert_array_equal(actual[0], np.full((2, 8), 5.0))
        np.testing.assert_array_equal(actual[1], np.full((2, 8), 3.0))
        np.testing.assert_array_equal(actual[2], np.full((8,), 2.0))
        self.assertEqual(jt.core.backend_fallback_count(), before)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_standard_rms_norm_bfloat16_matches_pytorch_order(self):
        class FixtureRMSNorm(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(
                    weight, dtype=torch.bfloat16))
                self.variance_epsilon = 1e-6

            def forward(self, hidden_states):
                self.forward_calls = getattr(self, "forward_calls", 0) + 1
                dtype = hidden_states.dtype
                values = hidden_states.float()
                variance = values.pow(2).mean(-1, keepdim=True)
                normalized = values * torch.rsqrt(variance + self.variance_epsilon)
                return self.weight * normalized.to(dtype)

        rng = np.random.RandomState(20260901)
        source_np = rng.randn(2, 3, 128).astype("float32")
        weight_np = rng.uniform(0.1, 1.2, size=(128,)).astype("float32")
        cotangent_np = rng.randn(2, 3, 128).astype("float32")
        source_bf = _bfloat16_round(source_np)
        weight_bf = _bfloat16_round(weight_np)
        cotangent_bf = _bfloat16_round(cotangent_np)

        module = FixtureRMSNorm(weight_bf)
        source = torch.tensor(
            source_bf, dtype=torch.bfloat16).requires_grad_(True)
        cotangent = torch.tensor(cotangent_bf, dtype=torch.bfloat16)
        output = module(source)
        repeated = module(source)
        self.assertEqual(module.forward_calls, 2)
        grad_source, grad_weight = torch.autograd.grad(
            (output * cotangent).sum(), (source, module.weight)
        )
        with torch.no_grad():
            inference = module(source)
        self.assertEqual(module.forward_calls, 3)
        values = _fetch_acl(
            self, [output, repeated, inference, grad_source, grad_weight],
            as_float=True)

        inverse_rms = np.float32(1.0) / np.sqrt(
            np.mean(
                source_bf * source_bf,
                axis=-1,
                keepdims=True,
                dtype=np.float32,
            ) + np.float32(1e-6)
        )
        normalized = source_bf * inverse_rms
        normalized_bf = _bfloat16_round(normalized)
        expected_output = _bfloat16_round(weight_bf * normalized_bf)
        grad_normalized = _bfloat16_round(cotangent_bf * weight_bf)
        mean_projection = np.mean(
            grad_normalized * normalized,
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        )
        expected_grad_source = _bfloat16_round(
            inverse_rms * (grad_normalized - normalized * mean_projection)
        )
        expected_grad_weight = _bfloat16_round(np.sum(
            _bfloat16_round(cotangent_bf * normalized_bf),
            axis=(0, 1),
            dtype=np.float32,
        ))
        expected = (
            expected_output,
            expected_output,
            expected_output,
            expected_grad_source,
            expected_grad_weight,
        )
        for actual, reference in zip(values, expected):
            np.testing.assert_array_equal(actual, reference)
        self.assertIsNone(getattr(module.weight, "_torch_acl_rms_norm_unit_weight", None))
        self.assertEqual(module.forward_calls, 3)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_dual_rms_norm_bfloat16_matches_pytorch_order(self):
        rng = np.random.RandomState(20260902)
        first_np = rng.randn(2, 3, 128).astype("float32")
        second_np = rng.randn(2, 2, 128).astype("float32")
        first_weight_np = rng.uniform(0.1, 1.2, 128).astype("float32")
        second_weight_np = rng.uniform(0.1, 1.2, 128).astype("float32")

        first_bf = _bfloat16_round(first_np)
        second_bf = _bfloat16_round(second_np)
        first_weight_bf = _bfloat16_round(first_weight_np)
        second_weight_bf = _bfloat16_round(second_weight_np)
        first = torch.tensor(first_bf, dtype=torch.bfloat16)
        second = torch.tensor(second_bf, dtype=torch.bfloat16)
        first_weight = torch.tensor(first_weight_bf, dtype=torch.bfloat16)
        second_weight = torch.tensor(second_weight_bf, dtype=torch.bfloat16)

        with torch.no_grad(), jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
            actual = jt.nn.dual_rms_norm(
                first, second, first_weight, second_weight, 1e-6)
            values = jt.fetch_sync([value.float() for value in actual])
            locations = tuple(value.location() for value in actual)

        def reference(value, weight):
            inverse_rms = np.float32(1.0) / np.sqrt(
                np.mean(
                    value * value, axis=-1, keepdims=True, dtype=np.float32
                ) + np.float32(1e-6)
            )
            normalized = _bfloat16_round(value * inverse_rms)
            return _bfloat16_round(weight * normalized)

        self.assertEqual(locations, ("device", "device"))
        np.testing.assert_array_equal(
            values[0], reference(first_bf, first_weight_bf))
        np.testing.assert_array_equal(
            values[1], reference(second_bf, second_weight_bf))

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_grouped_qk_rms_norm_rotary_matches_separate_ops(self):
        rng = np.random.RandomState(20260902)
        head_size = 128
        query_np = rng.randn(1, 16 * head_size).astype("float32")
        key_np = rng.randn(1, 8 * head_size).astype("float32")
        query_weight_np = rng.uniform(0.1, 1.2, head_size).astype("float32")
        key_weight_np = rng.uniform(0.1, 1.2, head_size).astype("float32")
        inv = 1.0 / (
            10000 ** (np.arange(0, head_size, 2) / head_size)
        )
        angles = np.arange(32)[:, None] * inv[None, :]
        cache_np = np.concatenate(
            (np.cos(angles), np.sin(angles)), axis=-1
        ).astype("float32")

        query = torch.tensor(query_np, dtype=torch.bfloat16)
        key = torch.tensor(key_np, dtype=torch.bfloat16)
        query_weight = torch.tensor(query_weight_np, dtype=torch.bfloat16)
        key_weight = torch.tensor(key_weight_np, dtype=torch.bfloat16)
        positions = torch.tensor([13], dtype=torch.int64)
        cache = torch.tensor(cache_np, dtype=torch.bfloat16)

        with torch.no_grad(), jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
            actual = backend_hooks.acl_grouped_qk_rms_norm_rotary(
                positions, query, key, query_weight, key_weight,
                cache, head_size, head_size, True, 1e-6)
            query_view = query.reshape((1, 16, head_size))
            key_view = key.reshape((1, 8, head_size))
            reference_query, reference_key = jt.nn.dual_rms_norm(
                query_view, key_view, query_weight, key_weight, 1e-6)
            reference = jt.nn.rotary_embedding(
                positions,
                reference_query.reshape(query.shape),
                reference_key.reshape(key.shape),
                cache,
                head_size=head_size,
                rotary_dim=head_size,
                is_neox=True,
            )
            values = jt.fetch_sync([
                actual[0].float(), actual[1].float(),
                reference[0].float(), reference[1].float(),
            ])
            locations = actual[0].location(), actual[1].location()

        self.assertEqual(locations, ("device", "device"))
        np.testing.assert_array_equal(values[0], values[2])
        np.testing.assert_array_equal(values[1], values[3])

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_python_float_truediv_stays_on_acl(self):
        source_np = np.array([0.12345679, 1.2345679, 3.25], dtype=np.float32)
        scale = 0.28209479177387814
        source = torch.tensor(source_np, dtype=torch.float32)
        source.requires_grad_(True)

        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            quotient = source / scale
            reflected = scale / source
            gradient = torch.autograd.grad(
                (quotient + reflected).sum(), source
            )[0]
            self.assertEqual(str(quotient.dtype).replace("torch.", ""), "float32")
            self.assertEqual(str(reflected.dtype).replace("torch.", ""), "float32")
            quotient, reflected, gradient = jt.fetch_sync(
                [quotient, reflected, gradient]
            )

        scale32 = np.float32(scale)
        np.testing.assert_array_equal(quotient, source_np / scale32)
        np.testing.assert_array_equal(reflected, scale32 / source_np)
        np.testing.assert_allclose(
            gradient,
            np.float32(1.0) / scale32 - scale32 / (source_np * source_np),
            rtol=2e-6,
            atol=2e-6,
        )

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_python_float_mul_keeps_bfloat16_on_acl(self):
        source_np = _bfloat16_round(np.asarray(
            [1.0, -2.0, 3.5, -7.25], dtype=np.float32))
        scale = 128 ** -0.5
        source = torch.tensor(source_np, dtype=torch.bfloat16)

        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            scaled = source * scale
            reflected = scale * source
            self.assertEqual(str(scaled.dtype).replace("torch.", ""), "bfloat16")
            self.assertEqual(str(reflected.dtype).replace("torch.", ""), "bfloat16")
            scaled.sync()
            reflected.sync()
            values = jt.fetch_sync([scaled.float(), reflected.float()])

        expected = _bfloat16_round(source_np * np.float32(scale))
        np.testing.assert_array_equal(values[0], expected)
        np.testing.assert_array_equal(values[1], expected)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_roll_bfloat16_forward_backward_stays_on_acl(self):
        rng = np.random.RandomState(20260901)
        source_np = _bfloat16_round(rng.randn(2, 3, 8).astype("float32"))
        cotangent_np = _bfloat16_round(rng.randn(2, 3, 8).astype("float32"))
        source = torch.tensor(
            source_np, dtype=torch.bfloat16).requires_grad_(True)
        cotangent = torch.tensor(cotangent_np, dtype=torch.bfloat16)

        output = torch.roll(source, shifts=4, dims=-1)
        flat = torch.roll(source, shifts=5)
        gradient = torch.autograd.grad(
            (output * cotangent).sum(), source
        )[0]
        values = _fetch_acl(
            self, [output, flat, gradient], as_float=True)

        np.testing.assert_array_equal(values[0], np.roll(source_np, 4, axis=-1))
        np.testing.assert_array_equal(values[1], np.roll(source_np.reshape(-1), 5).reshape(source_np.shape))
        np.testing.assert_array_equal(values[2], np.roll(cotangent_np, -4, axis=-1))

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_nearest_interpolate_forward_backward_stays_on_acl(self):
        source_np = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
        for output_size in ((6, 8), (5, 7)):
            with self.subTest(output_size=output_size):
                source = torch.tensor(source_np, dtype=torch.float32)
                source.requires_grad_(True)

                with jt.log_capture_scope(
                    log_v=0, log_vprefix="acl_op_exec.cc=100"
                ) as logs:
                    output = torch.nn.functional.interpolate(
                        source, size=output_size, mode="nearest"
                    )
                    weight = torch.arange(
                        output.numel(), dtype=torch.float32
                    ).reshape(output.shape)
                    gradient = torch.autograd.grad(
                        (output * weight).sum(), source
                    )[0]
                    output, gradient = jt.fetch_sync([output, gradient])

                row_indices = np.floor(
                    np.arange(output_size[0]) * source_np.shape[2] / output_size[0]
                ).astype(np.int64)
                column_indices = np.floor(
                    np.arange(output_size[1]) * source_np.shape[3] / output_size[1]
                ).astype(np.int64)
                expected_output = np.take(
                    np.take(source_np, row_indices, axis=2), column_indices, axis=3
                )
                expected_weight = np.arange(
                    np.prod(expected_output.shape), dtype=np.float32
                ).reshape(expected_output.shape)
                expected_gradient = np.zeros_like(source_np)
                for output_row, input_row in enumerate(row_indices):
                    for output_column, input_column in enumerate(column_indices):
                        expected_gradient[:, :, input_row, input_column] += (
                            expected_weight[:, :, output_row, output_column]
                        )

                np.testing.assert_array_equal(output, expected_output)
                np.testing.assert_array_equal(gradient, expected_gradient)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_group_norm_forward_backward_stays_on_acl(self):
        source_np = np.random.RandomState(0).randn(2, 4, 3, 5).astype("float32")
        weight_np = np.array([0.5, 1.25, -0.75, 2.0], dtype="float32")
        bias_np = np.array([-0.2, 0.1, 0.3, -0.4], dtype="float32")
        loss_weight_np = np.random.RandomState(1).randn(2, 4, 3, 5).astype(
            "float32"
        )

        candidates = []
        native_dispatches = []
        acl_group_norm = registered_kernel("nn.group_norm", "acl")
        self.assertIsNotNone(acl_group_norm)

        def record_group_norm(*args):
            result = acl_group_norm(*args)
            native_dispatches.append(result is not None)
            return result

        with override_kernel("nn.group_norm", "acl", record_group_norm):
            self.assertTrue(jt.flags.use_acl)
            self.assertTrue(jt.flags.use_cuda)
            with jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100"
            ) as logs:
                module = torch.nn.GroupNorm(2, 4, eps=1e-5)
                module.weight.assign(weight_np)
                module.bias.assign(bias_np)
                module_source = torch.tensor(source_np)
                module_source.requires_grad_(True)
                module_output = module(module_source)
                module_grads = torch.autograd.grad(
                    (module_output * torch.tensor(loss_weight_np)).sum(),
                    (module_source, module.weight, module.bias),
                )
                candidates.append(
                    jt.fetch_sync([module_output] + list(module_grads))
                )

                functional_source = torch.tensor(source_np)
                functional_weight = torch.tensor(weight_np)
                functional_bias = torch.tensor(bias_np)
                for value in (
                    functional_source, functional_weight, functional_bias
                ):
                    value.requires_grad_(True)
                functional_output = torch.nn.functional.group_norm(
                    functional_source, 2, functional_weight, functional_bias, 1e-5
                )
                functional_grads = torch.autograd.grad(
                    (functional_output * torch.tensor(loss_weight_np)).sum(),
                    (functional_source, functional_weight, functional_bias),
                )
                candidates.append(
                    jt.fetch_sync([functional_output] + list(functional_grads))
                )

        self.assertEqual(native_dispatches, [True, True])
        with jt.flag_scope(use_acl=0, use_cuda=0):
            reference_module = torch.nn.GroupNorm(2, 4, eps=1e-5)
            reference_module.weight.assign(weight_np)
            reference_module.bias.assign(bias_np)
            reference_source = torch.tensor(source_np)
            reference_source.requires_grad_(True)
            reference_output = reference_module(reference_source)
            reference_grads = torch.autograd.grad(
                (reference_output * torch.tensor(loss_weight_np)).sum(),
                (
                    reference_source,
                    reference_module.weight,
                    reference_module.bias,
                ),
            )
            reference = jt.fetch_sync([reference_output] + list(reference_grads))

        for candidate in candidates:
            for actual, expected in zip(candidate, reference):
                np.testing.assert_allclose(
                    actual, expected, rtol=2e-4, atol=2e-4
                )

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_batch_norm_eval_forward_backward_stays_on_acl(self):
        rng = np.random.RandomState(20260831)
        source_np = rng.randn(2, 4, 3, 5).astype("float32")
        weight_np = rng.randn(4).astype("float32")
        bias_np = rng.randn(4).astype("float32")
        mean_np = rng.randn(4).astype("float32")
        variance_np = (np.abs(rng.randn(4)) + 0.5).astype("float32")
        loss_weight_np = rng.randn(*source_np.shape).astype("float32")

        dispatches = []
        acl_batch_norm = registered_kernel("nn.batch_norm.eval", "acl")
        self.assertIsNotNone(acl_batch_norm)

        def record_batch_norm(*args):
            result = acl_batch_norm(*args)
            dispatches.append(result is not None)
            return result

        with override_kernel("nn.batch_norm.eval", "acl", record_batch_norm):
            module = torch.nn.BatchNorm2d(4)
            module.eval()
            module.weight.assign(weight_np).start_grad()
            module.bias.assign(bias_np).start_grad()
            module.running_mean.assign(mean_np)
            module.running_var.assign(variance_np)
            source = torch.tensor(source_np)
            source.requires_grad_(True)
            with jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100"
            ) as logs:
                output = module(source)
                gradients = torch.autograd.grad(
                    (output * torch.tensor(loss_weight_np)).sum(),
                    (source, module.weight, module.bias),
                )
                candidate = jt.fetch_sync([output] + list(gradients))

        invstd = 1.0 / np.sqrt(variance_np + 1e-5)
        broadcast = (None, slice(None), None, None)
        normalized = (
            source_np - mean_np[broadcast]
        ) * invstd[broadcast]
        expected = [
            normalized * weight_np[broadcast] + bias_np[broadcast],
            loss_weight_np * weight_np[broadcast] * invstd[broadcast],
            (loss_weight_np * normalized).sum(axis=(0, 2, 3)),
            loss_weight_np.sum(axis=(0, 2, 3)),
        ]

        self.assertEqual(dispatches, [True])
        for actual, reference in zip(candidate, expected):
            np.testing.assert_allclose(
                actual, reference, rtol=2e-4, atol=2e-4
            )

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_layer_norm_forward_backward_stays_on_acl(self):
        rng = np.random.RandomState(20260831)
        source_np = rng.randn(2, 12, 32).astype("float32")
        weight_np = rng.randn(32).astype("float32")
        bias_np = rng.randn(32).astype("float32")
        loss_weight_np = rng.randn(*source_np.shape).astype("float32")

        module = torch.nn.LayerNorm(32)
        module.weight.assign(weight_np)
        module.bias.assign(bias_np)
        source = torch.tensor(source_np)
        source.requires_grad_(True)
        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            output = module(source)
            gradients = torch.autograd.grad(
                (output * torch.tensor(loss_weight_np)).sum(),
                (source, module.weight, module.bias),
            )
            candidate = jt.fetch_sync([output] + list(gradients))

        with jt.flag_scope(use_acl=0, use_cuda=0):
            reference_module = torch.nn.LayerNorm(32)
            reference_module.weight.assign(weight_np)
            reference_module.bias.assign(bias_np)
            reference_source = torch.tensor(source_np)
            reference_source.requires_grad_(True)
            reference_output = reference_module(reference_source)
            reference_gradients = torch.autograd.grad(
                (reference_output * torch.tensor(loss_weight_np)).sum(),
                (
                    reference_source,
                    reference_module.weight,
                    reference_module.bias,
                ),
            )
            reference = jt.fetch_sync(
                [reference_output] + list(reference_gradients)
            )

        for actual, expected in zip(candidate, reference):
            np.testing.assert_allclose(
                actual, expected, rtol=2e-4, atol=2e-4
            )

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_conv2d_without_bias_forward_backward_stays_on_acl(self):
        rng = np.random.RandomState(20260831)
        source_np = rng.randn(2, 3, 8, 8).astype("float32")
        weight_np = rng.randn(5, 3, 3, 3).astype("float32")
        loss_weight_np = rng.randn(2, 5, 8, 8).astype("float32")

        source = torch.tensor(source_np)
        weight = torch.tensor(weight_np)
        source.requires_grad_(True)
        weight.requires_grad_(True)
        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            output = torch.nn.functional.conv2d(
                source, weight, bias=None, padding=1
            )
            gradients = torch.autograd.grad(
                (output * torch.tensor(loss_weight_np)).sum(),
                (source, weight),
            )
            candidate = jt.fetch_sync([output] + list(gradients))

        with jt.flag_scope(use_acl=0, use_cuda=0):
            reference_source = torch.tensor(source_np)
            reference_weight = torch.tensor(weight_np)
            reference_source.requires_grad_(True)
            reference_weight.requires_grad_(True)
            reference_output = torch.nn.functional.conv2d(
                reference_source, reference_weight, bias=None, padding=1
            )
            reference_gradients = torch.autograd.grad(
                (reference_output * torch.tensor(loss_weight_np)).sum(),
                (reference_source, reference_weight),
            )
            reference = jt.fetch_sync(
                [reference_output] + list(reference_gradients)
            )

        for actual, expected in zip(candidate, reference):
            np.testing.assert_allclose(
                actual, expected, rtol=3e-4, atol=3e-4
            )

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_silu_forward_backward_stays_on_acl(self):
        source_np = np.array(
            [-4.0, -1.25, -0.1, 0.0, 0.75, 3.5, 5.9375],
            dtype="float32",
        )
        loss_weight_np = np.array(
            [0.25, -0.5, 1.5, 2.0, -1.0, 0.75, 1.0], dtype="float32"
        )

        self.assertTrue(jt.flags.use_acl)
        self.assertTrue(jt.flags.use_cuda)
        source = torch.tensor(source_np)
        source.requires_grad_(True)
        output = torch.nn.functional.silu(source)
        self.assertIs(type(output), torch.Tensor)
        gradient = torch.autograd.grad(
            (output * torch.tensor(loss_weight_np)).sum(), source
        )[0]
        candidate = _fetch_acl(self, [output, gradient])

        with jt.flag_scope(use_acl=0, use_cuda=0):
            reference_source = torch.tensor(source_np)
            reference_source.requires_grad_(True)
            reference_output = torch.nn.functional.silu(reference_source)
            reference_gradient = torch.autograd.grad(
                (reference_output * torch.tensor(loss_weight_np)).sum(),
                reference_source,
            )[0]
            reference = jt.fetch_sync([reference_output, reference_gradient])

        for actual, expected in zip(candidate, reference):
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

        with jt.flag_scope(use_acl=1, use_cuda=1):
            self.assertTrue(jt.flags.use_acl)
            self.assertTrue(jt.flags.use_cuda)
            source_bf = torch.tensor(source_np, dtype=torch.bfloat16)
            source_bf.requires_grad_(True)
            loss_weight_bf = torch.tensor(loss_weight_np, dtype=torch.bfloat16)
            self.assertEqual(str(source_bf.dtype).replace("torch.", ""), "bfloat16")
            self.assertEqual(str(loss_weight_bf.dtype).replace("torch.", ""), "bfloat16")
            output_bf = torch.nn.functional.silu(source_bf)
            output_bf.sync()
            gradient_bf = torch.autograd.grad(
                (output_bf * loss_weight_bf).sum(), source_bf
            )[0]
            self.assertEqual(str(output_bf.dtype).replace("torch.", ""), "bfloat16")
            self.assertEqual(str(gradient_bf.dtype).replace("torch.", ""), "bfloat16")
            bf_values = _fetch_acl(self, [output_bf, gradient_bf])

        expected_output_bf = np.asarray(
            [-0.07177734375, -0.279296875, -0.047607421875,
             0.0, 0.5078125, 3.390625, 5.90625],
            dtype=np.float32,
        )
        expected_gradient_bf = np.asarray(
            [-0.01318359375, -0.0031585693359375, 0.67578125,
             1.0, -0.84375, 0.80078125, 1.015625],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(bf_values[0], expected_output_bf)
        np.testing.assert_array_equal(bf_values[1], expected_gradient_bf)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_sdpa_forward_backward_stays_on_acl(self):
        rng = np.random.RandomState(20260830)
        shape = (2, 1, 64, 32)
        query_np, key_np, value_np, loss_weight_np = (
            rng.randn(*shape).astype("float32") * 0.1 for _ in range(4)
        )

        self.assertTrue(jt.flags.use_acl)
        self.assertTrue(jt.flags.use_cuda)
        inputs = [
            torch.tensor(value) for value in (query_np, key_np, value_np)
        ]
        for value in inputs:
            value.requires_grad_(True)
        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            output = torch.nn.functional.scaled_dot_product_attention(
                *inputs, dropout_p=0.0, is_causal=False
            )
            grads = torch.autograd.grad(
                (output * torch.tensor(loss_weight_np)).sum(), inputs
            )
            candidate = jt.fetch_sync([output] + list(grads))

        with jt.flag_scope(use_acl=0, use_cuda=0):
            reference_inputs = [
                torch.tensor(value) for value in (query_np, key_np, value_np)
            ]
            for value in reference_inputs:
                value.requires_grad_(True)
            reference_output = torch.nn.functional.scaled_dot_product_attention(
                *reference_inputs, dropout_p=0.0, is_causal=False
            )
            reference_grads = torch.autograd.grad(
                (reference_output * torch.tensor(loss_weight_np)).sum(),
                reference_inputs,
            )
            reference = jt.fetch_sync(
                [reference_output] + list(reference_grads)
            )

        for actual, expected in zip(candidate, reference):
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_sdpa_causal_and_additive_backward_stay_on_acl(self):
        rng = np.random.RandomState(20260831)
        shape = (2, 2, 8, 32)
        query_np, key_np, value_np, loss_weight_np = (
            rng.randn(*shape).astype("float32") * 0.1 for _ in range(4)
        )
        additive_np = rng.randn(shape[-2], shape[-2]).astype("float32") * 0.05
        candidates = []
        native_dispatches = []
        acl_attention = registered_kernel("nn.scaled_dot_product_attention", "acl")
        self.assertIsNotNone(acl_attention)

        def record_attention(*args, **kwargs):
            result = acl_attention(*args, **kwargs)
            native_dispatches.append(result is not None)
            return result

        with override_kernel("nn.scaled_dot_product_attention", "acl", record_attention):
            with jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100"
            ) as logs:
                for is_causal, mask_np in (
                    (True, None),
                    (False, additive_np),
                ):
                    inputs = [
                        torch.tensor(value)
                        for value in (query_np, key_np, value_np)
                    ]
                    for value in inputs:
                        value.requires_grad_(True)
                    mask = (
                        None if mask_np is None
                        else torch.tensor(mask_np).requires_grad_(False)
                    )
                    output = torch.nn.functional.scaled_dot_product_attention(
                        *inputs,
                        attn_mask=mask,
                        dropout_p=0.0,
                        is_causal=is_causal,
                    )
                    grads = torch.autograd.grad(
                        (output * torch.tensor(loss_weight_np)).sum(), inputs
                    )
                    candidates.append(jt.fetch_sync([output] + list(grads)))

        self.assertEqual(native_dispatches, [True, True])
        trainable_mask = torch.tensor(additive_np)
        trainable_mask.requires_grad_(True)
        self.assertIsNone(
            acl_attention(
                *(torch.tensor(value) for value in (
                    query_np, key_np, value_np
                )),
                attn_mask=trainable_mask,
            )
        )
        references = []
        with jt.flag_scope(use_acl=0, use_cuda=0):
            for is_causal, mask_np in (
                (True, None),
                (False, additive_np),
            ):
                inputs = [
                    torch.tensor(value)
                    for value in (query_np, key_np, value_np)
                ]
                for value in inputs:
                    value.requires_grad_(True)
                mask = None if mask_np is None else torch.tensor(mask_np)
                output = torch.nn.functional.scaled_dot_product_attention(
                    *inputs,
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )
                grads = torch.autograd.grad(
                    (output * torch.tensor(loss_weight_np)).sum(), inputs
                )
                references.append(jt.fetch_sync([output] + list(grads)))

        for candidate, reference in zip(candidates, references):
            for actual, expected in zip(candidate, reference):
                np.testing.assert_allclose(
                    actual, expected, rtol=3e-5, atol=3e-5
                )

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_relu_inplace_argument_stays_on_acl(self):
        source = torch.tensor([-2.0, -0.5, 1.0, 3.0], dtype=torch.float32)
        source.requires_grad_(True)
        relu = torch.nn.ReLU(inplace=True)
        leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            output = (
                relu(source)
                + torch.nn.functional.relu(source, inplace=True)
                + leaky_relu(source)
                + torch.nn.functional.leaky_relu(
                    source, negative_slope=0.2, inplace=True
                )
            )
            gradient = torch.autograd.grad(output.sum(), source)[0]
            self.assertTrue(output.is_cuda)
            self.assertTrue(gradient.is_cuda)
            output, gradient = jt.fetch_sync([output, gradient])

        self.assertTrue(relu.inplace)
        self.assertTrue(leaky_relu.inplace)
        self.assertEqual(leaky_relu.negative_slope, 0.2)
        np.testing.assert_allclose(output, [-0.8, -0.2, 4.0, 12.0])
        np.testing.assert_allclose(gradient, [0.4, 0.4, 4.0, 4.0])

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_empty_native_shapes_stay_on_device(self):
        native_shape = torch.ones((2, 3)).shape
        for shape in ((2, 3), [2, 3], native_shape):
            value = torch.empty(shape)
            value.sync()
            self.assertEqual(tuple(value.shape), (2, 3))
            self.assertTrue(value.is_cuda)
            self.assertEqual(value.location(), "device")

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_empty_cuda_tensor(self):
        device = torch.device("cuda")
        empty = torch.tensor([], dtype=torch.float32, device=device)
        self.assertEqual(empty.numel(), 0)
        self.assertTrue(empty.is_cuda)

        value = torch.tensor([1.0], dtype=torch.float32, device=device)
        joined = torch.cat((empty, value))
        np.testing.assert_array_equal(joined.cpu().numpy(), [1.0])

    def test_default_device_follows_execution_flag(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks
        # CPU branches exercise metadata only; never allocate a CPU tensor.
        with jt.flag_scope(use_acl=0, use_cuda=0):
            self.assertEqual(torch.get_default_device().type, "cpu")
        with jt.flag_scope(use_acl=1, use_cuda=1), forbid_backend_fallbacks():
            previous = torch.get_default_device()
            try:
                torch.set_default_device("npu:0")
                self.assertEqual(str(torch.get_default_device()), "npu:0")
                self.assertIs(torch.get_device_module(), torch.npu)
                value = torch.ones((2, 3))
                _assert_acl_device(self, value)
                with torch.device("meta"):
                    self.assertEqual(torch.get_default_device().type, "meta")
                    self.assertTrue(torch.empty((2, 3)).is_meta)
                    self.assertEqual(value.device.type, "npu")
                    with torch.device("npu:0"):
                        self.assertEqual(str(torch.get_default_device()), "npu:0")
                    self.assertEqual(torch.get_default_device().type, "meta")
                with torch.device("cpu"):
                    self.assertEqual(torch.get_default_device().type, "cpu")
                self.assertEqual(str(torch.get_default_device()), "npu:0")
                torch.set_default_device("cpu")
                self.assertEqual(torch.get_default_device().type, "cpu")
                torch.set_default_device(None)
                self.assertEqual(torch.get_default_device().type, "cpu")
                torch.set_default_device("npu")
                self.assertEqual(str(torch.get_default_device()), "npu:0")
                _assert_acl_device(self, value)
            finally:
                torch.set_default_device(previous)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_constant_pad_forward_backward_stays_on_acl(self):
        acl_pad = registered_kernel("nn.constant_pad", "acl")
        self.assertIsNotNone(acl_pad)
        calls = []

        def record_acl_pad(x, amounts, value):
            calls.append((tuple(amounts), value))
            return acl_pad(x, amounts, value)

        with override_kernel("nn.constant_pad", "acl", record_acl_pad):
            with jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100"
            ) as logs:
                labels = torch.tensor([[1, 2, 3]], dtype=torch.int64)
                shifted = torch.nn.functional.pad(
                    labels, (0, 1), value=-100
                )[:, 1:]

                source = torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
                )
                source.requires_grad_(True)
                padded = torch.nn.functional.pad(
                    source, (1, 2, 2, 1), value=3.5
                )
                weight = torch.arange(
                    padded.numel(), dtype=torch.float32
                ).reshape(padded.shape)
                gradient = torch.autograd.grad(
                    (padded * weight).sum(), source
                )[0]
                self.assertTrue(shifted.is_cuda)
                self.assertTrue(padded.is_cuda)
                self.assertTrue(gradient.is_cuda)
                shifted, padded, gradient = jt.fetch_sync(
                    [shifted, padded, gradient]
                )

        np.testing.assert_array_equal(shifted, [[2, 3, -100]])
        np.testing.assert_array_equal(
            padded,
            [
                [3.5, 3.5, 3.5, 3.5, 3.5],
                [3.5, 3.5, 3.5, 3.5, 3.5],
                [3.5, 1.0, 2.0, 3.5, 3.5],
                [3.5, 3.0, 4.0, 3.5, 3.5],
                [3.5, 3.5, 3.5, 3.5, 3.5],
            ],
        )
        np.testing.assert_array_equal(gradient, [[11.0, 12.0], [16.0, 17.0]])

        self.assertEqual(calls, [((0, 1), -100), ((1, 2, 2, 1), 3.5)])

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_meta_factory_has_no_storage_and_preserves_gradient_metadata(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            jt.sync_all(True)
            with jt.flag_scope(use_stat_allocator=1):
                calls = jt.flags.stat_allocator_total_alloc_call
                allocated = jt.flags.stat_allocator_total_alloc_byte
                huge = torch.empty((1000000, 1000000), dtype=torch.float32, device="meta")
                self.assertEqual(tuple(huge.shape), (1000000, 1000000))
                self.assertTrue(huge.is_meta)
                self.assertTrue(huge.is_metadata)
                self.assertEqual(huge.location(), "meta")
                self.assertEqual(huge.placement_backend, -2)
                self.assertEqual(huge.device_id, -1)
                self.assertFalse(huge.requires_grad)

                # No keyword arguments exercises the factory's fast path.
                with torch.device("meta"):
                    value = torch.empty((2, 3))
                parameter = torch.nn.Parameter(value)
                self.assertTrue(value.is_meta)
                self.assertFalse(value.requires_grad)
                self.assertTrue(parameter.is_meta)
                self.assertTrue(parameter.requires_grad)
                for tensor, trainable in ((value, False), (parameter, True)):
                    cloned, detached = tensor.clone(), tensor.detach()
                    self.assertTrue(cloned.is_meta)
                    self.assertTrue(detached.is_meta)
                    self.assertEqual(bool(cloned.requires_grad), trainable)
                    self.assertFalse(detached.requires_grad)
                    self.assertEqual(tuple(cloned.shape), (2, 3))
                    self.assertEqual(cloned.dtype, tensor.dtype)

                # Global draining skips metadata holders, including the huge one.
                jt.sync_all(True)
                with self.assertRaisesRegex(RuntimeError, "metadata|meta"):
                    value.numpy()
                with self.assertRaisesRegex(NotImplementedError, "meta"):
                    value.to("npu")
                with self.assertRaisesRegex(RuntimeError, "metadata|meta"):
                    _ = value + value
                self.assertEqual(jt.flags.stat_allocator_total_alloc_call, calls)
                self.assertEqual(jt.flags.stat_allocator_total_alloc_byte, allocated)
                with self.assertRaisesRegex(RuntimeError, "default.*SFRL|unsupported"):
                    torch.npu.memory_allocated()
            # An unused descriptor must not invalidate the default pool history.
            self.assertGreaterEqual(torch.npu.memory_allocated(), 0)
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_meta_default_device_does_not_relabel_real_npu_tensor(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            real = torch.ones((2, 3), device="npu")
            _assert_acl_device(self, real)
            with torch.device("meta"):
                self.assertEqual(real.device.type, "npu")
                self.assertFalse(real.is_meta)
                metadata = torch.empty((2, 3))
                explicitly_real = torch.empty((2, 3), device="npu")
            self.assertTrue(metadata.is_meta)
            self.assertFalse(metadata.requires_grad)
            self.assertFalse(explicitly_real.requires_grad)
            self.assertEqual(explicitly_real.device.type, "npu")
            _assert_acl_device(self, explicitly_real)
            discarded = real.to("meta")
            self.assertTrue(discarded.is_meta)
            self.assertEqual(tuple(discarded.shape), tuple(real.shape))
            self.assertEqual(discarded.dtype, real.dtype)
            self.assertEqual(real.device.type, "npu")
            _assert_acl_device(self, real)
            np.testing.assert_array_equal(real.detach().cpu().numpy(), np.ones((2, 3)))
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_meta_load_state_assign_adopts_npu_dtype_and_preserves_trainability(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            for trainable in (False, True):
                with self.subTest(trainable=trainable):
                    module = torch.nn.Module()
                    original = torch.nn.Parameter(
                        torch.empty((2, 3), dtype=torch.float16, device="meta"),
                        requires_grad=trainable)
                    module.register_parameter("weight", original)
                    module.register_buffer("buffer", torch.empty((2,), dtype=torch.float16, device="meta"))
                    weights = {
                        "weight": torch.ones((2, 3), dtype=torch.float32, device="npu"),
                        "buffer": torch.zeros((2,), dtype=torch.float32, device="npu"),
                    }
                    result = module.load_state_dict(weights, assign=True)
                    self.assertFalse(result.missing_keys)
                    self.assertFalse(result.unexpected_keys)
                    self.assertIsNot(module.weight, original)
                    self.assertTrue(original.is_meta)
                    self.assertEqual(bool(module.weight.requires_grad), trainable)
                    self.assertFalse(module.buffer.requires_grad)
                    for name in ("weight", "buffer"):
                        tensor = getattr(module, name)
                        self.assertFalse(tensor.is_meta)
                        self.assertEqual(tensor.device.type, "npu")
                        self.assertEqual(tensor.dtype, torch.float32)
                        _assert_acl_device(self, tensor)
                        difference = (tensor - weights[name]).abs().sum()
                        _assert_acl_device(self, difference)
                        self.assertEqual(difference.item(), 0)
                    if trainable:
                        (module.weight * 2).sum().backward()
                        self.assertIsNotNone(module.weight.grad)
                        _assert_acl_device(self, module.weight.grad)
                        np.testing.assert_array_equal(
                            module.weight.grad.detach().cpu().numpy(), np.full((2, 3), 2.0))
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_meta_load_state_without_assign_warns_and_does_not_materialize(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            module = torch.nn.Module()
            original = torch.nn.Parameter(torch.empty((2, 3), device="meta"))
            module.register_parameter("weight", original)
            weight = torch.ones((2, 3), device="npu")
            _assert_acl_device(self, weight)
            with self.assertWarnsRegex(UserWarning, "meta"):
                result = module.load_state_dict({"weight": weight}, assign=False)
            self.assertFalse(result.missing_keys)
            self.assertFalse(result.unexpected_keys)
            self.assertIs(module.weight, original)
            self.assertTrue(module.weight.is_meta)
            self.assertTrue(module.weight.requires_grad)
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_meta_to_empty_preserves_ties_and_obeys_recurse(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            module = torch.nn.Module()
            tied = torch.nn.Parameter(torch.empty((2, 3), device="meta"))
            module.register_parameter("first", tied)
            module.register_parameter("second", tied)
            module.register_buffer("buffer", torch.empty((2,), device="meta"))
            module.child = torch.nn.Module()
            module.child.register_parameter("weight", torch.nn.Parameter(
                torch.empty((2,), device="meta"), requires_grad=False))
            child_original = module.child.weight
            self.assertIs(module.to_empty(device="npu", recurse=False), module)
            self.assertIs(module.first, module.second)
            self.assertTrue(module.first.requires_grad)
            self.assertIs(module.child.weight, child_original)
            self.assertTrue(module.child.weight.is_meta)
            _assert_acl_device(self, module.first)
            _assert_acl_device(self, module.buffer)
            self.assertIs(module.to_empty(device="npu", recurse=True), module)
            self.assertIs(module.first, module.second)
            self.assertTrue(module.first.requires_grad)
            self.assertFalse(module.child.weight.requires_grad)
            for tensor in (module.first, module.buffer, module.child.weight):
                self.assertEqual(tensor.device.type, "npu")
                _assert_acl_device(self, tensor)
                # to_empty has unspecified contents: write before reading.
                with torch.no_grad():
                    tensor.fill_(2)
                _assert_acl_device(self, tensor)
                self.assertEqual(tensor.sum().item(), 2 * tensor.numel())
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_npu_public_identity_matches_active_acl_backend(self):
        import sys
        import torch_npu
        from jittor._runtime.fallback import forbid_backend_fallbacks

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            self.assertTrue(torch.npu.is_available())
            self.assertFalse(torch.cuda.is_available())
            self.assertEqual(torch.accelerator.current_accelerator().type, "npu")
            count, current = torch.npu.device_count(), torch.npu.current_device()
            self.assertGreater(count, 0)
            self.assertGreaterEqual(current, 0)
            self.assertLess(current, count)
            self.assertTrue(torch_npu.__jittor_acl_facade__)
            self.assertNotIn("torch_npu._C", sys.modules)
            with self.assertRaises(NotImplementedError):
                torch_npu.npu_fusion_attention(None, None, None, 1, "BSND")
            value = torch.ones((2,), device="npu")
            _assert_acl_device(self, value)
            self.assertEqual(value.device.type, "npu")
            self.assertEqual(value.device_id, current)
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_npu_deterministic_policy_toggles_backend_and_restores_state(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        previous = torch.are_deterministic_algorithms_enabled()
        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            try:
                for enabled in (True, False, True):
                    torch.use_deterministic_algorithms(enabled)
                    self.assertEqual(torch.are_deterministic_algorithms_enabled(), enabled)
                    self.assertEqual(bool(jt.core.backend_get_deterministic_algorithms()), enabled)
                    value = torch.ones((4,), device="npu").sum()
                    _assert_acl_device(self, value)
                    self.assertEqual(value.item(), 4)
                # Reject unsupported policy without silently changing it.
                with self.assertRaisesRegex(NotImplementedError, "warn_only"):
                    torch.use_deterministic_algorithms(False, warn_only=True)
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
            finally:
                torch.use_deterministic_algorithms(previous)
            self.assertEqual(torch.are_deterministic_algorithms_enabled(), previous)
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1, use_sfrl_allocator=1,
                   use_nfef_allocator=0, use_stat_allocator=0,
                   use_cuda_managed_allocator=0)
    def test_npu_default_sfrl_event_peaks_lifecycle(self):
        """Real NPU default SFRL pool metrics; excludes workspace/driver bytes."""
        import gc
        from jittor._runtime.fallback import forbid_backend_fallbacks

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            index = torch.npu.current_device()
            device = "npu:{}".format(index)

            def drain_python_refs():
                gc.collect()
                torch.npu.synchronize(device=index)

            def current_bytes():
                return (torch.npu.memory_allocated(device=index),
                        torch.npu.memory_reserved(device=index))

            drain_python_refs()
            torch.npu.empty_cache()
            baseline_live, baseline_reserved = current_bytes()
            torch.npu.reset_peak_memory_stats(device=index)
            value = torch.ones(262144, dtype=torch.float32, device=device)
            _assert_acl_device(self, value)
            self.assertEqual(value.device_id, index)
            allocated_live, allocated_reserved = current_bytes()
            self.assertGreaterEqual(allocated_live - baseline_live, 1048576)
            pointer = value.data_ptr()
            del value
            drain_python_refs()  # Preserve cached blocks for the reuse check.
            freed_live, cached_reserved = current_bytes()
            self.assertEqual(freed_live, baseline_live)
            self.assertEqual(cached_reserved, allocated_reserved)
            # First peak reads occur AFTER the real allocation was released.
            # Query-time max(live) would miss precisely this high-water event.
            peak_live = torch.npu.max_memory_allocated(device=index)
            peak_reserved = torch.npu.max_memory_reserved(device=index)
            self.assertGreaterEqual(peak_live, allocated_live)
            self.assertGreaterEqual(peak_reserved, allocated_reserved)
            self.assertGreater(peak_live, freed_live)

            # Reusing an equal-sized cached block must not grow reservation or
            # inflate peaks. The real tensor pointer proves actual reuse.
            value = torch.ones(262144, dtype=torch.float32, device=device)
            _assert_acl_device(self, value)
            self.assertEqual(value.data_ptr(), pointer)
            self.assertEqual(current_bytes(), (allocated_live, allocated_reserved))
            self.assertEqual(torch.npu.max_memory_allocated(device=index), peak_live)
            self.assertEqual(torch.npu.max_memory_reserved(device=index), peak_reserved)

            # Reset while storage is occupied resets each peak to current bytes,
            # and neither releases the allocation nor fabricates zero usage.
            torch.npu.reset_peak_memory_stats(device=index)
            reset_live, reset_reserved = current_bytes()
            self.assertGreater(reset_live, baseline_live)
            self.assertEqual(torch.npu.max_memory_allocated(device=index), reset_live)
            self.assertEqual(torch.npu.max_memory_reserved(device=index), reset_reserved)

            # A distinct view shares real device storage. Dropping one owner
            # must not release its bytes; dropping the final owner must do so.
            shared = value.view(256, 1024)
            _assert_acl_device(self, shared)
            self.assertIsNot(shared, value)
            self.assertEqual(shared.data_ptr(), value.data_ptr())
            self.assertEqual(current_bytes(), (reset_live, reset_reserved))
            del value
            drain_python_refs()
            self.assertEqual(current_bytes(), (reset_live, reset_reserved))
            del shared
            drain_python_refs()
            self.assertEqual(current_bytes(), (baseline_live, reset_reserved))
            self.assertEqual(torch.npu.max_memory_allocated(device=index), reset_live)
            self.assertEqual(torch.npu.max_memory_reserved(device=index), reset_reserved)
            torch.npu.empty_cache()
            self.assertEqual(current_bytes(), (baseline_live, baseline_reserved))
            self.assertEqual(torch.npu.max_memory_allocated(device=index), reset_live)
            self.assertEqual(torch.npu.max_memory_reserved(device=index), reset_reserved)
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_npu_truth_reduce_dispatch_and_isin(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            values = torch.tensor([[0, 1, 0], [1, 1, 1]], dtype=torch.int64, device="npu")
            for operation, expected in ((jt.any, [[True], [True]]),
                                        (jt.all, [[False], [True]])):
                reduced = operation(values, dim=1, keepdims=True)
                _assert_acl_device(self, reduced)
                self.assertEqual(tuple(reduced.shape), (2, 1))
                np.testing.assert_array_equal(reduced.detach().cpu().numpy(), expected)
            elements = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="npu")
            selected = torch.isin(elements, torch.tensor([1, 3], dtype=torch.int64, device="npu"))
            _assert_acl_device(self, selected)
            np.testing.assert_array_equal(selected.detach().cpu().numpy(), [False, True, False, True])
            self.assertTrue(selected.any().item())
            self.assertFalse(selected.all().item())
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_npu_bool_slice_assignment_uses_exact_device_payload(self):
        """CANN bool slice writes lower via exact int8 payload, no host fallback."""
        from jittor._runtime.fallback import forbid_backend_fallbacks
        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            scalar = torch.zeros((1, 1), dtype=torch.bool, device="npu")
            scalar[0, 0] = True
            self.assertEqual(scalar.dtype, torch.bool)
            np.testing.assert_array_equal(_fetch_acl(self, [scalar])[0], [[True]])

            column = torch.tensor([[True, False, True], [False, True, False]],
                                  dtype=torch.bool, device="npu")
            column[:, 1] = torch.tensor([True, False], dtype=torch.bool, device="npu")
            self.assertEqual(column.dtype, torch.bool)
            np.testing.assert_array_equal(_fetch_acl(self, [column])[0],
                                          [[True, True, True], [False, False, False]])

            broadcast = torch.zeros((3, 3), dtype=torch.bool, device="npu")
            broadcast[1:, :] = torch.tensor([True, False, True], dtype=torch.bool, device="npu")
            np.testing.assert_array_equal(_fetch_acl(self, [broadcast])[0],
                                          [[False, False, False], [True, False, True], [True, False, True]])

            stepped = torch.ones((2, 4), dtype=torch.bool, device="npu")
            stepped[:, ::2] = False
            np.testing.assert_array_equal(_fetch_acl(self, [stepped])[0],
                                          [[False, True, False, True], [False, True, False, True]])

            # Casting to int8 before truth conversion would turn +/-256 into 0.
            numeric = torch.zeros((1, 3), dtype=torch.bool, device="npu")
            numeric[:, :] = torch.tensor([[-256, 0, 256]], dtype=torch.int32, device="npu")
            self.assertEqual(numeric.dtype, torch.bool)
            np.testing.assert_array_equal(_fetch_acl(self, [numeric])[0], [[True, False, True]])

            narrow = torch.tensor([[-128, 0, 127]], dtype=torch.int8, device="npu")
            narrow[:, 1:2] = torch.tensor([[-1]], dtype=torch.int8, device="npu")
            self.assertEqual(narrow.dtype, torch.int8)
            np.testing.assert_array_equal(_fetch_acl(self, [narrow])[0], [[-128, -1, 127]])
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)
    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_adamw_public_group_state_and_fresh_restore(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            def snapshot(*values):
                # Host fetch changes residency: observe independent copies so
                # the test cannot move live optimizer parameters or state.
                for value in values:
                    _assert_acl_device(self, value)
                copies = [value.detach().clone() for value in values]
                result = _fetch_acl(self, copies)
                for value in values:
                    _assert_acl_device(self, value)
                return result

            device = torch.device("npu:0")
            p = torch.nn.Parameter(torch.tensor([1., -2.], device=device))
            q = torch.nn.Parameter(torch.tensor([.25, -.75], device=device))
            metadata = {"m": "user-m", "grads": "user-grads",
                        "_torch_steps": "user-steps", "n_step": "user-counter"}
            opt = torch.optim.AdamW([dict(params=[p], **metadata)], lr=.01,
                                  betas=(.8, .9), eps=1e-6, weight_decay=.03)
            self.assertEqual(len(opt.state), 0)
            self.assertEqual(opt.state_dict()["state"], {})
            p.grad = torch.tensor([.5, -.25], device=device)
            opt.step()
            held = opt.state[p]["step"]
            self.assertTrue(torch.is_tensor(held))
            self.assertEqual(tuple(held.shape), ())
            self.assertEqual(held.dtype, torch.float32)
            self.assertEqual(held.device.type, "cpu")
            held.sync()
            self.assertEqual(held.location(), "cpu")
            first_parameter = snapshot(p)[0].copy()
            opt.zero_grad(set_to_none=True)
            self.assertIsNone(p.grad)
            self.assertEqual(opt.param_groups[0]["grads"], "user-grads")
            opt.step()
            self.assertEqual(held.item(), 1.)
            np.testing.assert_array_equal(snapshot(p)[0], first_parameter)
            held.fill_(4.)
            opt.state[p]["exp_avg"].fill_(.125)
            np.testing.assert_array_equal(snapshot(opt.state[p]["exp_avg"])[0], [.125, .125])
            opt.param_groups[0]["lr"] = .02
            opt.add_param_group({"params": [q], "lr": .005})
            self.assertEqual(opt.state.get(q, {}), {})
            p.grad = torch.tensor([-.125, .375], device=device)
            q.grad = torch.tensor([.25, -.5], device=device)
            opt.step()
            self.assertIs(opt.state[p]["step"], held)
            self.assertEqual(held.item(), 5.)
            self.assertEqual(opt.state[q]["step"].item(), 1.)
            # Zero is also a legal initialized step value, not absence of state.
            held.fill_(0.)
            self.assertEqual(set(opt.state[p]), {"step", "exp_avg", "exp_avg_sq"})
            held.fill_(5.)
            checkpoint = opt.state_dict()
            self.assertIs(checkpoint["state"][0]["step"], held)
            for key, value in metadata.items():
                self.assertEqual(checkpoint["param_groups"][0][key], value)
            for group in opt.param_groups:
                self.assertNotIn("values", group)
                for param in group["params"]:
                    _assert_acl_device(self, param)
                    state = opt.state[param]
                    for key in ("exp_avg", "exp_avg_sq"):
                        value = state[key]
                        self.assertTrue(torch.is_tensor(value))
                        self.assertEqual(value.device, param.device)
                        self.assertEqual(value.dtype, param.dtype)
                        self.assertEqual(tuple(value.shape), tuple(param.shape))
                        _assert_acl_device(self, value)
            r = torch.nn.Parameter(p.detach().clone())
            s = torch.nn.Parameter(q.detach().clone())
            fresh = torch.optim.AdamW([{"params": [r]}, {"params": [s]}], lr=.9)
            fresh.load_state_dict(checkpoint)
            # Loader owns cloned moment tensors: later updates cannot alias.
            self.assertIsNot(fresh.state[r]["exp_avg"], opt.state[p]["exp_avg"])
            self.assertEqual(fresh.param_groups[0]["lr"], .02)
            for param in (r, s):
                restored_step = fresh.state[param]["step"]
                self.assertEqual(restored_step.dtype, torch.float32)
                self.assertEqual(restored_step.device.type, "cpu")
                restored_step.sync()
                self.assertEqual(restored_step.location(), "cpu")
            for a, b in zip((p, q), (r, s)):
                a.grad = torch.tensor([.125, -.0625], device=device)
                b.grad = a.grad.clone()
            opt.step()
            fresh.step()
            for a, b in zip((p, q), (r, s)):
                av, bv = snapshot(a, b)
                np.testing.assert_array_equal(av, bv)
                for key in ("exp_avg", "exp_avg_sq"):
                    av, bv = snapshot(opt.state[a][key], fresh.state[b][key])
                    np.testing.assert_array_equal(av, bv)
                self.assertEqual(opt.state[a]["step"].item(), fresh.state[b]["step"].item())
            for key, value in metadata.items():
                self.assertEqual(fresh.param_groups[0][key], value)
            fresh.zero_grad(set_to_none=False)
            for param in (r, s):
                self.assertIsNotNone(param.grad)
                np.testing.assert_array_equal(snapshot(param.grad)[0], [0., 0.])
            for key, value in metadata.items():
                self.assertEqual(fresh.param_groups[0][key], value)
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_npu_single_bool_index_with_full_slices(self):
        """Swift labels[:, loss_mask] selects device coordinates, not host data."""
        from jittor._runtime.fallback import forbid_backend_fallbacks
        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            labels = torch.tensor([[-100, 11, -100, 13]], dtype=torch.int64, device="npu")
            mask = (labels != -100)[0]
            selected = labels[:, mask]
            self.assertEqual(tuple(selected.shape), (1, 2))
            self.assertEqual(selected.dtype, torch.int64)
            np.testing.assert_array_equal(_fetch_acl(self, [selected])[0], [[11, 13]])
            values = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]],
                                  device="npu", requires_grad=True)
            gathered = values[..., mask]
            gathered.sum().backward()
            np.testing.assert_array_equal(_fetch_acl(self, [gathered])[0], [[2., 4.], [6., 8.]])
            np.testing.assert_array_equal(_fetch_acl(self, [values.grad])[0],
                                          [[0., 1., 0., 1.], [0., 1., 0., 1.]])
            full = values[:, torch.ones(4, dtype=torch.bool, device="npu")]
            np.testing.assert_array_equal(_fetch_acl(self, [full])[0], [[1., 2., 3., 4.], [5., 6., 7., 8.]])
            empty = values[:, torch.zeros(4, dtype=torch.bool, device="npu")]
            self.assertEqual(tuple(empty.shape), (2, 0))
            self.assertEqual(empty.dtype, values.dtype)
            empty_gradient = torch.autograd.grad(empty.sum(), values)[0]
            np.testing.assert_array_equal(_fetch_acl(self, [empty_gradient])[0],
                                          [[0., 0., 0., 0.], [0., 0., 0., 0.]])
            with self.assertRaisesRegex(IndexError, "boolean index length"):
                _ = values[:, torch.ones(3, dtype=torch.bool, device="npu")]
            # Preserve negative/repeated integer index behavior and accumulation.
            other = torch.tensor([[1., 2., 3., 4.]], device="npu", requires_grad=True)
            repeated = other[:, torch.tensor([-1, 1, 1], dtype=torch.int64, device="npu")]
            repeated.sum().backward()
            np.testing.assert_array_equal(_fetch_acl(self, [repeated])[0], [[4., 2., 2.]])
            np.testing.assert_array_equal(_fetch_acl(self, [other.grad])[0], [[0., 2., 0., 1.]])
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_npu_leaf_hooks_preserve_identity_and_local_gradient(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks
        before_count = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            parameter = torch.nn.Parameter(torch.tensor([2.], device="npu:0"))
            optimizer = torch.optim.SGD([parameter], lr=.1)
            parameter._jittor_ddp_order = 17
            parameter._jittor_ddp_state = None
            def identity():
                return (id(parameter), parameter.is_leaf, parameter.is_backward_leaf,
                        parameter.grad_fn, parameter.requires_grad,
                        parameter._jittor_ddp_order, id(parameter._jittor_ddp_state))
            original = identity()
            self.assertTrue(original[1] and original[2])
            # The graph already exists before registration: hooks must still run.
            loss = (parameter * 2 + parameter * 3).sum()
            events = []
            def observe(g):
                events.append(("observe", g.detach().clone()))
                return None
            def scale(g):
                events.append(("scale", g.detach().clone()))
                return g * 2
            def shift(g):
                events.append(("shift", g.detach().clone()))
                return g + 1
            observer = parameter.register_hook(observe)
            scaling = parameter.register_hook(scale)
            shifting = parameter.register_hook(shift)
            self.assertEqual(identity(), original)
            loss.backward()
            self.assertEqual([name for name, _ in events], ["observe", "scale", "shift"])
            for (_, value), expected in zip(events, (5., 5., 10.)):
                np.testing.assert_array_equal(_fetch_acl(self, [value])[0], [expected])
            np.testing.assert_array_equal(_fetch_acl(self, [parameter.grad.detach().clone()])[0], [11.])
            events.clear()
            (parameter * 3).sum().backward()
            # Hooks observe this backward's local 3, not the accumulated 11+7.
            np.testing.assert_array_equal(_fetch_acl(self, [events[0][1]])[0], [3.])
            np.testing.assert_array_equal(_fetch_acl(self, [parameter.grad.detach().clone()])[0], [18.])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.assertEqual(identity(), original)
            scaling.remove(); scaling.remove(); shifting.remove()
            events.clear()
            (parameter * 4).sum().backward()
            self.assertEqual([name for name, _ in events], ["observe"])
            np.testing.assert_array_equal(_fetch_acl(self, [parameter.grad.detach().clone()])[0], [4.])
            observer.remove(); events.clear()
            # autograd.grad shares the same native callback mechanism; no extra
            # compat-only dispatch and no duplicate processing in backward.
            handle = parameter.register_hook(lambda g: g * 3)
            derivative, = torch.autograd.grad((parameter * 2).sum(), (parameter,))
            np.testing.assert_array_equal(_fetch_acl(self, [derivative.detach().clone()])[0], [6.])
            handle.remove()
            self.assertEqual(identity(), original)
            # Native jt.grad also sees the callback, including a leaf as loss.
            native = jt.array([1.]).start_grad()
            original_leaf = native.is_backward_leaf
            native_hook = native.register_hook(lambda g: g * 7)
            result = jt.grad(native, native)
            self.assertEqual(native.is_backward_leaf, original_leaf)
            np.testing.assert_array_equal(_fetch_acl(self, [result])[0], [7.])
            native_hook.remove()
            # Reject invalid replacement contracts; the device-negative path is
            # a D2H copy only, with no CPU model or numerical computation.
            for violation in ("shape", "dtype", "device"):
                with self.subTest(replacement=violation):
                    leaf = torch.nn.Parameter(torch.tensor([1.], device="npu:0"))
                    def bad_hook(g):
                        if violation == "shape":
                            return g.reshape(1, 1)
                        if violation == "dtype":
                            return g.half()
                        return g.detach().clone().to("cpu")
                    invalid = leaf.register_hook(bad_hook)
                    try:
                        with self.assertRaisesRegex(RuntimeError, "Leaf gradient hook"):
                            torch.autograd.grad((leaf * 2).sum(), (leaf,))
                    finally:
                        invalid.remove()
                    self.assertTrue(leaf.is_leaf and leaf.is_backward_leaf)
                    _assert_acl_device(self, leaf)
            self.assertEqual(jt.core.backend_fallback_count() - before_count, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_npu_optimizer_commit_keeps_leaf_hooks_and_state_policy(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks
        factories = (
            ("sgd-auto", lambda ps: torch.optim.SGD(ps, lr=.01)),
            # The shim SGD constructor does not yet accept fused=. The existing
            # parameter-group switch is read by the native SGD dispatcher.
            ("sgd-portable", lambda ps: torch.optim.SGD(
                [{"params": ps, "fused": False}], lr=.01, momentum=.9)),
            ("adam", lambda ps: torch.optim.Adam(ps, lr=.01)),
            ("adamw-portable", lambda ps: torch.optim.AdamW(ps, lr=.01, fused=False)),
            ("adamw-fused", lambda ps: torch.optim.AdamW(ps, lr=.01, fused=True)),
            ("rmsprop", lambda ps: torch.optim.RMSprop(ps, lr=.01)),
            ("adan", lambda ps: torch.optim.Adan(ps, lr=.01)),
        )
        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            for name, factory in factories:
                with self.subTest(optimizer=name):
                    parameter = torch.nn.Parameter(torch.tensor([1., 2.], device="npu:0"))
                    frozen = torch.nn.Parameter(torch.tensor([5., 6.], device="npu:0"), requires_grad=False)
                    optimizer = factory([parameter, frozen])
                    if name == "sgd-portable":
                        self.assertIs(optimizer.param_groups[0]["fused"], False)
                    parameter_id = id(parameter)
                    events = []
                    handle = parameter.register_hook(lambda g: events.append(g.detach().clone()))
                    for step in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        factor = float(step + 3)
                        (parameter * factor).sum().backward()
                        self.assertEqual(len(events), step + 1)
                        np.testing.assert_array_equal(_fetch_acl(self, [events[-1]])[0], [factor, factor])
                        previous = _fetch_acl(self, [parameter.detach().clone()])[0]
                        optimizer.step()
                        self.assertEqual(id(parameter), parameter_id)
                        self.assertTrue(parameter.requires_grad)
                        self.assertTrue(parameter.is_leaf and parameter.is_backward_leaf)
                        self.assertIsNone(parameter.grad_fn)
                        self.assertFalse(frozen.requires_grad)
                        _assert_acl_device(self, parameter)
                        actual = _fetch_acl(self, [parameter.detach().clone()])[0]
                        self.assertTrue(np.isfinite(actual).all())
                        self.assertTrue(np.any(actual != previous), "optimizer did not update")
                        np.testing.assert_array_equal(_fetch_acl(self, [frozen.detach().clone()])[0], [5., 6.])
                        state = optimizer.state.get(parameter, {})
                        expected_state = {
                            "sgd-auto": set(), "sgd-portable": {"momentum_buffer"},
                            "adam": {"step", "exp_avg", "exp_avg_sq"},
                            "adamw-portable": {"step", "exp_avg", "exp_avg_sq"},
                            "adamw-fused": {"step", "exp_avg", "exp_avg_sq"},
                            "rmsprop": {"step", "square_avg"},
                            "adan": {"step", "exp_avg", "exp_avg_sq", "exp_avg_diff", "pre_grad"},
                        }[name]
                        self.assertEqual(set(state), expected_state)
                        for key, value in state.items():
                            if torch.is_tensor(value):
                                self.assertFalse(value.requires_grad, key)
                                if key != "step":
                                    _assert_acl_device(self, value)
                    handle.remove()
                    optimizer.zero_grad(set_to_none=True)
                    (parameter * 7).sum().backward()
                    self.assertEqual(len(events), 2)
                    np.testing.assert_array_equal(_fetch_acl(self, [parameter.grad.detach().clone()])[0], [7., 7.])
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_npu_serialization_preserves_live_state_dict_residency(self):
        import tempfile
        from pathlib import Path
        import safetensors.torch as st
        import safetensors.numpy as sn
        from jittor._runtime.fallback import forbid_backend_fallbacks

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            module = torch.nn.Linear(2, 1, bias=True, device="npu:0")
            with torch.no_grad():
                module.weight.copy_(torch.tensor([[1.25, -2.5]], device="npu:0"))
                module.bias.fill_(.75)
            live = dict(module.named_parameters())
            identities = {name: id(value) for name, value in live.items()}
            state = module.state_dict()

            def assert_live():
                self.assertEqual(identities, {name: id(value)
                                             for name, value in module.named_parameters()})
                for name, value in live.items():
                    _assert_acl_device(self, value)
                    self.assertTrue(value.requires_grad)
                    self.assertEqual(state[name].data_ptr(), value.data_ptr())

            def assert_payload(values):
                self.assertEqual(set(values), {"weight", "bias"})
                np.testing.assert_array_equal(values["weight"], [[1.25, -2.5]])
                np.testing.assert_array_equal(values["bias"], [.75])

            assert_live()
            # tempfile follows the runner's TMPDIR; never write checkpoints in
            # the repository or a developer-specific absolute directory.
            with tempfile.TemporaryDirectory(prefix="acl-serialization-") as directory:
                path = Path(directory)
                st.save_file(state, str(path / "model.safetensors"))
                assert_live()
                assert_payload(sn.load_file(str(path / "model.safetensors")))
                payload = st.save(state)
                assert_live()
                assert_payload(sn.load(payload))
                torch.save(state, path / "model.pt")
                assert_live()
            # Saving an aliasing state_dict must preserve the next NPU graph.
            module(torch.tensor([[2., 3.]], device="npu:0")).sum().backward()
            expected = {"weight": [[2., 3.]], "bias": [1.]}
            for name, parameter in live.items():
                self.assertIsNotNone(parameter.grad)
                _assert_acl_device(self, parameter.grad)
                snapshot = parameter.grad.detach().clone()
                np.testing.assert_array_equal(_fetch_acl(self, [snapshot])[0], expected[name])
            assert_live()
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)
