"""Torch-grade optimizer / lr-scheduler parity for ``import torch``.

One optimizer step on a known loss (loss = sum(w^2) -> grad = 2w) is checked against the
exact analytic update rule. CPU+CUDA.

Run:  python -m pytest compat/tests/torch/test_torch_compat_optim.py
"""

from _helpers import capability as _test_capability
import unittest
import numpy as np
import torch
import jittor as jt

_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if _test_capability.any_accelerator_enabled(backend=jt) else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=atol, rtol=rtol,
                                   err_msg=msg)


class TestSGD(Base):
    def test_sgd_fp32_multistep_groups_and_missing_gradients(self):
        def body(dev):
            for momentum, dampening, nesterov, decay in (
                    (0.0, 0.0, False, 0.0),
                    (0.9, 0.0, False, 0.2),
                    (0.9, 0.0, True, 0.2),
                    (0.9, 0.3, False, 0.2)):
                refs = [np.array([0.1, -0.3], dtype=np.float64),
                        np.array([0.4, -0.2], dtype=np.float64)]
                params = [torch.tensor(ref.astype(np.float32), device=dev,
                                       requires_grad=True) for ref in refs]
                optimizer = torch.optim.SGD(
                    [{"params": [params[0]], "lr": 0.03},
                     {"params": [params[1]], "lr": 0.07}],
                    lr=0.01, momentum=momentum, dampening=dampening,
                    nesterov=nesterov, weight_decay=decay)
                velocities = [np.zeros(2), np.zeros(2)]
                steps = [0, 0]
                for iteration in range(5):
                    optimizer.zero_grad(set_to_none=True)
                    for index, param in enumerate(params):
                        if index == 1 and iteration in (0, 2):
                            continue
                        gradient = np.array(
                            [0.025 * (iteration + 1), -0.05], dtype=np.float32)
                        param.grad = torch.tensor(gradient, device=dev)
                        dp = gradient.astype(np.float64) + refs[index] * decay
                        if momentum:
                            # Preserve native SGD's first-step dampening policy.
                            velocities[index] = (momentum * velocities[index]
                                                 + dp * (1 - dampening))
                            update = (dp + momentum * velocities[index]
                                      if nesterov else velocities[index])
                        else:
                            update = dp
                        refs[index] -= optimizer.param_groups[index]["lr"] * update
                        steps[index] += 1
                    optimizer.step()
                    for index, param in enumerate(params):
                        label = f"SGD {momentum}/{dampening}/{nesterov} step {iteration} group {index} {dev}"
                        self.ac(param.detach().clone().cpu().numpy(), refs[index],
                                atol=1e-6, rtol=1e-6, msg=label)
                        self.assertEqual(
                            optimizer.param_groups[index]["_torch_steps"][0],
                            steps[index], label)
                        if momentum and steps[index]:
                            state = optimizer.state[param]["momentum_buffer"]
                            self.ac(state.detach().clone().cpu().numpy(),
                                    velocities[index], atol=1e-7, rtol=1e-5,
                                    msg=label)
                        else:
                            self.assertNotIn(param, optimizer.state)
                    for group in optimizer.param_groups:
                        group["lr"] *= 0.8

        both_devices(body)

    def test_sgd_preserves_parameter_subclass_subtraction(self):
        def body(dev):
            calls = []

            class OffsetParameter(torch.nn.Parameter):
                def __sub__(self, other):
                    calls.append(other)
                    return super().__sub__(other) + 0.25

            reference = np.array([0.1, -0.3], dtype=np.float64)
            param = OffsetParameter(
                torch.tensor(reference.astype(np.float32), device=dev))
            optimizer = torch.optim.SGD([param], lr=0.1)
            for step in range(1, 4):
                optimizer.zero_grad(set_to_none=True)
                param.grad = torch.ones(2, device=dev)
                optimizer.step()
                reference += 0.25 - 0.1
                self.ac(param.detach().clone().cpu().numpy(), reference,
                        atol=1e-6, rtol=1e-6)
                self.assertEqual(len(calls), step)

        both_devices(body)

    def test_sgd_plain(self):
        w0 = np.array([1., 2., 3.], "float32")
        def body(dev):
            w = jt.array(w0)
            opt = torch.optim.SGD([w], lr=0.1)
            opt.step((w * w).sum())              # grad = 2w
            self.ac(w.numpy(), w0 - 0.1 * 2 * w0, msg=f"sgd {dev}")
        both_devices(body)

    def test_sgd_weight_decay(self):
        w0 = np.array([1., 2., 3.], "float32")
        def body(dev):
            w = jt.array(w0)
            opt = torch.optim.SGD([w], lr=0.1, weight_decay=0.5)
            opt.step((w * w).sum())              # effective grad = 2w + 0.5w = 2.5w
            self.ac(w.numpy(), w0 - 0.1 * 2.5 * w0, rtol=1e-4, msg=f"sgd wd {dev}")
        both_devices(body)

    def test_sgd_momentum_two_steps(self):
        w0 = np.array([1., 2.], "float32")
        lr, mu = 0.1, 0.9
        def body(dev):
            w = jt.array(w0)
            opt = torch.optim.SGD([w], lr=lr, momentum=mu)
            # analytic (torch): v1 = g0; v2 = mu*v1 + g1; w2 = w1 - lr*v2
            opt.step((w * w).sum())
            w1 = w.numpy().copy()
            g0 = 2 * w0
            self.ac(w1, w0 - lr * g0, rtol=1e-4, msg=f"sgd mom step1 {dev}")
            opt.step((w * w).sum())
            g1 = 2 * w1
            v2 = mu * g0 + g1
            self.ac(w.numpy(), w1 - lr * v2, rtol=1e-3, msg=f"sgd mom step2 {dev}")
        both_devices(body)

    def test_sgd_torch_style_step_keeps_grad_until_zeroed(self):
        def body(dev):
            w0 = np.array([1.0, 2.0], "float32")
            w = torch.tensor(w0, device=dev, requires_grad=True)
            opt = torch.optim.SGD([w], lr=0.1)

            (w * w).sum().backward()
            published = w.grad
            self.assertIsNotNone(published, f"SGD publishes grad {dev}")
            self.ac(published.detach().clone().cpu().numpy(), 2.0 * w0,
                    msg=f"SGD initial grad {dev}")

            opt.step()
            self.assertIs(w.grad, published, f"SGD step preserves grad identity {dev}")
            self.ac(w.grad.detach().clone().cpu().numpy(), 2.0 * w0,
                    msg=f"SGD step preserves grad {dev}")
            self.ac(w.detach().clone().cpu().numpy(), w0 - 0.1 * 2.0 * w0,
                    msg=f"SGD first torch step {dev}")

            # torch reuses an uncleared gradient on a second step.
            opt.step()
            self.ac(w.detach().clone().cpu().numpy(), w0 - 0.2 * 2.0 * w0,
                    msg=f"SGD repeated step {dev}")

            opt.zero_grad(set_to_none=True)
            before_empty = w.detach().clone().cpu().numpy()
            self.assertIsNone(w.grad, f"SGD set_to_none clears grad {dev}")
            opt.step()
            self.ac(w.detach().clone().cpu().numpy(), before_empty, atol=0.0, rtol=0.0,
                    msg=f"SGD empty step is a no-op {dev}")

            calls = []
            def closure():
                calls.append(1)
                loss = (w * w).sum()
                loss.backward()
                return loss
            returned = opt.step(closure)
            self.assertEqual(len(calls), 1, f"SGD positional closure called once {dev}")
            self.assertIsNotNone(returned, f"SGD positional closure returns loss {dev}")
            self.assertIsNotNone(w.grad, f"SGD positional closure publishes grad {dev}")

        both_devices(body)

    def test_sgd_momentum_state_dict_round_trip(self):
        def body(dev):
            source_value = torch.tensor([1.0, 2.0], device=dev, requires_grad=True)
            source = torch.optim.SGD(
                [source_value], lr=0.1, momentum=0.9)
            (source_value * source_value).sum().backward()
            source.step()
            state_dict = source.state_dict()
            self.assertIn("momentum_buffer", state_dict["state"][0])
            self.assertNotIn("exp_avg_sq", state_dict["state"][0])
            self.assertIn(source_value, source.state)
            self.assertIn("momentum_buffer", source.state[source_value])

            restored_value = torch.tensor([1.0, 2.0], device=dev, requires_grad=True)
            restored = torch.optim.SGD(
                [restored_value], lr=0.1, momentum=0.9)
            restored.load_state_dict(state_dict)
            self.ac(restored.param_groups[0]["values"][0].detach().clone().cpu().numpy(),
                    source.param_groups[0]["values"][0].detach().clone().cpu().numpy(),
                    msg=f"SGD momentum state round trip {dev}")

        both_devices(body)

    def test_native_backward_does_not_double_advance_step(self):
        def body(dev):
            for optimizer_type in (torch.optim.SGD, torch.optim.RMSprop,
                                   torch.optim.Adan):
                initial = np.array([1.0, 2.0], "float32")
                direct_value = jt.array(initial)
                split_value = jt.array(initial)
                direct = optimizer_type([direct_value], lr=0.01)
                split = optimizer_type([split_value], lr=0.01)

                direct.step((direct_value * direct_value).sum())
                split.backward((split_value * split_value).sum())
                self.assertEqual(split.n_step, 1,
                                 f"{optimizer_type.__name__} backward counter {dev}")
                split.step()
                self.assertEqual(split.n_step, 1,
                                 f"{optimizer_type.__name__} split step counter {dev}")
                self.ac(split_value.numpy(), direct_value.numpy(),
                        atol=2e-5, rtol=2e-5,
                        msg=f"{optimizer_type.__name__} split step parity {dev}")

            assigned = torch.tensor([1.0, 2.0], device=dev, requires_grad=True)
            assigned_opt = torch.optim.Adan([assigned], lr=0.01)
            assigned_opt.backward((assigned * assigned).sum())
            assigned.grad = assigned.grad
            assigned_opt.step()
            self.assertEqual(assigned_opt.n_step, 1,
                             f"grad reassignment preserves native counter {dev}")

            accumulated = torch.tensor([1.0, 2.0], device=dev, requires_grad=True)
            accumulated_opt = torch.optim.Adan([accumulated], lr=0.01)
            accumulated_opt.backward((accumulated * accumulated).sum())
            (accumulated * 3.0).sum().backward()
            accumulated_opt.step()
            self.assertEqual(accumulated_opt.n_step, 1,
                             f"mixed backward preserves native counter {dev}")

        both_devices(body)


class TestAdam(Base):
    def test_adamw_preserves_parameter_subclass_multiplication(self):
        def body(dev):
            calls = []

            class OffsetParameter(torch.nn.Parameter):
                def __mul__(self, other):
                    calls.append(other)
                    return super().__mul__(other) + 0.25

            reference = np.array([0.1, -0.3], dtype=np.float64)
            param = OffsetParameter(
                torch.tensor(reference.astype(np.float32), device=dev))
            optimizer = torch.optim.AdamW(
                [param], lr=0.1, eps=0.01, weight_decay=0.1)
            for step in range(1, 4):
                optimizer.zero_grad(set_to_none=True)
                param.grad = torch.ones(2, device=dev)
                optimizer.step()
                reference = reference * 0.99 + 0.25 - 0.1 / 1.01
                self.ac(param.detach().clone().cpu().numpy(), reference,
                        atol=1e-6, rtol=1e-6)
                self.assertEqual(len(calls), step)

        both_devices(body)

    def test_adamw_explicit_cpu_parameters_with_cuda_default(self):
        if not any(use_cuda for _, use_cuda in _DEVICES):
            self.skipTest("requires a real CUDA runtime default")
        with jt.flag_scope(use_cuda=1):
            reference = np.array([0.1, -0.3], dtype=np.float64)
            gradient = np.array([0.025, -0.05], dtype=np.float32)
            param = torch.tensor(reference.astype(np.float32), device="cpu",
                                 requires_grad=True)
            optimizer = torch.optim.AdamW(
                [param], lr=0.03, betas=(0.4, 0.8), eps=0.02,
                weight_decay=0.2)
            for step in range(1, 4):
                optimizer.zero_grad(set_to_none=True)
                param.grad = torch.tensor(gradient, device="cpu")
                optimizer.step()
                reference *= 1 - 0.03 * 0.2
                reference -= 0.03 * gradient.astype(np.float64) / (
                    np.abs(gradient.astype(np.float64)) + 0.02)
                self.ac(param.detach().clone().cpu().numpy(), reference,
                        atol=1e-6, rtol=1e-6)
                state = optimizer.state[param]
                self.assertEqual(int(state["step"]), step)
                for tensor in (param, param.grad,
                               state["exp_avg"], state["exp_avg_sq"]):
                    self.assertEqual(tensor.device.type, "cpu")
                self.ac(state["exp_avg"].detach().clone().cpu().numpy(),
                        gradient * (1 - 0.4 ** step), atol=1e-7)
                self.ac(state["exp_avg_sq"].detach().clone().cpu().numpy(),
                        gradient * gradient * (1 - 0.8 ** step), atol=1e-8)
                self.assertEqual(jt.flags.use_cuda, 1)

    def test_adam_fp32_multistep_groups_and_missing_gradients(self):
        def body(dev):
            for algorithm in (torch.optim.Adam, torch.optim.AdamW):
                refs = [np.array([0.1, -0.3], dtype=np.float64),
                        np.array([0.4, -0.2], dtype=np.float64)]
                params = [torch.tensor(ref.astype(np.float32), device=dev,
                                       requires_grad=True) for ref in refs]
                optimizer = algorithm(
                    [{"params": [params[0]], "lr": 0.003},
                     {"params": [params[1]], "lr": 0.007}],
                    lr=0.001, betas=(0.7, 0.93), eps=1e-4,
                    weight_decay=0.2)
                moments = [np.zeros(2), np.zeros(2)]
                variances = [np.zeros(2), np.zeros(2)]
                steps = [0, 0]
                for iteration in range(4):
                    optimizer.zero_grad(set_to_none=True)
                    for index, param in enumerate(params):
                        if index == 1 and iteration == 1:
                            continue
                        gradient = np.array(
                            [1e-5 * (iteration + 1), -3e-5],
                            dtype=np.float32)
                        param.grad = torch.tensor(gradient, device=dev)
                        steps[index] += 1
                        lr = optimizer.param_groups[index]["lr"]
                        g = gradient.astype(np.float64)
                        if algorithm is torch.optim.AdamW:
                            refs[index] *= 1 - lr * 0.2
                        else:
                            g = g + refs[index] * 0.2
                        moments[index] = 0.7 * moments[index] + 0.3 * g
                        variances[index] = 0.93 * variances[index] + 0.07 * g * g
                        corrected_m = moments[index] / (1 - 0.7 ** steps[index])
                        corrected_v = variances[index] / (1 - 0.93 ** steps[index])
                        refs[index] -= lr * corrected_m / (
                            np.sqrt(corrected_v) + 1e-4)
                    optimizer.step()
                    for index, param in enumerate(params):
                        label = f"{algorithm.__name__} step {iteration} group {index} {dev}"
                        self.ac(param.detach().clone().cpu().numpy(),
                                refs[index], atol=1e-6,
                                rtol=1e-6, msg=label)
                        state = optimizer.state[param]
                        self.assertEqual(int(state["step"]), steps[index], label)
                        self.ac(state["exp_avg"].detach().clone().cpu().numpy(),
                                moments[index],
                                atol=1e-7, rtol=1e-5, msg=label)
                        self.ac(state["exp_avg_sq"].detach().clone().cpu().numpy(),
                                variances[index],
                                atol=1e-9, rtol=1e-5, msg=label)
                    for group in optimizer.param_groups:
                        group["lr"] *= 0.8

        both_devices(body)

    def test_adam_first_step(self):
        w0 = np.array([1., 2., 3.], "float32")
        lr, eps = 0.1, 1e-8
        def body(dev):
            w = jt.array(w0)
            opt = torch.optim.Adam([w], lr=lr, eps=eps)
            opt.step((w * w).sum())              # g = 2w > 0
            # first step: m_hat=g, v_hat=g^2 -> w -= lr * g/(sqrt(g^2)+eps) ~ lr*sign(g)
            g = 2 * w0
            ref = w0 - lr * g / (np.sqrt(g * g) + eps)
            self.ac(w.numpy(), ref, atol=1e-4, msg=f"adam {dev}")
        both_devices(body)

    def test_adam_tiny_gradient_eps_matches_torch_formula(self):
        w0 = np.array([0.0, 1.0], "float32")
        grad = np.array([1e-15, -2e-15], "float32")
        lr, eps = 1e-3, 1e-15
        b0, b1 = 0.9, 0.999

        def body(dev):
            w = jt.array(w0)
            opt = torch.optim.Adam([w], lr=lr, eps=eps, betas=(b0, b1))
            for _ in range(3):
                opt.step((w * jt.array(grad)).sum())
            m = (1 - b0) * grad
            v = (1 - b1) * grad * grad
            m_hat = m / (1 - b0)
            v_hat = v / (1 - b1)
            ref = w0 - 3 * lr * m_hat / (np.sqrt(v_hat) + eps)
            self.ac(w.numpy(), ref, atol=1e-7, rtol=1e-5,
                    msg=f"adam tiny-grad eps {dev}")
        both_devices(body)

    def test_adamw_float32_step_does_not_build_float64_temporaries(self):
        def body(dev):
            jt.clean()
            value = jt.array(np.array([1.0, 2.0], dtype=np.float32))
            optimizer = torch.optim.AdamW([value], lr=0.01)
            optimizer.step((value * value).sum())
            optimizer.step((value * value).sum())
            float64_nodes = [
                node for node in jt.dump_all_graphs().nodes_info
                if "float64" in node
            ]
            self.assertEqual(
                float64_nodes, [],
                f"AdamW float32 graph stays float32 on {dev}",
            )

        both_devices(body)

    def test_adamw_accepts_and_serializes_fused_option(self):
        value = jt.array(np.array([1.0, -2.0], dtype=np.float32))
        optimizer = torch.optim.AdamW(
            [value], lr=0.01, weight_decay=0.1, fused=True)
        optimizer.step((value * value).sum())

        self.assertIs(optimizer.fused, True)
        self.assertTrue(np.isfinite(value.numpy()).all())
        self.assertIs(
            optimizer.state_dict()["param_groups"][0]["fused"], True)

    def test_adam_step_inside_no_grad_keeps_param_trainable(self):
        def body(dev):
            w = torch.tensor([1.0, 2.0], device=dev, requires_grad=True)
            opt = torch.optim.Adam([w], lr=0.1)
            (w * w).sum().backward()
            with torch.no_grad():
                opt.step()
                opt.zero_grad()
            self.assertFalse(w.is_stop_grad(), f"param remains trainable {dev}")
            (w * w).sum().backward()
            with torch.no_grad():
                opt.step()
                opt.zero_grad()
            self.assertEqual(opt.n_step, 2, f"step counter advances {dev}")
        both_devices(body)

    def test_bound_initializers_inside_no_grad_keep_parameter_trainable(self):
        def body(dev):
            def assert_stays_on_device(value, label):
                if dev == "cuda":
                    value.sync()
                    self.assertEqual(value.location(), "device",
                                     f"{label} stays on CUDA")

            operations = (
                ("normal_", lambda value: value.data.normal_(mean=0.0, std=0.1)),
                ("uniform_", lambda value: value.data.uniform_(-0.2, 0.2)),
                ("zero_", lambda value: value.data.zero_()),
                ("fill_", lambda value: value.data.fill_(0.5)),
            )
            for name, operation in operations:
                value = torch.tensor([1.0, 2.0], device=dev,
                                     requires_grad=True)
                self.assertFalse(value.is_stop_grad(), f"{name} starts trainable {dev}")
                with torch.no_grad():
                    operation(value)
                assert_stays_on_device(value, name)
                self.assertFalse(value.is_stop_grad(), f"{name} stays trainable {dev}")
                grad, = torch.autograd.grad((value * 3.0).sum(), value)
                self.assertGreater(float(np.abs(grad.numpy()).max()), 0.0,
                                   f"{name} gradient flows {dev}")

            # Torch factories default to requires_grad=False; this part checks
            # that mutating a parameter data view preserves an enabled flag.
            parent = torch.ones((2, 3), device=dev, requires_grad=True)
            with torch.no_grad():
                parent.data[0].zero_()
            assert_stays_on_device(parent, "data view zero_")
            self.ac(parent.numpy()[0], np.zeros(3, dtype=np.float32),
                    atol=0.0, rtol=0.0, msg=f"view zero_ writes parent {dev}")
            self.assertFalse(parent.is_stop_grad(),
                             f"view zero_ keeps parent trainable {dev}")

            setitem_parent = torch.ones((2, 3), device=dev)
            retained_data = setitem_parent.data
            self.assertIsNot(retained_data, setitem_parent,
                             f"data returns a detached alias {dev}")
            self.assertTrue(retained_data.is_stop_grad(),
                            f"data alias is detached {dev}")
            retained_data[1] = 4.0
            assert_stays_on_device(setitem_parent, "data setitem")
            self.assertEqual(tuple(retained_data.shape), (2, 3),
                             f"data setitem keeps alias shape {dev}")
            self.ac(setitem_parent.numpy()[1], np.full(3, 4.0, dtype=np.float32),
                    atol=0.0, rtol=0.0,
                    msg=f"data setitem writes parent {dev}")
            self.ac(retained_data.numpy(), setitem_parent.numpy(), atol=0.0, rtol=0.0,
                    msg=f"retained data alias observes mutation {dev}")

            parent[1].add_(2.0)
            self.ac(parent.numpy()[1], np.full(3, 3.0, dtype=np.float32),
                    atol=0.0, rtol=0.0, msg=f"view add_ writes parent {dev}")

            chained = torch.zeros((2, 3, 4), device=dev)
            chained[1][2].fill_(7.0)
            self.ac(chained.numpy()[1, 2], np.full(4, 7.0, dtype=np.float32),
                    atol=0.0, rtol=0.0,
                    msg=f"chained view fill_ writes root parent {dev}")

            direct = torch.zeros((2, 3, 4), device=dev)
            middle = direct[1]
            middle[1:] = 5.0
            self.ac(direct.numpy()[1, 1:], np.full((2, 4), 5.0, dtype=np.float32),
                    atol=0.0, rtol=0.0,
                    msg=f"chained direct setitem writes root parent {dev}")

            deep = torch.zeros((2, 3, 4, 5), device=dev)
            deep_view = deep[1][2][3]
            self.assertTrue(deep_view._is_view())
            self.assertNotIn("_torch_index_parent", deep_view.__dict__)
            self.assertNotIn("_torch_index_slices", deep_view.__dict__)
            deep_view.fill_(9.0)
            self.ac(deep.numpy()[1, 2, 3], np.full(5, 9.0, dtype=np.float32),
                    atol=0.0, rtol=0.0,
                    msg=f"deep chained view writes root parent {dev}")

            frozen = torch.ones((2, 3), device=dev)
            frozen.normal_(0.0, 0.1)
            frozen[0].add_(1.0)
            self.assertFalse(frozen.requires_grad,
                             f"in-place ops keep frozen tensor frozen {dev}")

        both_devices(body)

    def test_adamw_step_without_gradients_is_noop(self):
        def body(dev):
            value = torch.tensor([1.0, 2.0], device=dev, requires_grad=True)
            before = value.numpy().copy()
            optimizer = torch.optim.AdamW([value], lr=0.1)
            optimizer.zero_grad(set_to_none=False)
            self.assertIsNone(value.grad,
                              f"AdamW zero_grad preserves absent grad {dev}")
            optimizer.step()
            self.ac(value.numpy(), before, atol=0.0, rtol=0.0,
                    msg=f"AdamW zero-filled empty step no-op {dev}")
            self.assertEqual(optimizer.n_step, 0,
                             f"AdamW zero-filled empty step counter {dev}")

            optimizer.zero_grad(set_to_none=True)
            optimizer.step()
            self.ac(value.numpy(), before, atol=0.0, rtol=0.0,
                    msg=f"AdamW no-gradient no-op {dev}")
            self.assertEqual(optimizer.n_step, 0,
                             f"AdamW empty step counter {dev}")

            (value * value).sum().backward()
            published_grad = value.grad
            self.assertIsNotNone(published_grad, f"AdamW publishes grad {dev}")
            optimizer.zero_grad(set_to_none=False)
            self.assertIs(value.grad, published_grad,
                          f"AdamW zero_grad preserves grad identity {dev}")
            self.ac(published_grad.numpy(), np.zeros_like(before),
                    atol=0.0, rtol=0.0,
                    msg=f"AdamW zero_grad clears retained grad object {dev}")
            (value * value).sum().backward()
            self.assertIs(value.grad, published_grad,
                          f"AdamW backward preserves retained grad identity {dev}")
            optimizer.step()
            self.assertEqual(optimizer.n_step, 1,
                             f"AdamW first real step counter {dev}")
            after_first = value.numpy().copy()
            optimizer.zero_grad(set_to_none=True)
            self.assertIsNone(value.grad, f"AdamW set_to_none clears grad {dev}")
            optimizer.step()
            self.ac(value.numpy(), after_first, atol=0.0, rtol=0.0,
                    msg=f"AdamW cleared gradient is not reused {dev}")
            self.assertEqual(optimizer.n_step, 1,
                             f"AdamW cleared empty step counter {dev}")

        both_devices(body)

    def test_adamw_grad_setter_updates_optimizer_slot(self):
        def body(dev):
            value = torch.tensor([1.0, 2.0], device=dev, requires_grad=True)
            optimizer = torch.optim.AdamW([value], lr=0.1, weight_decay=0.1)
            (value * value).sum().backward()
            before = value.numpy().copy()

            value.grad = None
            optimizer.step()
            self.assertIsNone(value.grad, f"manual None clears grad {dev}")
            self.ac(value.numpy(), before, atol=0.0, rtol=0.0,
                    msg=f"cleared manual grad makes step a no-op {dev}")
            self.assertEqual(optimizer.n_step, 0,
                             f"cleared manual grad keeps step counter {dev}")

            manual = jt.ones_like(value).stop_grad()
            value.grad = manual
            self.assertIs(value.grad, manual, f"manual grad identity {dev}")
            (value * value).sum().backward()
            self.assertIs(value.grad, manual,
                          f"backward accumulates into manual grad object {dev}")
            self.ac(value.grad.numpy(), np.ones_like(before) + 2.0 * before,
                    msg=f"manual grad participates in accumulation {dev}")
            optimizer.step()
            self.assertEqual(optimizer.n_step, 1,
                             f"manual grad advances optimizer {dev}")
            self.assertGreater(float(np.abs(value.numpy() - before).max()), 0.0,
                               f"manual grad updates parameter {dev}")

        both_devices(body)

    def test_shared_parameter_uses_one_published_grad_slot(self):
        def body(dev):
            value = torch.tensor([1.0, 2.0], device=dev, requires_grad=True)
            first = torch.optim.AdamW([value], lr=0.1)
            second = torch.optim.AdamW([value], lr=0.1)

            (value * value).sum().backward()
            published = value.grad
            self.assertIs(first.param_groups[0]["grads"][0], published,
                          f"first optimizer shares published grad {dev}")
            self.assertIs(second.param_groups[0]["grads"][0], published,
                          f"second optimizer shares published grad {dev}")

            first.zero_grad(set_to_none=False)
            self.assertIs(value.grad, published,
                          f"shared zero_grad keeps published identity {dev}")
            self.assertIs(second.param_groups[0]["grads"][0], published,
                          f"shared optimizer slot follows zero_grad {dev}")
            (value * value).sum().backward()
            self.assertIs(value.grad, published,
                          f"shared backward keeps published identity {dev}")
            self.assertIs(second.param_groups[0]["grads"][0], published,
                          f"shared backward keeps one optimizer slot {dev}")

        both_devices(body)

    def test_adamw_unused_parameter_stays_none_and_unchanged(self):
        def body(dev):
            used = torch.tensor([1.0, 2.0], device=dev, requires_grad=True)
            unused = torch.tensor([3.0, 4.0], device=dev, requires_grad=True)
            unused_before = unused.numpy().copy()
            optimizer = torch.optim.AdamW(
                [used, unused], lr=0.1, weight_decay=0.2)

            (used * used).sum().backward()
            self.assertIsNotNone(used.grad, f"used parameter gets grad {dev}")
            self.assertIsNone(unused.grad, f"unused parameter keeps grad None {dev}")
            optimizer.step()
            self.ac(unused.numpy(), unused_before, atol=0.0, rtol=0.0,
                    msg=f"unused parameter skips AdamW weight decay {dev}")
            self.ac(optimizer.param_groups[0]["m"][1].numpy(),
                    np.zeros_like(unused_before), atol=0.0, rtol=0.0,
                    msg=f"unused parameter keeps first moment zero {dev}")
            self.ac(optimizer.param_groups[0]["values"][1].numpy(),
                    np.zeros_like(unused_before), atol=0.0, rtol=0.0,
                    msg=f"unused parameter keeps second moment zero {dev}")

        both_devices(body)

    def test_adamw_default_decay_distinguishes_zero_grad_from_none(self):
        def body(dev):
            zero_grad = torch.tensor([2.0], device=dev, requires_grad=True)
            missing_grad = torch.tensor([2.0], device=dev, requires_grad=True)
            optimizer = torch.optim.AdamW(
                [zero_grad, missing_grad], lr=1e-3)
            self.assertEqual(optimizer.weight_decay, 0.01)
            no_decay = torch.optim.AdamW([torch.ones(1, device=dev)], lr=1e-3,
                                         weight_decay=0)
            self.assertEqual(no_decay.weight_decay, 0)

            (zero_grad * 0).sum().backward()
            self.assertIsNotNone(zero_grad.grad)
            self.ac(zero_grad.grad.numpy(), np.zeros(1, dtype=np.float32),
                    atol=0, rtol=0, msg=f"explicit zero gradient {dev}")
            self.assertIsNone(missing_grad.grad)
            optimizer.step()
            self.ac(zero_grad.numpy(), np.array([1.99998], dtype=np.float32),
                    atol=2e-7, rtol=0, msg=f"zero gradient gets decay {dev}")
            self.ac(missing_grad.numpy(), np.array([2.0], dtype=np.float32),
                    atol=0, rtol=0, msg=f"None gradient skips decay {dev}")
            self.assertEqual(float(optimizer.state[zero_grad]["step"]), 1.0)
            self.assertNotIn(missing_grad, optimizer.state)

            optimizer.zero_grad(set_to_none=True)
            (zero_grad * 0).sum().backward()
            optimizer.step()
            self.ac(zero_grad.numpy(), np.array([1.9999601], dtype=np.float32),
                    atol=2e-7, rtol=0, msg=f"second zero-gradient decay {dev}")
            self.assertEqual(float(optimizer.state[zero_grad]["step"]), 2.0)

        both_devices(body)

    def test_adamw_late_parameter_uses_its_first_step_bias(self):
        def body(dev):
            first = torch.tensor([1.0, 2.0], device=dev, requires_grad=True)
            late_initial = np.array([3.0, 4.0], "float32")
            late = torch.tensor(late_initial, device=dev, requires_grad=True)
            optimizer = torch.optim.AdamW(
                [first, late], lr=0.01, weight_decay=0.2)

            (first * first).sum().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.assertNotIn(late, optimizer.state,
                             f"unused AdamW state stays uninitialized {dev}")

            (late * late).sum().backward()
            optimizer.step()

            fresh = torch.tensor(late_initial, device=dev, requires_grad=True)
            fresh_optimizer = torch.optim.AdamW(
                [fresh], lr=0.01, weight_decay=0.2)
            (fresh * fresh).sum().backward()
            fresh_optimizer.step()

            self.ac(late.numpy(), fresh.numpy(), atol=1e-6, rtol=1e-6,
                    msg=f"late AdamW parameter gets first-step correction {dev}")
            self.assertEqual(int(optimizer.state[late]["step"]), 1,
                             f"late AdamW per-parameter step {dev}")
            state_dict = optimizer.state_dict()
            self.assertEqual(float(state_dict["state"][1]["step"].item()), 1.0,
                             f"late AdamW state_dict step {dev}")

        both_devices(body)

    def test_adamw_partial_load_clears_absent_parameter_state(self):
        def body(dev):
            source_params = [
                torch.tensor([1.0, 2.0], device=dev, requires_grad=True),
                torch.tensor([3.0, 4.0], device=dev, requires_grad=True),
            ]
            source = torch.optim.AdamW(source_params, lr=0.01)
            (source_params[0] * source_params[0]).sum().backward()
            source.step()
            partial = source.state_dict()
            self.assertEqual(set(partial["state"]), {0})

            target_params = [
                torch.tensor([1.0, 2.0], device=dev, requires_grad=True),
                torch.tensor([3.0, 4.0], device=dev, requires_grad=True),
            ]
            target = torch.optim.AdamW(target_params, lr=0.01)
            ((target_params[0] * target_params[0]).sum()
             + (target_params[1] * target_params[1]).sum()).backward()
            target.step()
            self.assertEqual(set(target.state_dict()["state"]), {0, 1})

            target.load_state_dict(partial)
            self.assertEqual(target.n_step, 1)
            self.assertIn(target_params[0], target.state)
            self.assertNotIn(target_params[1], target.state)
            self.ac(target.param_groups[0]["m"][1].numpy(),
                    np.zeros(2, dtype="float32"), atol=0.0, rtol=0.0,
                    msg=f"AdamW absent first moment reset {dev}")
            self.ac(target.param_groups[0]["values"][1].numpy(),
                    np.zeros(2, dtype="float32"), atol=0.0, rtol=0.0,
                    msg=f"AdamW absent second moment reset {dev}")

        both_devices(body)

    def test_optimizer_load_rejects_group_shape_before_mutation(self):
        value = torch.tensor([1.0, 2.0], requires_grad=True)
        optimizer = torch.optim.AdamW([value], lr=0.01)
        (value * value).sum().backward()
        optimizer.step()
        before_step = optimizer.n_step
        before_m = optimizer.param_groups[0]["m"][0].numpy().copy()
        bad = optimizer.state_dict()
        bad["param_groups"][0]["params"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "doesn't match the size"):
            optimizer.load_state_dict(bad)
        self.assertEqual(optimizer.n_step, before_step)
        self.ac(optimizer.param_groups[0]["m"][0].numpy(), before_m,
                atol=0.0, rtol=0.0,
                msg="mismatched optimizer load is non-mutating")

    def test_optimizer_load_rejects_malformed_state_before_mutation(self):
        value = torch.tensor([1.0, 2.0], requires_grad=True)
        optimizer = torch.optim.AdamW([value], lr=0.01)
        (value * value).sum().backward()
        optimizer.step()
        before_step = optimizer.n_step
        before_m = optimizer.param_groups[0]["m"][0].numpy().copy()

        bad_state = optimizer.state_dict()
        bad_state["state"] = None
        with self.assertRaisesRegex(TypeError, "state must be a mapping"):
            optimizer.load_state_dict(bad_state)
        self.assertEqual(optimizer.n_step, before_step)
        self.ac(optimizer.param_groups[0]["m"][0].numpy(), before_m,
                atol=0.0, rtol=0.0)

        bad_step = optimizer.state_dict()
        bad_step["state"][0]["step"] = "invalid"
        with self.assertRaisesRegex(ValueError, "step must be numeric"):
            optimizer.load_state_dict(bad_step)
        self.assertEqual(optimizer.n_step, before_step)
        self.ac(optimizer.param_groups[0]["m"][0].numpy(), before_m,
                atol=0.0, rtol=0.0)

    def test_adam_state_field_assignment_and_delete_are_live(self):
        def body(dev):
            value = torch.tensor([1.0, 2.0], device=dev, requires_grad=True)
            optimizer = torch.optim.AdamW([value], lr=0.01)
            (value * value).sum().backward()
            optimizer.step()

            replacement = jt.array(np.array([7.0, 8.0], "float32")).stop_grad()
            optimizer.state[value]["exp_avg"] = replacement
            self.assertIs(optimizer.param_groups[0]["m"][0], replacement)
            self.assertIs(optimizer.state[value]["exp_avg"], replacement)
            replacement_before = replacement.numpy().copy()

            del optimizer.state[value]
            self.assertNotIn(value, optimizer.state)
            self.assertEqual(optimizer.n_step, 0)
            self.assertIsNot(optimizer.param_groups[0]["m"][0], replacement)
            self.ac(optimizer.param_groups[0]["m"][0].numpy(),
                    np.zeros(2, dtype="float32"), atol=0.0, rtol=0.0)
            self.ac(replacement.numpy(), replacement_before,
                    atol=0.0, rtol=0.0,
                    msg=f"deleted state does not mutate saved references {dev}")

            optimizer.zero_grad(set_to_none=True)
            fresh_value = torch.tensor(value.numpy().copy(), device=dev,
                                       requires_grad=True)
            fresh = torch.optim.AdamW([fresh_value], lr=0.01)
            (value * value).sum().backward()
            (fresh_value * fresh_value).sum().backward()
            optimizer.step()
            fresh.step()
            self.ac(value.numpy(), fresh_value.numpy(), atol=1e-6, rtol=1e-6,
                    msg=f"deleted Adam state restarts from first step {dev}")

        both_devices(body)

    def test_adam_state_dict_is_torch_shaped(self):
        def body(dev):
            w = jt.array(np.array([1.0, 2.0], "float32"))
            opt = torch.optim.Adam([{"params": [w], "lr": 0.1, "name": "w"}], lr=0.0)
            opt.step((w * w).sum())
            sd = opt.state_dict()
            self.assertIn("state", sd, f"state key {dev}")
            self.assertIn("param_groups", sd, f"param_groups key {dev}")
            self.assertEqual(sd["param_groups"][0]["params"], [0], f"param id {dev}")
            self.assertEqual(sd["param_groups"][0]["name"], "w", f"group name {dev}")
            self.assertIn("exp_avg", sd["state"][0], f"exp_avg {dev}")
            self.assertIn("exp_avg_sq", sd["state"][0], f"exp_avg_sq {dev}")
            self.assertIn("step", sd["state"][0], f"step {dev}")
        both_devices(body)


class TestScheduler(Base):
    def test_steplr(self):
        def body(dev):
            w = jt.array(np.array([1.0], "float32"))
            opt = torch.optim.SGD([w], lr=1.0)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)
            lrs = []
            for _ in range(5):
                lrs.append(float(opt.lr))
                sched.step()
            # lr = 1.0 for epochs 0,1; 0.1 for 2,3; 0.01 for 4
            self.ac(lrs, [1.0, 1.0, 0.1, 0.1, 0.01], rtol=1e-5, msg=f"steplr {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
