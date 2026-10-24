"""Public Adam/AdamW state policy and three-step/load trajectory proposal.

Same file, independent official Torch2.8 or normal candidate2.11 bootstrap;
one CPU/CUDA node per process via JITTOR_TEST_DEVICES. All full arrays remain
RAM-only, scalar/control metadata exact, numeric atol=rtol=1e-5 throughout.
Not device-validated. Candidate Adam(True) is explicitly unsupported: the
negative branch records refusal, not Native parity for that legal capability.
Proposed permanent path: compat/tests/torch/test_optimizer_decoupled_state.py.
No FSDP/full PPO/checkpoint/RNG/L3 acceptance is claimed by this small control.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import os
import sys
import unittest
import numpy as np


INITIAL = np.array([[.5, -.75, 1.25], [-1.5, .25, .875]], dtype=np.float32)
GRADS = np.array([[[.25, -.5, .75], [-.125, .625, -.375]],
                  [[-.5, .25, .125], [.75, -.25, .5]],
                  [[.375, -.75, .5], [-.625, .125, -.25]]], dtype=np.float32)
OPTIONS = dict(lr=.125, betas=(.5, .75), eps=.125, weight_decay=.25)


class TestOptimizerDecoupledState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.jt = sys.modules.get("jittor")
        if cls.jt is None:
            assert str(torch.__version__) == "2.8.0+cu128" and hasattr(torch, "_C")
            assert not any(name.startswith("jittor.") for name in sys.modules)
        else:
            from jittor.compat.torch.publication import namespace_owner
            # Reporting the API version is an explicit policy, not activation.
            # Read both genuine identities; never rewrite a version for a test.
            assert str(torch.__torch_version__) == str(torch.version.__version__) == "2.11.0"
            assert str(torch.__jittor_version__) == str(cls.jt.__version__)
            assert str(torch.__version__) in (str(torch.__jittor_version__), str(torch.__torch_version__))
            assert namespace_owner(torch) is cls.jt and torch.Tensor is not cls.jt.Var

    @contextmanager
    def _scope(self, device):
        selected = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
        if selected not in ("cpu", "cuda"):
            raise ValueError("select exactly one CPU or CUDA node per process")
        if selected != device:
            self.skipTest("node excluded by explicit single-device selection")
        with ExitStack() as stack:
            if device == "cuda":
                self.assertTrue(self.torch.cuda.is_available(), "requested real CUDA unavailable")
                self.assertEqual(self.torch.cuda.current_device(), 0)
            if self.jt is not None:
                if device == "cuda":
                    self.assertTrue(self.jt.compiler.has_cuda and self.jt.compiler.is_cuda)
                    self.assertFalse(any(getattr(self.jt.compiler, key, False)
                                         for key in ("has_acl", "has_rocm", "has_corex")))
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(self.jt.runtime.scope(use_cuda=int(device == "cuda"),
                    device_id=0 if device == "cuda" else -1, backend_fallback="error"))
                stack.enter_context(forbid_backend_fallbacks())
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)
            yield
            if self.jt is not None:
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def _resident(self, value, device):
        self.assertEqual(value.device.type, device)
        if device == "cuda":
            self.assertEqual(value.device.index, 0)
        if self.jt is not None:
            native = getattr(value, "_var", value)
            native.sync()
            self.assertEqual(self.jt.core.dispatch_context([native]), (device, -1 if device == "cpu" else 0))
            self.assertEqual(native.placement_backend, 0 if device == "cpu" else 1)
            self.assertEqual(native.location(), "cpu" if device == "cpu" else "device")
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)
        elif device == "cuda":
            self.torch.cuda.synchronize(value.device)

    @staticmethod
    def _flags(value):
        return (tuple(value.shape), tuple(value.stride()), value.dtype, str(value.device),
                value.requires_grad, value.is_leaf, value.grad_fn is None)

    def _array(self, value):
        device = value.device.type
        before = self._flags(value)
        self._resident(value, device)
        with self.torch.no_grad():
            owned = value.clone()
            self.assertTrue(bool(owned is not value), "snapshot_independent_object")
            self.assertFalse(owned.requires_grad)
            self._resident(owned, device)
            host = owned.cpu()
            self._resident(host, "cpu")
            result = np.array(host.numpy(), copy=True)
        self.assertEqual(self._flags(value), before)
        self._resident(value, device)
        self.assertTrue(bool(np.isfinite(result).all()), "full_array_finite")
        return result

    def _close(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(bool(np.isfinite(actual).all() and np.isfinite(expected).all()), "full_arrays_finite")
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        self.assertTrue(bool(np.all(error <= 1e-5 + 1e-5 * np.abs(expected))), "all_elements_fixed_1e5")

    def _copy_tree(self, value):
        if isinstance(value, self.torch.Tensor):
            before = self._flags(value)
            self._resident(value, value.device.type)
            with self.torch.no_grad():
                result = value.clone()
            self._resident(result, value.device.type)
            self.assertTrue(bool(result is not value), "checkpoint_independent_tensor")
            self.assertEqual(self._flags(value), before)
            return result
        if type(value) is dict:
            return {key: self._copy_tree(item) for key, item in value.items()}
        if type(value) in (list, tuple):
            return type(value)(self._copy_tree(item) for item in value)
        self.assertTrue(value is None or type(value) in (bool, int, float, str), "supported_complete_metadata")
        return deepcopy(value)

    def _same_tree(self, left, right):
        if isinstance(left, self.torch.Tensor):
            self.assertTrue(bool(isinstance(right, self.torch.Tensor)), "state_tensor_type")
            self.assertEqual(self._flags(left), self._flags(right))
            self._close(self._array(left), self._array(right))
        else:
            self.assertTrue(bool(type(left) is type(right)), "exact_metadata_type")
            if type(left) is dict:
                self.assertEqual(set(left), set(right), "complete_metadata_key_set")
                for key in left:
                    self._same_tree(left[key], right[key])
            elif type(left) in (list, tuple):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    self._same_tree(a, b)
            elif type(left) is float:
                self.assertEqual(left.hex(), right.hex(), "exact_control_float")
            else:
                self.assertEqual(left, right)

    def _make(self, kind, device, initial=INITIAL):
        torch = self.torch
        parameter = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float32, device=device))
        unused = torch.nn.Parameter(torch.tensor([.5, -.25], dtype=torch.float32, device=device))
        opt = getattr(torch.optim, kind)([dict(params=[parameter, unused], unknown_false=False)], **OPTIONS)
        return parameter, unused, opt

    def _policy(self, state, decoupled):
        self.assertEqual(set(state), {"state", "param_groups"})
        self.assertEqual(len(state["param_groups"]), 1)
        group = state["param_groups"][0]
        self.assertIn("decoupled_weight_decay", group)
        self.assertTrue(bool(group["decoupled_weight_decay"] is decoupled), "exact_decay_policy")
        self.assertTrue(bool(group["unknown_false"] is False), "exact_unknown_false")

    def _step(self, parameter, unused, opt, index, device):
        opt.zero_grad(set_to_none=True)
        cotangent = self.torch.tensor(GRADS[index], device=device)
        loss = (parameter * cotangent).sum()
        self._resident(loss, device)
        loss.backward()
        self.assertTrue(bool(parameter.grad is not None), "active_parameter_grad_present")
        self.assertEqual(parameter.grad.dtype, self.torch.float32)
        self.assertTrue(bool(unused.grad is None), "unused_parameter_grad_none")
        self._close(self._array(parameter.grad), GRADS[index])
        opt.step()
        self._resident(parameter, device)
        self._close(self._array(unused), np.array([.5, -.25], dtype=np.float32))
        self.assertTrue(parameter.requires_grad and parameter.is_leaf)

    @staticmethod
    def _reference(decoupled, count):
        p = INITIAL.astype(np.float64)
        m = np.zeros_like(p)
        v = np.zeros_like(p)
        for i in range(count):
            g = GRADS[i].astype(np.float64)
            if decoupled:
                p = p * (1 - OPTIONS["lr"] * OPTIONS["weight_decay"])
            else:
                g = g + OPTIONS["weight_decay"] * p
            m = .5 * m + .5 * g
            v = .75 * v + .25 * g * g
            p = p - OPTIONS["lr"] * (m / (1 - .5 ** (i + 1))) / (np.sqrt(v / (1 - .75 ** (i + 1))) + OPTIONS["eps"])
        return p, m, v

    def _state_values(self, parameter, unused, opt, count, decoupled):
        state = opt.state_dict()  # Original public method, including Native Dynamo.
        self._policy(state, decoupled)
        ids = state["param_groups"][0]["params"]
        self.assertEqual(len(ids), 2)
        self.assertEqual(set(state["state"]), {ids[0]})
        entry = state["state"][ids[0]]
        self.assertEqual(set(entry), {"step", "exp_avg", "exp_avg_sq"})
        for value in (parameter, entry["exp_avg"], entry["exp_avg_sq"], entry["step"]):
            self.assertEqual(value.dtype, self.torch.float32)
        expected, m, v = self._reference(decoupled, count)
        self._close(self._array(parameter), expected)
        self._close(self._array(entry["exp_avg"]), m)
        self._close(self._array(entry["exp_avg_sq"]), v)
        step = self._array(entry["step"])
        self.assertEqual(step.shape, ())
        self.assertEqual(float(step), count, "exact_discrete_step")
        self.assertTrue(bool(unused.grad is None), "unused_parameter_grad_none")
        return state

    def _roundtrip(self, kind, mode, device):
        decoupled = kind == "AdamW"
        p, unused, opt = self._make(kind, device)
        self._policy(opt.state_dict(), decoupled)
        self._step(p, unused, opt, 0, device)
        saved = self._copy_tree(self._state_values(p, unused, opt, 1, decoupled))
        if mode == "legacy":
            del saved["param_groups"][0]["decoupled_weight_decay"]
        elif mode == "false":
            saved["param_groups"][0]["decoupled_weight_decay"] = False
        saved_before = self._copy_tree(saved)
        q, unused_q, restored = self._make(kind, device, self._array(p))
        restored.load_state_dict(saved)  # Actual public load, no replacement/unwrap.
        self._same_tree(saved, saved_before)
        self.assertTrue(bool(restored.param_groups[0]["params"][0] is q), "loaded_parameter_identity")
        self._same_tree(opt.state_dict(), restored.state_dict())
        for index in (1, 2):
            self._step(p, unused, opt, index, device)
            self._step(q, unused_q, restored, index, device)
            a = self._state_values(p, unused, opt, index + 1, decoupled)
            b = self._state_values(q, unused_q, restored, index + 1, decoupled)
            self._same_tree(a, b)
            self._close(self._array(p), self._array(q))

    def _unsupported_adam_true(self, device):
        p, unused, opt = self._make("Adam", device)
        before = self._copy_tree(opt.state_dict())
        saved = self._copy_tree(before)
        saved["param_groups"][0]["decoupled_weight_decay"] = True
        if self.jt is not None:
            with self.assertRaisesRegex(NotImplementedError, "supported"):
                opt.load_state_dict(saved)
            self._same_tree(before, opt.state_dict())
            self._close(self._array(p), INITIAL)
        else:
            opt.load_state_dict(saved)
            for index in range(3):
                self._step(p, unused, opt, index, device)
                self._state_values(p, unused, opt, index + 1, True)

    def _run(self, device):
        with self._scope(device):
            for kind in ("Adam", "AdamW"):
                for mode in ("roundtrip", "legacy", "false"):
                    with self.subTest(kind=kind, mode=mode):
                        self._roundtrip(kind, mode, device)
            with self.subTest(scope="Adam(True): Native capability / candidate explicit unsupported"):
                self._unsupported_adam_true(device)

    def test_cuda_public_state_groups_bases_and_safe_checkpoint(self):
        from collections import defaultdict
        import importlib
        import io
        torch = self.torch
        with self._scope('cuda'):
            for name, bases in (('Optimizer', (object,)),
                    ('Adam', (torch.optim.Optimizer,)), ('AdamW', (torch.optim.Adam,))):
                cls = getattr(torch.optim, name)
                self.assertEqual(cls.__bases__, bases)
                self.assertIs(getattr(importlib.import_module(cls.__module__), name), cls)
            p = torch.nn.Parameter(torch.tensor(INITIAL, device='cuda:0'))
            custom = ['caller-owned-config']
            group = dict(params=[p], m=custom, values=custom, _torch_steps=custom,
                         unknown_false=False)
            opt = torch.optim.AdamW([group], **OPTIONS)
            self.assertIs(type(opt.param_groups), list)
            self.assertIs(type(opt.param_groups[0]), dict)
            self.assertIs(opt.param_groups[0], group)
            expected_keys = set(opt.defaults) | {'params', 'm', 'values', '_torch_steps', 'unknown_false'}
            self.assertEqual(set(group), expected_keys)
            self.assertIs(type(opt.state), defaultdict)
            self.assertIs(opt.state.default_factory, dict)
            entry = opt.state[p]
            self.assertIs(type(entry), dict)
            self.assertEqual(entry, {})
            p.grad = torch.tensor(GRADS[0], device='cuda:0')
            opt.step()
            self.assertIs(opt.state[p], entry)
            step = entry['step']
            self.assertIs(type(step), torch.Tensor)
            self.assertEqual((tuple(step.shape), step.dtype, str(step.device)),
                             ((), torch.float32, 'cpu'))
            self.assertFalse(step.requires_grad)
            self.assertTrue(step.is_leaf)
            self.assertIsNone(step.grad_fn)
            self.assertEqual(step.item(), 1.)
            self.assertEqual(set(group), expected_keys)
            saved = opt.state_dict()
            for key in ('m', 'values', '_torch_steps'):
                self.assertIs(group[key], custom)
                self.assertIs(saved['param_groups'][0][key], custom)
            packed = saved['state'][saved['param_groups'][0]['params'][0]]
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                self.assertIs(packed[key], entry[key])
            stream = io.BytesIO()
            torch.save(saved, stream)
            stream.seek(0)
            loaded = torch.load(stream, weights_only=True)
            self._same_tree(saved, loaded)
            q = torch.nn.Parameter(p.detach().clone())
            restored = torch.optim.AdamW([q], **OPTIONS)
            restored.load_state_dict(loaded)
            self.assertIs(type(restored.state), defaultdict)
            self.assertIs(restored.state.default_factory, dict)
            self.assertIs(restored.param_groups[0]['params'][0], q)
            self._same_tree(opt.state_dict(), restored.state_dict())
            for index in (1, 2):
                for parameter, optimizer in ((p, opt), (q, restored)):
                    optimizer.zero_grad(set_to_none=True)
                    (parameter * torch.tensor(GRADS[index], device='cuda:0')).sum().backward()
                    optimizer.step()
                    self._resident(parameter, 'cuda')
                self._close(self._array(p), self._array(q))
                self._same_tree(opt.state_dict(), restored.state_dict())
            self.assertIs(entry['step'], step)
            self.assertEqual(step.item(), 3.)

    def test_cpu(self):
        self._run("cpu")

    def test_cuda(self):
        self._run("cuda")


if __name__ == "__main__":
    unittest.main()
