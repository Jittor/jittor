"""Independent PyTorch oracle coverage for canonical multi-head attention."""

import unittest

import numpy as np

from _helpers import capability as _test_capability

import jittor as jt
import jittor.nn as jnn
from _helpers.torch_runtime import import_torch_modules, modules_available


skip_this_test = not modules_available("torch")
torch = None
tnn = None


def setUpModule():
    global torch, tnn
    if not skip_this_test:
        torch, tnn = import_torch_modules("torch", "torch.nn")


def _copy_parameter(destination, source):
    if destination is not None:
        destination.assign(jt.array(source.detach().cpu().numpy()))


@unittest.skipIf(skip_this_test, "No independent Torch found")
class TestAttentionOracle(unittest.TestCase):
    def _compare(
        self,
        name,
        module_kwargs,
        query_shape,
        key_shape=None,
        value_shape=None,
        batch_first=False,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=True,
        average_attn_weights=True,
    ):
        rng = np.random.RandomState(sum(ord(char) for char in name))
        torch_module = tnn.MultiheadAttention(batch_first=batch_first, **module_kwargs).eval()
        key_shape = query_shape if key_shape is None else key_shape
        value_shape = key_shape if value_shape is None else value_shape
        arrays = [
            rng.randn(*shape).astype("float32") for shape in (query_shape, key_shape, value_shape)
        ]
        torch_tensors = [torch.from_numpy(array) for array in arrays]
        torch_mask = torch.from_numpy(attn_mask) if attn_mask is not None else None
        torch_padding = torch.from_numpy(key_padding_mask) if key_padding_mask is not None else None

        with torch.no_grad():
            expected_output, expected_weights = torch_module(
                *torch_tensors,
                attn_mask=torch_mask,
                key_padding_mask=torch_padding,
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
            )
        with jt.flag_scope(use_cuda=0):
            jittor_module = jnn.MultiheadAttention(batch_first=batch_first, **module_kwargs).eval()
            for attribute in (
                "in_proj_weight",
                "in_proj_bias",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
                "bias_k",
                "bias_v",
            ):
                _copy_parameter(
                    getattr(jittor_module, attribute, None),
                    getattr(torch_module, attribute, None),
                )
            _copy_parameter(jittor_module.out_proj.weight, torch_module.out_proj.weight)
            _copy_parameter(jittor_module.out_proj.bias, torch_module.out_proj.bias)
            jittor_tensors = [jt.array(array) for array in arrays]
            jittor_mask = jt.array(attn_mask) if attn_mask is not None else None
            jittor_padding = jt.array(key_padding_mask) if key_padding_mask is not None else None
            actual_output, actual_weights = jittor_module(
                *jittor_tensors,
                attn_mask=jittor_mask,
                key_padding_mask=jittor_padding,
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
            )
            actual_output_array = actual_output.numpy()
            actual_weights_array = actual_weights.numpy() if actual_weights is not None else None
        np.testing.assert_allclose(
            actual_output_array,
            expected_output.numpy(),
            rtol=3e-5,
            atol=3e-5,
            err_msg=name + " output",
        )
        if need_weights:
            np.testing.assert_allclose(
                actual_weights_array,
                expected_weights.numpy(),
                rtol=3e-5,
                atol=3e-5,
                err_msg=name + " weights",
            )
        else:
            self.assertIsNone(actual_weights)
            self.assertIsNone(expected_weights)

    def test_layouts_masks_and_projection_variants(self):
        common = {"embed_dim": 8, "num_heads": 2}
        self._compare("sequence-first", common, (4, 3, 8))
        self._compare("batch-first", common, (3, 4, 8), batch_first=True)
        self._compare("unbatched", common, (4, 8), average_attn_weights=False)

        boolean_mask = np.triu(np.ones((4, 5), dtype=bool), 1)
        padding_mask = np.array(
            [
                [False, False, False, True, True],
                [False, False, False, False, True],
            ]
        )
        self._compare(
            "boolean-mask-and-padding",
            common,
            (4, 2, 8),
            (5, 2, 8),
            (5, 2, 8),
            attn_mask=boolean_mask,
            key_padding_mask=padding_mask,
        )
        additive_mask = np.stack(
            [np.where(boolean_mask, -100.0, 0.0).astype("float32") for _ in range(4)]
        )
        additive_mask[1, :, 0] -= 0.5
        additive_mask[2, :, 4] -= 1.0
        additive_mask[3, :, 2] += 0.25
        self._compare(
            "additive-mask-per-head-weights",
            common,
            (4, 2, 8),
            (5, 2, 8),
            (5, 2, 8),
            attn_mask=additive_mask,
            average_attn_weights=False,
        )
        self._compare(
            "separate-key-value-dimensions",
            {"embed_dim": 8, "num_heads": 2, "kdim": 6, "vdim": 7},
            (4, 2, 8),
            (5, 2, 6),
            (5, 2, 7),
            need_weights=False,
        )
        self._compare(
            "bias-and-zero-attention",
            {
                "embed_dim": 8,
                "num_heads": 2,
                "add_bias_kv": True,
                "add_zero_attn": True,
            },
            (4, 2, 8),
            (5, 2, 8),
            (5, 2, 8),
        )

    def test_self_attention_backward_matches_torch(self):
        rng = np.random.RandomState(4017)
        x_array = rng.randn(8, 8, 16).astype("float32")
        target_array = rng.randn(8, 8, 16).astype("float32")
        padding_array = np.zeros((8, 8), dtype=bool)
        padding_array[1, -2:] = True
        padding_array[3, -1:] = True
        padding_array[5, 2:4] = True

        for device, use_cuda in (("cpu", 0), ("cuda", 1)):
            if device == "cuda" and not _test_capability.check_accelerator(
                    "cuda", backend=jt).enabled:
                continue
            torch_device = device if torch.cuda.is_available() else "cpu"
            for need_weights in (False, True):
                with self.subTest(device=device, need_weights=need_weights):
                    torch_module = tnn.MultiheadAttention(
                        16, 2, batch_first=True, device=torch_device).eval()
                    torch_x = torch.from_numpy(x_array).to(torch_device)
                    torch_target = torch.from_numpy(target_array).to(torch_device)
                    torch_padding = torch.from_numpy(padding_array).to(torch_device)
                    torch_output, torch_weights = torch_module(
                        torch_x, torch_x, torch_x,
                        key_padding_mask=torch_padding,
                        need_weights=need_weights)
                    torch_loss = ((torch_output - torch_target) ** 2).mean()
                    torch_loss.backward()
                    torch.cuda.synchronize() if torch_device == "cuda" else None

                    with jt.flag_scope(use_cuda=use_cuda):
                        jittor_module = jnn.MultiheadAttention(
                            16, 2, batch_first=True).eval()
                        if use_cuda:
                            jittor_module = jittor_module.cuda(0)
                        for name, parameter in jittor_module.named_parameters():
                            source = dict(torch_module.named_parameters())[name]
                            value = jt.array(source.detach().cpu().numpy())
                            parameter.update(value.cuda(0) if use_cuda else value)
                        jittor_x = jt.array(x_array)
                        jittor_target = jt.array(target_array)
                        jittor_padding = jt.array(padding_array)
                        if use_cuda:
                            jittor_x = jittor_x.cuda(0)
                            jittor_target = jittor_target.cuda(0)
                            jittor_padding = jittor_padding.cuda(0)
                        jittor_output, jittor_weights = jittor_module(
                            jittor_x, jittor_x, jittor_x,
                            key_padding_mask=jittor_padding,
                            need_weights=need_weights)
                        jittor_loss = ((jittor_output - jittor_target) ** 2).mean()
                        parameters = list(jittor_module.parameters())
                        gradients = jt.grad(jittor_loss, parameters)
                        jt.sync([jittor_loss] + gradients, weak_sync=False)
                        actual_loss = float(jittor_loss.item())
                        actual_gradients = [gradient.numpy() for gradient in gradients]

                    self.assertEqual(jittor_weights is None, not need_weights)
                    self.assertEqual(torch_weights is None, not need_weights)
                    np.testing.assert_allclose(
                        actual_loss, float(torch_loss.item()), atol=2e-6, rtol=2e-6)
                    torch_gradients = [
                        parameter.grad.detach().cpu().numpy()
                        for parameter in torch_module.parameters()
                    ]
                    for name, actual, expected in zip(
                            (name for name, _ in jittor_module.named_parameters()),
                            actual_gradients, torch_gradients):
                        np.testing.assert_allclose(
                            actual, expected, atol=2e-6, rtol=2e-6,
                            err_msg="{} need_weights={} {} backward".format(
                                device, need_weights, name))

    def test_self_attention_cls_only_backward_matches_torch(self):
        rng = np.random.RandomState(4018)
        x_array = rng.randn(8, 17, 16).astype("float32")
        target_array = rng.randn(8, 1, 16).astype("float32")

        for device, use_cuda in (("cpu", 0), ("cuda", 1)):
            if device == "cuda" and not _test_capability.check_accelerator(
                    "cuda", backend=jt).enabled:
                continue
            torch_device = device if torch.cuda.is_available() else "cpu"
            for need_weights in (False, True):
                with self.subTest(device=device, need_weights=need_weights):
                    torch.manual_seed(4019)
                    torch_module = tnn.MultiheadAttention(
                        16, 2, batch_first=True, device=torch_device).eval()
                    torch_x = torch.from_numpy(x_array).to(torch_device)
                    torch_target = torch.from_numpy(target_array).to(torch_device)
                    torch_output, _ = torch_module(
                        torch_x, torch_x, torch_x, need_weights=need_weights)
                    torch_loss = ((torch_output[:, :1] - torch_target) ** 2).mean()
                    torch_loss.backward()
                    if torch_device == "cuda":
                        torch.cuda.synchronize()

                    with jt.flag_scope(use_cuda=use_cuda):
                        jittor_module = jnn.MultiheadAttention(
                            16, 2, batch_first=True).eval()
                        if use_cuda:
                            jittor_module = jittor_module.cuda(0)
                        torch_parameters = dict(torch_module.named_parameters())
                        for name, parameter in jittor_module.named_parameters():
                            value = jt.array(
                                torch_parameters[name].detach().cpu().numpy())
                            parameter.update(value.cuda(0) if use_cuda else value)
                        jittor_x = jt.array(x_array)
                        jittor_target = jt.array(target_array)
                        if use_cuda:
                            jittor_x = jittor_x.cuda(0)
                            jittor_target = jittor_target.cuda(0)
                        jittor_output, _ = jittor_module(
                            jittor_x, jittor_x, jittor_x,
                            need_weights=need_weights)
                        jittor_loss = (
                            (jittor_output[:, :1] - jittor_target) ** 2).mean()
                        gradients = jt.grad(
                            jittor_loss, list(jittor_module.parameters()))
                        jt.sync([jittor_loss] + gradients, weak_sync=False)

                    np.testing.assert_allclose(
                        float(jittor_loss.item()),
                        float(torch_loss.item()),
                        atol=2e-6,
                        rtol=2e-6)
                    torch_gradients = [
                        parameter.grad.detach().cpu().numpy()
                        for parameter in torch_module.parameters()
                    ]
                    for (name, _), actual, expected in zip(
                            jittor_module.named_parameters(),
                            [gradient.numpy() for gradient in gradients],
                            torch_gradients):
                        np.testing.assert_allclose(
                            actual,
                            expected,
                            atol=2e-6,
                            rtol=2e-6,
                            err_msg=(
                                "{} need_weights={} {} backward".format(
                                    device, need_weights, name)))

    def test_cross_attention_backward_matches_torch(self):
        rng = np.random.RandomState(20261016)
        query_array = rng.normal(0, 0.2, (8, 8, 16)).astype("float32")
        key_array = rng.normal(0, 0.2, (8, 8, 16)).astype("float32")
        value_array = rng.normal(0, 0.2, (8, 8, 16)).astype("float32")
        target_array = rng.normal(0, 0.2, (8, 8, 16)).astype("float32")
        padding_array = np.zeros((8, 8), dtype=bool)
        padding_array[1, -2:] = True
        padding_array[3, -1:] = True
        padding_array[5, 2:4] = True

        for device, use_cuda in (("cpu", 0), ("cuda", 1)):
            if device == "cuda" and not _test_capability.check_accelerator(
                    "cuda", backend=jt).enabled:
                continue
            torch_device = device if torch.cuda.is_available() else "cpu"
            for batch_first in (False, True):
                if batch_first:
                    query = query_array
                    key = key_array
                    value = value_array
                else:
                    query = query_array.transpose(1, 0, 2)
                    key = key_array.transpose(1, 0, 2)
                    value = value_array.transpose(1, 0, 2)
                target = target_array if batch_first else target_array.transpose(1, 0, 2)
                for shared_kv in (False, True):
                    value_input = key if shared_kv else value
                    for need_weights in (False, True):
                        with self.subTest(
                                device=device,
                                batch_first=batch_first,
                                shared_kv=shared_kv,
                                need_weights=need_weights):
                            torch.manual_seed(20261017)
                            torch_module = tnn.MultiheadAttention(
                                16, 2, batch_first=batch_first,
                                device=torch_device).eval()
                            torch_query = torch.tensor(
                                query.copy(), device=torch_device,
                                requires_grad=True)
                            torch_key = torch.tensor(
                                key.copy(), device=torch_device,
                                requires_grad=True)
                            torch_value = (
                                torch_key if shared_kv else torch.tensor(
                                    value.copy(), device=torch_device,
                                    requires_grad=True))
                            torch_target = torch.tensor(
                                target.copy(), device=torch_device)
                            torch_padding = torch.from_numpy(
                                padding_array).to(torch_device)
                            torch_output, torch_weights = torch_module(
                                torch_query,
                                torch_key,
                                torch_value,
                                key_padding_mask=torch_padding,
                                need_weights=need_weights)
                            torch_loss = ((torch_output - torch_target) ** 2).mean()
                            torch_loss.backward()
                            if torch_device == "cuda":
                                torch.cuda.synchronize()

                            with jt.flag_scope(use_cuda=use_cuda):
                                jittor_module = jnn.MultiheadAttention(
                                    16, 2, batch_first=batch_first).eval()
                                if use_cuda:
                                    jittor_module = jittor_module.cuda(0)
                                torch_parameters = dict(torch_module.named_parameters())
                                for name, parameter in jittor_module.named_parameters():
                                    source = torch_parameters[name]
                                    parameter_value = jt.array(
                                        source.detach().cpu().numpy())
                                    parameter.update(
                                        parameter_value.cuda(0)
                                        if use_cuda else parameter_value)

                                jittor_query = jt.array(query)
                                jittor_key = jt.array(key)
                                jittor_value = jt.array(value_input)
                                jittor_target = jt.array(target)
                                jittor_padding = jt.array(padding_array)
                                if use_cuda:
                                    jittor_query = jittor_query.cuda(0)
                                    jittor_key = jittor_key.cuda(0)
                                    jittor_value = jittor_value.cuda(0)
                                    jittor_target = jittor_target.cuda(0)
                                    jittor_padding = jittor_padding.cuda(0)
                                jittor_output, jittor_weights = jittor_module(
                                    jittor_query,
                                    jittor_key,
                                    jittor_key if shared_kv else jittor_value,
                                    key_padding_mask=jittor_padding,
                                    need_weights=need_weights)
                                jittor_loss = (
                                    (jittor_output - jittor_target) ** 2).mean()
                                gradients = jt.grad(
                                    jittor_loss, list(jittor_module.parameters()))
                                jt.sync([jittor_loss] + gradients, weak_sync=False)
                                actual_loss = float(jittor_loss.item())
                                actual_output = jittor_output.numpy()
                                actual_gradients = [
                                    gradient.numpy() for gradient in gradients]

                            self.assertEqual(jittor_weights is None, not need_weights)
                            self.assertEqual(torch_weights is None, not need_weights)
                            np.testing.assert_allclose(
                                actual_loss,
                                float(torch_loss.item()),
                                atol=2e-6,
                                rtol=2e-6)
                            np.testing.assert_allclose(
                                actual_output,
                                torch_output.detach().cpu().numpy(),
                                atol=2e-6,
                                rtol=2e-6)
                            torch_gradients = [
                                parameter.grad.detach().cpu().numpy()
                                for parameter in torch_module.parameters()
                            ]
                            names = [
                                name for name, _ in jittor_module.named_parameters()]
                            for name, actual, expected in zip(
                                    names, actual_gradients, torch_gradients):
                                np.testing.assert_allclose(
                                    actual,
                                    expected,
                                    atol=2e-6,
                                    rtol=2e-6,
                                    err_msg=(
                                        "{} batch_first={} shared_kv={} "
                                        "need_weights={} {} backward".format(
                                            device, batch_first, shared_kv,
                                            need_weights, name)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
