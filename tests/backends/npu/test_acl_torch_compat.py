import unittest

import numpy as np

import jittor as torch
import jittor as jt


@unittest.skipIf(not jt.compiler.has_acl, "No ACL found")
class TestACLTorchCompat(unittest.TestCase):
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
            self.assertEqual(str(quotient.dtype), "float32")
            self.assertEqual(str(reflected.dtype), "float32")
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
        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

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
                messages = [entry["msg"].lower() for entry in logs]
                self.assertFalse(
                    any("compile cpu" in message for message in messages)
                )
                self.assertFalse(
                    any("fallback cpu" in message for message in messages)
                )

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
        acl_group_norm = jt.nn._group_norm_cuda

        def record_group_norm(*args):
            result = acl_group_norm(*args)
            native_dispatches.append(result is not None)
            return result

        jt.nn._group_norm_cuda = record_group_norm
        try:
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
        finally:
            jt.nn._group_norm_cuda = acl_group_norm

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
        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

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
        acl_batch_norm = jt.nn._batch_norm_eval_cuda

        def record_batch_norm(*args):
            result = acl_batch_norm(*args)
            dispatches.append(result is not None)
            return result

        jt.nn._batch_norm_eval_cuda = record_batch_norm
        try:
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
        finally:
            jt.nn._batch_norm_eval_cuda = acl_batch_norm

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
        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

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
        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

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
        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_silu_forward_backward_stays_on_acl(self):
        source_np = np.array(
            [-4.0, -1.25, -0.1, 0.0, 0.75, 3.5], dtype="float32"
        )
        loss_weight_np = np.array(
            [0.25, -0.5, 1.5, 2.0, -1.0, 0.75], dtype="float32"
        )

        self.assertTrue(jt.flags.use_acl)
        self.assertTrue(jt.flags.use_cuda)
        source = torch.tensor(source_np)
        source.requires_grad_(True)
        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            output = torch.nn.functional.silu(source)
            gradient = torch.autograd.grad(
                (output * torch.tensor(loss_weight_np)).sum(), source
            )[0]
            candidate = jt.fetch_sync([output, gradient])

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
        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

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
        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

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
        acl_attention = jt.nn._acl_scaled_dot_product_attention

        def record_attention(*args, **kwargs):
            result = acl_attention(*args, **kwargs)
            native_dispatches.append(result is not None)
            return result

        jt.nn._acl_scaled_dot_product_attention = record_attention
        try:
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
                        else torch.tensor(mask_np).stop_grad()
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
        finally:
            jt.nn._acl_scaled_dot_product_attention = acl_attention

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
        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

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
        np.testing.assert_allclose(output, [-0.8, -0.2, 4.0, 12.0])
        np.testing.assert_allclose(gradient, [0.4, 0.4, 4.0, 4.0])
        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))

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
        with jt.flag_scope(use_acl=0, use_cuda=0):
            self.assertEqual(torch.get_default_device().type, "cpu")
        with jt.flag_scope(use_acl=1, use_cuda=1):
            self.assertEqual(torch.get_default_device().type, "cuda")

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_constant_pad_forward_backward_stays_on_acl(self):
        acl_pad = jt.nn._acl_constant_pad
        calls = []

        def record_acl_pad(x, amounts, value):
            calls.append((tuple(amounts), value))
            return acl_pad(x, amounts, value)

        jt.nn._acl_constant_pad = record_acl_pad
        try:
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
        finally:
            jt.nn._acl_constant_pad = acl_pad

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
        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))
