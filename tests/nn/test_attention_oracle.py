"""Independent PyTorch oracle coverage for canonical multi-head attention."""

import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
