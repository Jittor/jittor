"""CPU regressions for verl through Jittor's independent Torch frontend."""

import importlib.metadata
import importlib.util
import os
import unittest

import numpy as np


_HAS_VERL = importlib.util.find_spec("verl") is not None
_REQUIRE_VERL = os.environ.get("JITTOR_REQUIRE_VERL") == "1"
_RUN_MODEL_SMOKE = os.environ.get("JITTOR_VERL_MODEL_SMOKE") == "1"
if _REQUIRE_VERL and not _HAS_VERL:
    raise ModuleNotFoundError("JITTOR_REQUIRE_VERL=1 but verl is not installed")


@unittest.skipUnless(_HAS_VERL, "verl is not installed")
class TestVerlCompat(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        if os.environ.get("JITTOR_VERL_RUNTIME") == "oracle":
            self.assertTrue(hasattr(torch, "_C"), "oracle must be an official PyTorch build")
            self.assertFalse(
                hasattr(torch, "_torch_compat_install_context"),
                "oracle resolves to the Jittor shim",
            )
            return

        import jittor
        from jittor.compat.shim.runtime import activation_status
        from jittor.compat.torch.publication import namespace_owner

        self.assertTrue(
            namespace_owner(torch) is jittor
            and activation_status(jittor).active
            and torch.Tensor is not jittor.Var,
            "verl must run through the independent Jittor Torch frontend",
        )
        self.assertEqual(jittor.flags.use_cuda, 0, "this verl regression is CPU-only")

    @staticmethod
    def _numpy(tensor):
        return tensor.detach().cpu().numpy()

    def test_supported_release_and_transformers_torch_identity(self):
        import verl
        from transformers.utils import import_utils

        self.assertEqual(importlib.metadata.version("verl"), "0.9.1")
        self.assertEqual(verl.__version__, "0.9.1")
        self.assertTrue(import_utils.is_torch_available())
        if os.environ.get("JITTOR_VERL_RUNTIME") != "oracle":
            self.assertEqual(import_utils.get_torch_version(), self.torch.__torch_version__)

    def test_dataproto_chunk_repeat_and_cpu_transfer(self):
        from verl.protocol import DataProto

        proto = DataProto.from_dict(
            tensors={
                "input_ids": self.torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
                "attention_mask": self.torch.tensor([[1, 1], [1, 1], [1, 0], [1, 0]]),
            },
            non_tensors={"uid": np.array(["a", "b", "c", "d"], dtype=object)},
            meta_info={"source": "jittor-verl-regression"},
        ).to("cpu")

        chunks = proto.chunk(2)
        self.assertEqual([len(chunk) for chunk in chunks], [2, 2])
        np.testing.assert_array_equal(self._numpy(chunks[1].batch["input_ids"]), [[5, 6], [7, 8]])
        np.testing.assert_array_equal(chunks[0].non_tensor_batch["uid"], ["a", "b"])
        self.assertEqual(chunks[0].meta_info["source"], "jittor-verl-regression")

        repeated = chunks[0].repeat(2, interleave=True)
        np.testing.assert_array_equal(
            self._numpy(repeated.batch["input_ids"]),
            [[1, 2], [1, 2], [3, 4], [3, 4]],
        )
        np.testing.assert_array_equal(repeated.non_tensor_batch["uid"], ["a", "a", "b", "b"])

    def test_ppo_loss_gradient_and_grpo_advantage_match_oracle(self):
        from verl.trainer.ppo.core_algos import (
            compute_grpo_outcome_advantage,
            compute_policy_loss,
        )

        old_log_prob = self.torch.tensor([[-1.0, -0.5, -0.2], [-0.3, -0.8, -1.2]])
        log_prob = self.torch.tensor(
            [[-0.9, -0.8, -0.1], [-0.4, -0.6, -1.5]], requires_grad=True
        )
        advantages = self.torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.25, 0.5]])
        response_mask = self.torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        loss, clip_fraction, ppo_kl, lower_clip_fraction = compute_policy_loss(
            old_log_prob,
            log_prob,
            advantages,
            response_mask,
            cliprange=0.2,
        )
        loss.backward()

        np.testing.assert_allclose(
            [loss.item(), clip_fraction.item(), ppo_kl.item(), lower_clip_fraction.item()],
            [-0.0941485539, 0.4, 0.0799999908, 0.0],
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            self._numpy(log_prob.grad),
            [[-0.221034199, 0.0, 0.0], [0.180967495, 0.0, -0.074081831]],
            rtol=1e-6,
            atol=1e-7,
        )

        rewards = self.torch.tensor(
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        )
        grpo_mask = self.torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
        )
        result, returns = compute_grpo_outcome_advantage(
            rewards,
            grpo_mask,
            np.array(["a", "a", "b", "b"]),
        )
        expected = np.array(
            [
                [-0.707106292, -0.707106292, -0.707106292],
                [0.707106292, 0.707106292, 0.0],
                [0.707106292, 0.707106292, 0.707106292],
                [-0.707106292, 0.0, 0.0],
            ]
        )
        np.testing.assert_allclose(self._numpy(result), expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(self._numpy(returns), expected, rtol=1e-6, atol=1e-7)

    def test_critic_clipped_value_loss_gradient_and_step(self):
        from verl.trainer.ppo.core_algos import compute_value_loss

        # Interior/tied branches, upper/lower clipped branches, raw-loss branch
        # outside the clip interval, and a masked token. The expected values
        # follow the scalar formula, independently of either framework.
        predictions = self.torch.nn.Parameter(
            self.torch.tensor([[0.1, 0.6, -0.7, 0.8, 9.0]])
        )
        optimizer = self.torch.optim.SGD([predictions], lr=0.1)
        loss, fraction = compute_value_loss(
            vpreds=predictions,
            returns=self.torch.tensor([[0.4, 1.0, -1.0, -0.5, 0.0]]),
            values=self.torch.zeros_like(predictions),
            response_mask=self.torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]]),
            cliprange_value=0.2,
        )
        loss.backward()
        np.testing.assert_allclose([loss.item(), fraction.item()], [0.3825, 0.5], atol=1e-7)
        np.testing.assert_allclose(self._numpy(predictions.grad),
                                   [[-0.075, 0.0, 0.0, 0.325, 0.0]], atol=1e-7)
        optimizer.step()
        np.testing.assert_allclose(self._numpy(predictions),
                                   [[0.1075, 0.6, -0.7, 0.7675, 9.0]], atol=1e-7)

    @unittest.skipUnless(_RUN_MODEL_SMOKE, "set JITTOR_VERL_MODEL_SMOKE=1 for model execution")
    def test_tiny_qwen3_forward_and_backward(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
        )
        model = Qwen3ForCausalLM(config)
        input_ids = self.torch.tensor([[1, 2, 3, 4]], dtype=self.torch.long)
        output = model(input_ids=input_ids, labels=input_ids, use_cache=False)

        self.assertEqual(tuple(output.logits.shape), (1, 4, 32))
        self.assertTrue(self.torch.isfinite(output.loss).item())
        output.loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertEqual(len(gradients), 14)
        self.assertTrue(all(gradient is not None for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
