
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy
# ***************************************************************
# CUDA device regression for transformers through the direct alias path:
#
#     import torch
#
# No transformers source files are modified. This test focuses on the PyTorch
# style device API used by real HF code: torch.device("cuda"), tensor(...,
# device=...), Tensor.to(device), and Module.to(device).
# ***************************************************************
import os
import sys
import unittest
from _helpers import capability as _test_capability

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

try:
    import torch
    import jittor as jt

    _ALIAS_REGISTERED = sys.modules.get("torch") is torch
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
    )

    _HAS = hasattr(torch, "tensor")
except Exception:
    AutoConfig = AutoModel = AutoModelForCausalLM = AutoModelForSeq2SeqLM = None
    _ALIAS_REGISTERED = False
    _HAS = False


_CASES = {
    "bert": (
        AutoModel,
        dict(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            vocab_size=50,
            max_position_embeddings=16,
            pad_token_id=0,
        ),
    ),
    "gpt2": (
        AutoModelForCausalLM,
        dict(n_layer=1, n_embd=16, n_head=2, vocab_size=50, n_positions=16, pad_token_id=0),
    ),
    "t5": (
        AutoModel,
        dict(
            d_model=16,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            d_kv=8,
            vocab_size=50,
            pad_token_id=0,
            decoder_start_token_id=0,
        ),
    ),
}

_T5_GENERATION_CONFIG = dict(
    d_model=64, d_ff=128, num_layers=2, num_heads=2, d_kv=32,
    vocab_size=128, dropout_rate=0.5, pad_token_id=0,
    decoder_start_token_id=0, eos_token_id=1,
)
_MIXTRAL_CONFIG = dict(
    hidden_size=64, intermediate_size=128, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=1, vocab_size=128,
    max_position_embeddings=128, num_local_experts=2,
    num_experts_per_tok=2, attention_dropout=0.0, pad_token_id=0,
    bos_token_id=1, eos_token_id=127, use_cache=True,
)


def _output_tensor(out):
    logits = getattr(out, "logits", None)
    if logits is not None:
        return logits
    return out.last_hidden_state


def _inputs(torch, cfg, vocab_size, device):
    ids = torch.tensor(
        np.random.randint(0, int(vocab_size), (2, 8)).astype("int64"),
        device=device,
    )
    data = {
        "input_ids": ids,
        "attention_mask": torch.ones((2, 8), dtype=torch.long, device=device),
    }
    if getattr(cfg, "is_encoder_decoder", False):
        data["decoder_input_ids"] = ids.to(device)
    return data


@_test_preserve_policy(jt, 'use_cuda')
@unittest.skipUnless(_HAS, "needs jittor torch alias + transformers")
class TestTorchHFCudaDevice(unittest.TestCase):
    def setUp(self):
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        if not _test_capability.check_accelerator("cuda", backend=jt).enabled:
            self.skipTest("needs CUDA")
        # Restored in tearDown: without it this class turned CUDA on for every
        # test that ran after it, in every file, for the rest of the session.
        self._previous_use_cuda = jt.introspection.policy.runtime.use_cuda
        _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1))

    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            jt.sync_all()
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self._previous_use_cuda))

    def test_direct_alias_registers_torch_and_cuda(self):
        self.assertTrue(_ALIAS_REGISTERED)
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(str(torch.device("cuda")), "cuda")

    def test_explicit_cuda_tensor_device_roundtrip(self):
        device = torch.device("cuda")
        x = torch.tensor(np.arange(6, dtype="float32").reshape(2, 3), device=device)
        self.assertTrue(x.is_cuda)
        self.assertEqual(x.device.type, "cuda")

        cpu = x.cpu()
        self.assertFalse(cpu.is_cuda)
        self.assertEqual(cpu.device.type, "cpu")

        back = cpu.to(device)
        self.assertTrue(back.is_cuda)
        self.assertEqual(back.device.type, "cuda")
        self.assertTrue(np.allclose(back.clone().cpu().numpy(), x.clone().cpu().numpy()))

    def test_explicit_cuda_empty_tensor(self):
        device = torch.device("cuda")
        empty = torch.tensor([], dtype=torch.float32, device=device)

        self.assertEqual(empty.numel(), 0)
        self.assertTrue(empty.is_cuda)
        self.assertEqual(empty.device.type, "cuda")

        value = torch.tensor([1.0], dtype=torch.float32, device=device)
        joined = torch.cat((empty, value))
        self.assertTrue(np.array_equal(joined.cpu().numpy(), np.array([1.0], dtype=np.float32)))

    def test_cuda_bool_mask_inplace_multiply_keeps_extremes_finite(self):
        values = torch.tensor(
            [[-3.4028235e38, -3.4028235e38],
             [-3.4028235e38, -3.4028235e38]],
            dtype=torch.float32, device="cuda")
        mask = torch.tensor(
            [[True, False], [False, True]], dtype=torch.bool, device="cuda")

        values *= mask

        actual = values.cpu().numpy()
        self.assertFalse(np.isnan(actual).any())
        np.testing.assert_array_equal(
            actual,
            np.asarray([[-3.4028235e38, 0.0], [0.0, -3.4028235e38]],
                       dtype=np.float32),
        )

        # Sampling helpers receive tensor inputs but are not named ``*_like``;
        # their internal random tensors must inherit the input's CUDA device.
        sampled = torch.multinomial(
            torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32, device="cuda"),
            num_samples=1,
        )
        self.assertTrue(sampled.is_cuda)

    def test_implicit_data_constructors_default_to_cpu(self):
        source = np.array([1, 2, 3], dtype=np.int64)
        values = {
            "tensor_list": torch.tensor([1, 2, 3]),
            "tensor_numpy": torch.tensor(source),
            "as_tensor": torch.as_tensor(source),
            "from_numpy": torch.from_numpy(source),
            # Data factories must keep Torch's CPU default even when Jittor's
            # process-wide CUDA backend is enabled.
            "arange": torch.arange(3),
            "arange_none": torch.arange(3, device=None),
            "zeros": torch.zeros(3, dtype=torch.int64),
            "zeros_none": torch.zeros(3, dtype=torch.int64, device=None),
        }
        for name, value in values.items():
            with self.subTest(constructor=name):
                self.assertFalse(value.is_cuda)
                self.assertEqual(value.device.type, "cpu")
                self.assertEqual(value.dtype, torch.int64)
                expected = (
                    np.arange(3, dtype=np.int64)
                    if name.startswith("arange")
                    else np.zeros(3, dtype=np.int64)
                    if name.startswith("zeros")
                    else source
                )
                self.assertTrue(np.array_equal(value.numpy(), expected))

        float_range = torch.arange(torch.tensor(0.0), torch.tensor(3.0))
        self.assertEqual(float_range.dtype, torch.float32)
        self.assertTrue(
            np.array_equal(
                float_range.numpy(), np.array([0.0, 1.0, 2.0], dtype=np.float32)
            )
        )

        cuda = torch.tensor(source, device="cuda")
        self.assertTrue(torch.as_tensor(cuda).is_cuda)
        self.assertTrue(torch.tensor(cuda).is_cuda)

        cpu = torch.tensor(source)
        self.assertFalse(torch.as_tensor(cpu).is_cuda)
        self.assertIs(torch.as_tensor(cpu), cpu)
        copied_cpu = torch.tensor(cpu)
        converted_cpu = torch.tensor(cpu, dtype=torch.float32)
        copied_cuda = torch.tensor(cpu, device="cuda")
        cpu.sync()
        self.assertFalse(copied_cpu.is_cuda)
        self.assertFalse(converted_cpu.is_cuda)
        self.assertEqual(converted_cpu.dtype, torch.float32)
        self.assertTrue(copied_cuda.is_cuda)
        self.assertFalse(cpu.is_cuda)
        self.assertEqual(cpu.location(), "cpu")

        converted_cpu = torch.as_tensor(cpu, dtype=torch.float32)
        converted_cpu.sync()
        self.assertFalse(converted_cpu.is_cuda)
        self.assertEqual(converted_cpu.device.type, "cpu")
        self.assertEqual(converted_cpu.location(), "cpu")
        self.assertEqual(converted_cpu.dtype, torch.float32)

        meta = torch.as_tensor(cpu, device="meta")
        self.assertTrue(meta.is_meta)
        self.assertIsNot(meta, cpu)
        self.assertFalse(cpu.is_meta)

    def test_input_factories_inherit_device_inside_meta_context(self):
        cpu = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        cuda = cpu.to("cuda")
        with torch.device("meta"):
            shape_default = torch.zeros(2)
            cpu_like = torch.zeros_like(cpu)
            cpu_empty_like = torch.empty_like(cpu)
            cpu_empty_like_none = torch.empty_like(cpu, dtype=None, device=None)
            cpu_tril = torch.tril(cpu)
            cuda_like = torch.ones_like(cuda)

        self.assertTrue(shape_default.is_meta)
        for value in (cpu_like, cpu_empty_like, cpu_empty_like_none, cpu_tril):
            value.sync()
            self.assertFalse(value.is_meta)
            self.assertFalse(value.is_cuda)
            self.assertEqual(value.device.type, "cpu")
            self.assertEqual(value.location(), "cpu")
        self.assertEqual(cpu_empty_like_none.dtype, cpu.dtype)
        self.assertFalse(cuda_like.is_meta)
        self.assertTrue(cuda_like.is_cuda)
        self.assertEqual(cuda_like.device.type, "cuda")

    def test_cpu_host_export_does_not_change_source_residency(self):
        source = torch.tensor([1, 2, 3], dtype=torch.int64)
        host = source.detach().cpu().numpy()

        self.assertFalse(source.is_cuda)
        self.assertEqual(source.device.type, "cpu")
        self.assertEqual(source.location(), "cpu")
        self.assertTrue(np.array_equal(host, np.array([1, 2, 3], dtype=np.int64)))

        moved = source.to(torch.device("cuda"))
        moved.sync()
        self.assertTrue(moved.is_cuda)
        self.assertEqual(moved.device.type, "cuda")
        self.assertEqual(moved.location(), "device")
        self.assertFalse(source.is_cuda)
        self.assertEqual(source.location(), "cpu")

    def test_accelerate_meta_tied_bias_checkpoint_retie(self):
        try:
            from accelerate import init_empty_weights as accelerate_init_empty_weights
            from transformers.integrations.accelerate import (
                init_empty_weights as transformers_init_empty_weights,
            )
        except ImportError:
            self.skipTest("needs accelerate")

        class TiedBias(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = torch.nn.Linear(1, 2, bias=False)
                self.embedding = torch.nn.Embedding(3, 2)
                self.bias = torch.nn.Parameter(torch.zeros(2))
                self.decoder.bias = self.bias

            def roberta_tie_weights(self):
                if self.decoder.bias.device.type == "meta":
                    self.decoder.bias = self.bias
                    return "decoder_from_bias"
                self.bias = self.decoder.bias
                return "bias_from_decoder"

        for provider, init_empty_weights in (
            ("accelerate", accelerate_init_empty_weights),
            ("transformers", transformers_init_empty_weights),
        ):
            with self.subTest(provider=provider):
                with init_empty_weights():
                    model = TiedBias()

                self.assertTrue(model.decoder.weight.is_meta)
                self.assertTrue(model.embedding.weight.is_meta)
                self.assertTrue(model.bias.is_meta)
                self.assertTrue(model.decoder.bias.is_meta)
                canonical = torch.tensor([-0.75, 0.5], device="cuda")
                model.load_state_dict({"bias": canonical}, strict=False, assign=True)
                self.assertFalse(model.bias.is_meta)
                self.assertTrue(model.decoder.bias.is_meta)
                self.assertEqual(model.roberta_tie_weights(), "decoder_from_bias")
                self.assertIs(model.bias, model.decoder.bias)
                np.testing.assert_array_equal(
                    model.bias.detach().cpu().numpy(),
                    np.array([-0.75, 0.5], dtype=np.float32),
                )

    def test_t5_batched_generate_and_encoder_decoder_cache(self):
        device = torch.device("cuda")
        config = AutoConfig.for_model("t5", **_T5_GENERATION_CONFIG)
        model = AutoModelForSeq2SeqLM.from_config(config).to(device).eval()
        self.assertTrue(all(parameter.is_cuda for parameter in model.parameters()))
        input_ids = torch.tensor(
            np.array([[5, 6, 7, 1], [8, 9, 10, 1]], dtype="int64"),
            device=device,
        )
        attention_mask = torch.ones((2, 4), dtype=torch.long, device=device)

        generation = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            min_new_tokens=2, max_new_tokens=2, do_sample=False,
            num_beams=1, use_cache=True, return_dict_in_generate=True,
            output_scores=True,
        )
        self.assertEqual(tuple(generation.sequences.shape), (2, 3))
        self.assertTrue(generation.sequences.is_cuda)
        self.assertEqual(len(generation.scores), 2)
        for score in generation.scores:
            self.assertTrue(score.is_cuda)
            values = score.float().numpy()
            self.assertFalse(np.isnan(values).any())
            self.assertTrue(np.isneginf(values[:, config.eos_token_id]).all())
            self.assertTrue(
                np.isfinite(np.delete(values, config.eos_token_id, axis=1)).all())

        decoder_prefix = torch.tensor(
            np.array([[0, 11], [0, 12]], dtype="int64"), device=device)
        with torch.no_grad():
            prefill = model(
                input_ids=input_ids, attention_mask=attention_mask,
                decoder_input_ids=decoder_prefix, use_cache=True,
                return_dict=True,
            )
            prefill_cache = prefill.past_key_values
            prefill_length = int(prefill_cache.get_seq_length())
            step_tokens = torch.tensor(
                np.array([[13], [14]], dtype="int64"), device=device)
            cached = model(
                input_ids=input_ids, attention_mask=attention_mask,
                decoder_input_ids=step_tokens,
                past_key_values=prefill_cache,
                use_cache=True, return_dict=True,
            )
            full = model(
                input_ids=input_ids, attention_mask=attention_mask,
                decoder_input_ids=torch.cat([decoder_prefix, step_tokens], dim=1),
                use_cache=False, return_dict=True,
            )
        self.assertEqual(prefill_length, 2)
        self.assertIs(cached.past_key_values, prefill_cache)
        self.assertEqual(int(cached.past_key_values.get_seq_length()), 3)
        self.assertTrue(cached.logits.is_cuda)
        self.assertTrue(
            cached.past_key_values.self_attention_cache.layers[0].keys.is_cuda)
        self.assertTrue(np.isfinite(cached.logits.float().numpy()).all())
        np.testing.assert_allclose(
            cached.logits[:, -1].float().numpy(),
            full.logits[:, -1].float().numpy(), atol=1e-5, rtol=1e-5)

    def test_mixtral_router_grad_cache_and_generate(self):
        device = torch.device("cuda")
        config = AutoConfig.for_model("mixtral", **_MIXTRAL_CONFIG)
        config._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_config(config).to(device)
        self.assertTrue(all(parameter.is_cuda for parameter in model.parameters()))
        input_ids = torch.tensor(
            np.array([[1, 5, 6, 7, 8], [1, 9, 10, 11, 12]], dtype="int64"),
            device=device,
        )
        attention_mask = torch.ones((2, 5), dtype=torch.long, device=device)

        model.train()
        output = model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=input_ids, use_cache=False, output_router_logits=True,
            return_dict=True,
        )
        self.assertTrue(output.loss.is_cuda)
        self.assertTrue(output.aux_loss.is_cuda)
        self.assertEqual(tuple(output.router_logits[0].shape), (10, 2))
        output.loss.backward()
        trainable = [(name, parameter) for name, parameter in model.named_parameters()
                     if parameter.requires_grad]
        missing = [name for name, parameter in trainable if parameter.grad is None]
        self.assertEqual(missing, [], "parameters without gradients: %s" % missing)
        for name, parameter in trainable:
            self.assertTrue(parameter.grad.is_cuda, name)
            self.assertTrue(np.isfinite(parameter.grad.float().numpy()).all(), name)
        routed = [(name, parameter.grad.float().numpy())
                  for name, parameter in trainable
                  if ".block_sparse_moe.gate." in name
                  or ".block_sparse_moe.experts." in name]
        self.assertEqual(sum(".gate." in name for name, _ in routed), 1)
        self.assertEqual(sum(".experts." in name for name, _ in routed), 6)
        for name, gradient in routed:
            self.assertTrue(np.any(np.abs(gradient) > 0), name)

        model.zero_grad()
        model.eval()
        with torch.no_grad():
            prefill = model(
                input_ids=input_ids[:, :4], attention_mask=attention_mask[:, :4],
                use_cache=True, output_router_logits=False, return_dict=True)
            prefill_cache = prefill.past_key_values
            prefill_length = int(prefill_cache.get_seq_length())
            cached = model(
                input_ids=input_ids[:, 4:], attention_mask=attention_mask,
                past_key_values=prefill_cache, use_cache=True,
                output_router_logits=False, return_dict=True)
            cached_last = cached.logits[:, -1].float().numpy()
            full = model(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=False, output_router_logits=False, return_dict=True)
            full_last = full.logits[:, -1].float().numpy()
            generated = model.generate(
                input_ids=input_ids[:, :4], attention_mask=attention_mask[:, :4],
                min_new_tokens=2, max_new_tokens=2, do_sample=False,
                num_beams=2, use_cache=True)
        self.assertEqual(prefill_length, 4)
        self.assertIs(cached.past_key_values, prefill_cache)
        self.assertEqual(int(cached.past_key_values.get_seq_length()), 5)
        self.assertTrue(cached.logits.is_cuda)
        self.assertTrue(generated.is_cuda)
        self.assertEqual(tuple(generated.shape), (2, 6))
        np.testing.assert_allclose(
            cached_last, full_last, atol=1e-5, rtol=1e-5)

    def test_small_transformers_cuda_device_forward(self):
        device = torch.device("cuda")
        for arch, (cls, kwargs) in _CASES.items():
            with self.subTest(model=arch):
                cfg = AutoConfig.for_model(arch, **kwargs)
                model = cls.from_config(cfg)
                model.cpu()
                self.assertTrue(any(not p.is_cuda for p in model.parameters()))
                model.to(device)
                self.assertTrue(all(p.is_cuda for p in model.parameters()))
                model.eval()

                inputs = _inputs(torch, cfg, kwargs["vocab_size"], device)
                self.assertTrue(all(getattr(v, "is_cuda", False) for v in inputs.values()))
                with torch.no_grad():
                    y1 = _output_tensor(model(**inputs)).float()
                    y2 = _output_tensor(model(**inputs)).float()
                    jt.sync_all(True)
                self.assertTrue(y1.is_cuda)
                a1 = y1.clone().cpu().numpy()
                a2 = y2.clone().cpu().numpy()
                self.assertTrue(np.isfinite(a1).all(), f"{arch} produced non-finite values")
                self.assertTrue(np.allclose(a1, a2, atol=1e-5), f"{arch} eval forward is unstable")


if __name__ == "__main__":
    unittest.main()
