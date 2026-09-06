# ***************************************************************
# CUDA device regression for transformers through the direct alias path:
#
#     import jittor as torch
#
# No transformers source files are modified. This test focuses on the PyTorch
# style device API used by real HF code: torch.device("cuda"), tensor(...,
# device=...), Tensor.to(device), and Module.to(device).
# ***************************************************************
import os
import sys
import unittest

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

try:
    import jittor as torch
    import jittor as jt

    _ALIAS_REGISTERED = sys.modules.get("torch") is torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    _HAS = hasattr(torch, "tensor")
except Exception:
    AutoConfig = AutoModel = AutoModelForCausalLM = None
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


@unittest.skipUnless(_HAS, "needs jittor torch alias + transformers")
class TestTorchHFCudaDevice(unittest.TestCase):
    def setUp(self):
        if not getattr(jt, "has_cuda", 0):
            self.skipTest("needs CUDA")
        jt.flags.use_cuda = 1

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

    def test_implicit_data_constructors_default_to_cpu(self):
        source = np.array([1, 2, 3], dtype=np.int64)
        values = {
            "tensor_list": torch.tensor([1, 2, 3]),
            "tensor_numpy": torch.tensor(source),
            "as_tensor": torch.as_tensor(source),
            "from_numpy": torch.from_numpy(source),
        }
        for name, value in values.items():
            with self.subTest(constructor=name):
                self.assertFalse(value.is_cuda)
                self.assertEqual(value.device.type, "cpu")
                self.assertEqual(value.dtype, torch.int64)
                self.assertTrue(np.array_equal(value.numpy(), source))

        cuda = torch.tensor(source, device="cuda")
        self.assertTrue(torch.as_tensor(cuda).is_cuda)
        self.assertTrue(torch.tensor(cuda).is_cuda)

        cpu = torch.tensor(source)
        self.assertFalse(torch.as_tensor(cpu).is_cuda)
        self.assertIs(torch.as_tensor(cpu), cpu)
        copied_cpu = torch.tensor(cpu)
        converted_cpu = torch.tensor(cpu, dtype=torch.float32)
        copied_cuda = torch.tensor(cpu, device="cuda")
        self.assertFalse(copied_cpu.is_cuda)
        self.assertFalse(converted_cpu.is_cuda)
        self.assertEqual(converted_cpu.dtype, torch.float32)
        self.assertTrue(copied_cuda.is_cuda)
        self.assertFalse(cpu.is_cuda)
        self.assertEqual(cpu.location(), "cpu")

        converted_cpu = torch.as_tensor(cpu, dtype=torch.float32)
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
