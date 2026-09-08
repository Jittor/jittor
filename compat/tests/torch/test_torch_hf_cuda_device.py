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
        # Restored in tearDown: without it this class turned CUDA on for every
        # test that ran after it, in every file, for the rest of the session.
        self._previous_use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.sync_all()
        jt.flags.use_cuda = self._previous_use_cuda

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
