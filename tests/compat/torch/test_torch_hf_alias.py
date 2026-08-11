# ***************************************************************
# Regression test: run small HuggingFace transformers models through the
# direct alias path:
#
#     import jittor as torch
#
# transformers imports torch internally, so the test asserts that jittor's
# compatibility layer registered sys.modules["torch"] before importing
# transformers. No transformers source files are modified.
# ***************************************************************
import os
import sys
import unittest

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

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
    "roberta": (
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


@unittest.skipUnless(_HAS, "needs jittor torch alias + transformers")
class TestTorchHFAlias(unittest.TestCase):
    def test_direct_alias_registers_torch_module(self):
        self.assertTrue(_ALIAS_REGISTERED)

    def test_small_transformers_forward_direct_alias(self):
        for arch, (cls, kwargs) in _CASES.items():
            with self.subTest(model=arch):
                cfg = AutoConfig.for_model(arch, **kwargs)
                model = cls.from_config(cfg)
                model.eval()
                ids = torch.tensor(
                    np.random.randint(0, int(kwargs["vocab_size"]), (2, 8)).astype("int64")
                )
                inputs = {"input_ids": ids}
                if getattr(cfg, "is_encoder_decoder", False):
                    inputs["decoder_input_ids"] = ids
                with torch.no_grad():
                    y1 = _output_tensor(model(**inputs)).float().numpy()
                    y2 = _output_tensor(model(**inputs)).float().numpy()
                self.assertTrue(np.isfinite(y1).all(), f"{arch} produced non-finite values")
                self.assertEqual(y1.shape[0], 2)
                self.assertEqual(y1.shape[1], 8)
                self.assertTrue(np.allclose(y1, y2, atol=1e-5), f"{arch} eval forward is unstable")

    def test_alias_uses_cuda_when_available(self):
        if getattr(jt, "has_cuda", 0):
            self.assertEqual(jt.flags.use_cuda, 1)


if __name__ == "__main__":
    unittest.main()
