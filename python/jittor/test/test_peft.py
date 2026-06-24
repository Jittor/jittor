# ***************************************************************
# PEFT / LoRA regression (LlamaFactory's core fine-tuning path): exercise
# get_peft_model wrapping, correct LoRA grad semantics, a well-posed training
# fit, and adapter save_pretrained -> PeftModel.from_pretrained roundtrip, all
# through `import torch` -> jittor. Verified end-to-end on jittor (Ascend + CUDA):
#   - LoRA wraps and freezes the base (only lora_A/lora_B trainable),
#   - standard LoRA init (B=0) -> lora_A grad is 0 on step 1, lora_B grad nonzero,
#   - a reachable Linear+LoRA regression converges to ~0 loss,
#   - adapter save/load reloads exact adapter weights (output diff 0.0).
#
#   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
#   /home/yizhang/miniconda3/envs/jt-torch/bin/python -m jittor.test.test_peft
# Skips cleanly if torch_shim / peft are unavailable.
# ***************************************************************
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import unittest, tempfile, numpy as np

try:
    import torch  # torch_shim -> jittor
    import jittor as jt
    import jittor.nn as nn
    from peft import LoraConfig, get_peft_model, PeftModel
    _HAS = (getattr(torch, '__name__', '') == 'torch') and hasattr(torch, 'tensor')
except Exception:
    _HAS = False


class _Tiny(nn.Module if _HAS else object):
    def __init__(self):
        super().__init__(); self.proj = nn.Linear(8, 8)
    def execute(self, x):
        return self.proj(x)


def _lora(model, r=4):
    return get_peft_model(model, LoraConfig(r=r, lora_alpha=8, target_modules=["proj"], lora_dropout=0.0))


@unittest.skipUnless(_HAS, "needs torch_shim + peft")
class TestPeftLora(unittest.TestCase):
    def test_wrap_freezes_base_and_grad_semantics(self):
        pm = _lora(_Tiny())
        trainable = [n for n, p in pm.named_parameters() if getattr(p, 'requires_grad', True)]
        self.assertTrue(trainable and all('lora' in n.lower() for n in trainable),
                        "only LoRA params should be trainable")
        A = next(p for n, p in pm.named_parameters() if 'lora_A' in n)
        B = next(p for n, p in pm.named_parameters() if 'lora_B' in n)
        X = jt.array(np.random.RandomState(0).randn(4, 8).astype('float32'))
        loss = (pm(X) ** 2).mean()
        gA = jt.grad(loss, [A])[0]; gB = jt.grad(loss, [B])[0]
        # standard LoRA init (B=0): lora_A grad 0 on step 1, lora_B grad nonzero
        self.assertEqual(float(jt.abs(gA).sum().item()), 0.0, "lora_A grad should be 0 at init (B=0)")
        self.assertGreater(float(jt.abs(gB).sum().item()), 0.0, "lora_B grad should be nonzero")

    def test_lora_fit_converges(self):
        # reachable target: base output + a linear delta (LoRA r=8 can fit any 8x8 delta)
        rs = np.random.RandomState(0)
        pm = _lora(_Tiny(), r=8)
        X = jt.array(rs.randn(16, 8).astype('float32'))
        base = pm(X).numpy().copy()
        Wd = (rs.randn(8, 8) * 0.3).astype('float32')
        tgt = jt.array((X.numpy() @ Wd.T + base).astype('float32'))
        ps = [p for n, p in pm.named_parameters() if getattr(p, 'requires_grad', True)]
        opt = torch.optim.Adam(ps, lr=1e-2)
        first = last = None
        for i in range(200):
            loss = ((pm(X) - tgt) ** 2).mean()
            opt.zero_grad(); opt.step(loss)
            if first is None: first = float(loss.item())
            last = float(loss.item())
        self.assertLess(last, first * 1e-2, f"LoRA fit did not converge ({first:.4g} -> {last:.4g})")

    def test_adapter_save_load_roundtrip(self):
        rs = np.random.RandomState(1)
        core = _Tiny()
        W0 = core.proj.weight.numpy().copy(); b0 = core.proj.bias.numpy().copy()
        pm = _lora(core)
        X = jt.array(rs.randn(4, 8).astype('float32'))
        ps = [p for n, p in pm.named_parameters() if getattr(p, 'requires_grad', True)]
        opt = torch.optim.Adam(ps, lr=1e-2); tgt = jt.array(rs.randn(4, 8).astype('float32'))
        for i in range(20):
            opt.zero_grad(); opt.step(((pm(X) - tgt) ** 2).mean())
        ref = pm(X).numpy().copy()
        d = tempfile.mkdtemp(); pm.save_pretrained(d)
        self.assertIn("adapter_model.safetensors", os.listdir(d))
        fresh = _Tiny(); fresh.proj.weight.update(jt.array(W0)); fresh.proj.bias.update(jt.array(b0))
        pm2 = PeftModel.from_pretrained(fresh, d)
        self.assertLess(float(np.abs(pm2(X).numpy() - ref).max()), 1e-5,
                        "reloaded adapter produced different output")


if __name__ == '__main__':
    unittest.main()
