# ***************************************************************
# diffusers generation regression (#10): run the core Stable-Diffusion
# building blocks (UNet2DModel, AutoencoderKL, a DDIM denoising loop) through
# `import torch` -> jittor. These were validated to match real torch ~1e-6 (forward
# 1.1e-6, backward 1.45e-6, denoising loop 3.1e-5, VAE 1.43e-6); this guards against
# regressions of the diffusers GENERATION path (loading real checkpoints via
# from_pretrained is a separate, tracked limitation -- see ALL_TODO.md).
#
#   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
#   /home/yizhang/miniconda3/envs/jt-torch/bin/python -m jittor.test.test_diffusers
# Skips cleanly if torch_shim/diffusers are unavailable.
# ***************************************************************
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import unittest, numpy as np

try:
    import torch  # torch_shim -> jittor
    import jittor as jt
    from diffusers import UNet2DModel, AutoencoderKL, DDIMScheduler
    _HAS = (getattr(torch, '__name__', '') == 'torch') and hasattr(torch, 'tensor')
except Exception:
    _HAS = False


def _unet():
    return UNet2DModel(
        sample_size=16, in_channels=3, out_channels=3, block_out_channels=(16, 32),
        layers_per_block=1, norm_num_groups=4,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"))


@unittest.skipUnless(_HAS, "needs torch_shim + diffusers")
class TestDiffusers(unittest.TestCase):
    def test_unet_forward_eval_determinism(self):
        m = _unet(); m.eval()
        x = torch.tensor(np.random.RandomState(0).randn(2, 3, 16, 16).astype('float32'))
        t = torch.tensor(np.array([3, 7]).astype('int64'))
        with torch.no_grad():
            a = m(x, t).sample.float().numpy()
            b = m(x, t).sample.float().numpy()
        self.assertEqual(a.shape, (2, 3, 16, 16))
        self.assertTrue(np.isfinite(a).all(), "UNet output non-finite")
        self.assertTrue(np.allclose(a, b, atol=1e-5), "UNet eval non-deterministic")

    def test_unet_backward_grads(self):
        m = _unet(); m.eval()
        x = torch.tensor(np.random.RandomState(1).randn(1, 3, 16, 16).astype('float32'))
        t = torch.tensor(np.array([5]).astype('int64'))
        named = list(m.named_parameters())
        loss = m(x, t).sample.float().pow(2).sum()
        grads = jt.grad(loss, [p for _, p in named], retain_graph=True)
        none = [n for (n, _), g in zip(named, grads) if g is None]
        self.assertEqual(none, [], f"{len(none)} UNet params have no grad")

    def test_vae_encode_decode(self):
        m = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
                          block_out_channels=(16,), layers_per_block=1, norm_num_groups=4,
                          down_block_types=("DownEncoderBlock2D",),
                          up_block_types=("UpDecoderBlock2D",))
        m.eval()
        x = torch.tensor(np.random.RandomState(2).randn(1, 3, 16, 16).astype('float32'))
        with torch.no_grad():
            lat = m.encode(x).latent_dist.mean
            rec = m.decode(lat).sample
        self.assertEqual(tuple(rec.shape), (1, 3, 16, 16))
        self.assertTrue(np.isfinite(lat.float().numpy()).all() and np.isfinite(rec.float().numpy()).all(),
                        "VAE output non-finite")

    def test_ddim_denoising_loop(self):
        m = _unet(); m.eval()
        sched = DDIMScheduler(num_train_timesteps=1000); sched.set_timesteps(3)
        s = torch.tensor(np.random.RandomState(3).randn(1, 3, 16, 16).astype('float32'))
        for t in sched.timesteps:
            with torch.no_grad():
                noise = m(s, t).sample
            s = sched.step(noise, t, s).prev_sample
        out = s.float().numpy()
        self.assertEqual(out.shape, (1, 3, 16, 16))
        self.assertTrue(np.isfinite(out).all(), "denoised sample non-finite")


if __name__ == '__main__':
    unittest.main()
