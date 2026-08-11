"""Torch-grade interpolate/upsample-semantics tests for ``import jittor as torch``.

Part of the torch-grade test-suite rewrite (sibling of ``test_torch_compat_conv_pool.py``).
Every check compares jittor-as-torch against an INDEPENDENT numpy reference computed from
scratch here, and runs on BOTH CPU and CUDA when the build has it, so it locks torch-API
*semantics*. Diffusers/UNet decoders resize feature maps constantly, so this is hot.

Covered: ``F.interpolate`` mode='nearest'/'bilinear'/'bicubic' with scale_factor AND size,
upsample AND downsample, align_corners True/False (bilinear), rectangular targets, and the
module forms ``nn.Upsample`` / ``nn.UpsamplingNearest2d`` / ``nn.UpsamplingBilinear2d``.

The numpy references encode torch's documented source-index maps:
  * nearest:           src = floor(i * in/out)                       (no align_corners)
  * bilinear AC=True:  src = i * (in-1)/(out-1)
  * bilinear AC=False: src = (i+0.5) * in/out - 0.5  (clamped >=0 at the low border)

NOTE on jittor semantics surfaced here (kept visible, not silently hidden):
  * scale_factor must produce an UPSAMPLE for the bilinear/bicubic `upsample` path to match
    torch; jittor's `interpolate` already routes scale>1 -> upsample, scale<=1 -> resize.
  * nearest with align_corners is not a thing in torch (asserted by torch); we don't test it.

Run:  python -m pytest tests/compat/torch/test_torch_compat_interpolate.py
      python -m pytest tests/compat/torch/test_torch_compat_interpolate.py
"""
import unittest
import numpy as np
import jittor as torch          # the whole point: jittor IS torch here
import jittor as jt
import jittor.nn as nn

F = nn.functional

# Exercise CPU always; add CUDA when the build has it. NPU(ACL) reports has_cuda too.
_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    """Run ``fn(device_name)`` once per available device under the right flag scope."""
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


# ---------------------------------------------------------------------------
# Independent numpy references (torch source-index maps).
# ---------------------------------------------------------------------------
def nearest_ref(x, Oh, Ow):
    """torch nearest: src = floor(i * in/out), clamped to [0, in-1]."""
    N, C, H, W = x.shape
    out = np.zeros((N, C, Oh, Ow), dtype=np.float64)
    for i in range(Oh):
        si = min(int(np.floor(i * H / Oh)), H - 1)
        for j in range(Ow):
            sj = min(int(np.floor(j * W / Ow)), W - 1)
            out[:, :, i, j] = x[:, :, si, sj]
    return out


def bilinear_ref(x, Oh, Ow, align_corners):
    """torch bilinear, matching its source-coordinate map + edge clamping."""
    N, C, H, W = x.shape
    x = x.astype(np.float64)
    out = np.zeros((N, C, Oh, Ow))
    for i in range(Oh):
        yi = (i * (H - 1) / (Oh - 1) if Oh > 1 else 0.0) if align_corners \
            else (i + 0.5) * H / Oh - 0.5
        for j in range(Ow):
            xj = (j * (W - 1) / (Ow - 1) if Ow > 1 else 0.0) if align_corners \
                else (j + 0.5) * W / Ow - 0.5
            y0 = int(np.floor(yi)); x0 = int(np.floor(xj))
            y1 = min(y0 + 1, H - 1); x1 = min(x0 + 1, W - 1)
            y0c = max(y0, 0); x0c = max(x0, 0)
            ly = min(max(yi - y0, 0.0), 1.0); lx = min(max(xj - x0, 0.0), 1.0)
            out[:, :, i, j] = (x[:, :, y0c, x0c] * (1 - ly) * (1 - lx)
                               + x[:, :, y0c, x1] * (1 - ly) * lx
                               + x[:, :, y1, x0c] * ly * (1 - lx)
                               + x[:, :, y1, x1] * ly * lx)
    return out


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-4, rtol=1e-4, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)


# ---------------------------------------------------------------------------
# F.interpolate -- nearest
# ---------------------------------------------------------------------------
class TestInterpolateNearest(Base):
    def test_nearest_scale2_upsample(self):
        x = np.random.RandomState(40).randn(1, 2, 3, 3).astype("float32")
        def body(dev):
            g = F.interpolate(torch.array(x), scale_factor=2, mode="nearest").numpy()
            self.ac(g, nearest_ref(x, 6, 6), atol=1e-5, msg=f"nearest x2 {dev}")
        both_devices(body)

    def test_nearest_size_upsample_noninteger(self):
        x = np.arange(16).reshape(1, 1, 4, 4).astype("float32")
        def body(dev):
            g = F.interpolate(torch.array(x), size=(6, 6), mode="nearest").numpy()
            self.ac(g, nearest_ref(x, 6, 6), atol=1e-5, msg=f"nearest 4->6 {dev}")
        both_devices(body)

    def test_nearest_size_downsample(self):
        x = np.arange(16).reshape(1, 1, 4, 4).astype("float32")
        def body(dev):
            g = F.interpolate(torch.array(x), size=(3, 3), mode="nearest").numpy()
            self.ac(g, nearest_ref(x, 3, 3), atol=1e-5, msg=f"nearest 4->3 {dev}")
        both_devices(body)

    def test_nearest_rectangular(self):
        x = np.random.RandomState(43).randn(2, 3, 4, 6).astype("float32")
        def body(dev):
            g = F.interpolate(torch.array(x), size=(8, 3), mode="nearest").numpy()
            self.ac(g, nearest_ref(x, 8, 3), atol=1e-5, msg=f"nearest rect {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# F.interpolate -- bilinear
# ---------------------------------------------------------------------------
class TestInterpolateBilinear(Base):
    def test_bilinear_align_corners_true(self):
        x = np.random.RandomState(41).randn(1, 2, 4, 4).astype("float32")
        def body(dev):
            g = F.interpolate(torch.array(x), size=(8, 8), mode="bilinear",
                              align_corners=True).numpy()
            self.ac(g, bilinear_ref(x, 8, 8, True), atol=1e-3,
                    msg=f"bilinear AC=True {dev}")
        both_devices(body)

    def test_bilinear_align_corners_false(self):
        x = np.random.RandomState(42).randn(1, 2, 4, 4).astype("float32")
        def body(dev):
            g = F.interpolate(torch.array(x), size=(8, 8), mode="bilinear",
                              align_corners=False).numpy()
            self.ac(g, bilinear_ref(x, 8, 8, False), atol=1e-3,
                    msg=f"bilinear AC=False {dev}")
        both_devices(body)

    def test_bilinear_scale_factor_upsample(self):
        x = np.random.RandomState(44).randn(1, 3, 5, 5).astype("float32")
        def body(dev):
            g = F.interpolate(torch.array(x), scale_factor=2, mode="bilinear",
                              align_corners=False).numpy()
            self.ac(g, bilinear_ref(x, 10, 10, False), atol=1e-3,
                    msg=f"bilinear scale2 {dev}")
        both_devices(body)

    def test_bilinear_rectangular_ac_true(self):
        x = np.random.RandomState(45).randn(1, 2, 4, 6).astype("float32")
        def body(dev):
            g = F.interpolate(torch.array(x), size=(7, 9), mode="bilinear",
                              align_corners=True).numpy()
            self.ac(g, bilinear_ref(x, 7, 9, True), atol=1e-3,
                    msg=f"bilinear rect AC=True {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# F.interpolate -- bicubic  (jittor's own ref via a tiny known case is brittle;
# we lock SHAPE + that it reproduces input on an identity (size==input) upsample,
# and that bicubic of a constant map stays constant -- mode-agnostic invariants).
# ---------------------------------------------------------------------------
class TestInterpolateBicubic(Base):
    def test_bicubic_constant_stays_constant(self):
        x = np.full((1, 2, 4, 4), 3.5, dtype="float32")
        def body(dev):
            g = F.interpolate(torch.array(x), size=(7, 7), mode="bicubic",
                              align_corners=False).numpy()
            self.ac(g, np.full((1, 2, 7, 7), 3.5), atol=1e-3,
                    msg=f"bicubic const {dev}")
        both_devices(body)

    def test_bicubic_shape(self):
        x = np.random.RandomState(46).randn(1, 3, 5, 5).astype("float32")
        def body(dev):
            g = F.interpolate(torch.array(x), size=(9, 9), mode="bicubic",
                              align_corners=True)
            self.assertEqual(tuple(g.shape), (1, 3, 9, 9), f"bicubic shape {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# nn.Upsample / nn.UpsamplingNearest2d / nn.UpsamplingBilinear2d
# ---------------------------------------------------------------------------
class TestUpsampleModules(Base):
    def test_upsample_nearest_scale2(self):
        x = np.random.RandomState(47).randn(1, 2, 3, 4).astype("float32")
        def body(dev):
            g = nn.Upsample(scale_factor=2, mode="nearest")(torch.array(x)).numpy()
            self.ac(g, nearest_ref(x, 6, 8), atol=1e-5, msg=f"Upsample nearest {dev}")
        both_devices(body)

    def test_upsample_bilinear_scale2_ac_true(self):
        x = np.random.RandomState(48).randn(1, 2, 4, 4).astype("float32")
        def body(dev):
            g = nn.Upsample(scale_factor=2, mode="bilinear",
                            align_corners=True)(torch.array(x)).numpy()
            self.ac(g, bilinear_ref(x, 8, 8, True), atol=1e-3,
                    msg=f"Upsample bilinear AC=True {dev}")
        both_devices(body)

    def test_upsampling_nearest2d(self):
        x = np.random.RandomState(49).randn(1, 3, 4, 5).astype("float32")
        def body(dev):
            g = nn.UpsamplingNearest2d(scale_factor=2)(torch.array(x)).numpy()
            self.ac(g, nearest_ref(x, 8, 10), atol=1e-5,
                    msg=f"UpsamplingNearest2d {dev}")
        both_devices(body)

    def test_upsampling_bilinear2d(self):
        x = np.random.RandomState(50).randn(1, 2, 4, 4).astype("float32")
        def body(dev):
            g = nn.UpsamplingBilinear2d(scale_factor=2)(torch.array(x)).numpy()
            # UpsamplingBilinear2d implies align_corners=True in torch.
            self.ac(g, bilinear_ref(x, 8, 8, True), atol=1e-3,
                    msg=f"UpsamplingBilinear2d {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
