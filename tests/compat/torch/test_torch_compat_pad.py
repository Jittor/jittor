"""Torch-grade padding-semantics tests for ``import jittor as torch``.

Part of the torch-grade test-suite rewrite (sibling of ``test_torch_compat_conv_pool.py``).
Every check compares jittor-as-torch against an INDEPENDENT numpy reference (np.pad with the
matching mode) and runs on BOTH CPU and CUDA when the build has it, so it locks torch-API
*semantics*. Diffusers/UNet/StyleGAN lean on reflect/replicate/circular pad heavily.

Covered: ``F.pad`` mode='constant'/'reflect'/'replicate'/'circular' over the LAST 1/2/3
dims, asymmetric and zero-on-one-side pads, non-zero constant value, the default
(zero constant), the ``pad=`` keyword alias and bool fill values. Module forms include
``nn.ConstantPad1d/2d/3d``, ``nn.ZeroPad2d``, ``nn.ReflectionPad2d`` and
``nn.ReplicationPad2d``; 2D forms cover both int and 4-tuple padding.

torch's pad tuple is REVERSED and trailing-dim-first: ``F.pad(x, (l, r, t, b))`` pads the
last dim by (l, r) and the 2nd-to-last by (t, b). The np reference encodes exactly that.

REGRESSION: nn.ReflectionPad2d / nn.ReplicationPad2d crashed with a TUPLE padding
(``if padding < 0`` ran ``tuple < int`` -> TypeError) before reaching the type dispatch;
the standard torch call ``nn.ReplicationPad2d((l,r,t,b))`` was unusable. The regression
remains covered after the implementation moved behind the ``jittor.nn`` facade.

Run:  python -m pytest tests/compat/torch/test_torch_compat_pad.py
      python -m pytest tests/compat/torch/test_torch_compat_pad.py
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


_NP_MODE = {"constant": "constant", "replicate": "edge",
            "reflect": "reflect", "circular": "wrap"}


def pad_ref(x, pad, mode="constant", value=0.0):
    """torch F.pad reference: `pad` is reversed & applies to the trailing dims.

    pad=(l,r) pads last dim; (l,r,t,b) pads last two; (l,r,t,b,f,k) pads last three.
    """
    npairs = len(pad) // 2
    # build np.pad width list: leading dims get (0,0); trailing dims get reversed pairs.
    widths = [(0, 0)] * (x.ndim - npairs)
    for i in range(npairs):
        # pad[0:2] is the LAST dim, pad[2:4] the previous, ... -> reverse pair order.
        lo = pad[2 * (npairs - 1 - i)]
        hi = pad[2 * (npairs - 1 - i) + 1]
        widths.append((lo, hi))
    if mode == "constant":
        return np.pad(x, widths, mode="constant", constant_values=value)
    return np.pad(x, widths, mode=_NP_MODE[mode])


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)


# ---------------------------------------------------------------------------
# F.pad -- constant
# ---------------------------------------------------------------------------
class TestPadConstant(Base):
    def setUp(self):
        self.x = np.random.RandomState(50).randn(1, 1, 4, 5).astype("float32")

    def test_constant_asymmetric(self):
        x = self.x; pad = (1, 2, 3, 1)
        def body(dev):
            self.ac(F.pad(torch.array(x), pad, mode="constant", value=7.0).numpy(),
                    pad_ref(x, pad, "constant", 7.0), msg=f"pad const asym {dev}")
        both_devices(body)

    def test_default_is_zero_constant(self):
        x = self.x; pad = (1, 2, 3, 1)
        def body(dev):
            self.ac(F.pad(torch.array(x), pad).numpy(),
                    pad_ref(x, pad, "constant", 0.0), msg=f"pad default {dev}")
        both_devices(body)

    def test_constant_last_dim_only(self):
        x = self.x; pad = (2, 3)              # pads only the last dim
        def body(dev):
            self.ac(F.pad(torch.array(x), pad, value=-1.0).numpy(),
                    pad_ref(x, pad, "constant", -1.0), msg=f"pad const 1d {dev}")
        both_devices(body)

    def test_constant_zero_on_one_side(self):
        x = self.x; pad = (0, 3, 2, 0)        # right & top only
        def body(dev):
            self.ac(F.pad(torch.array(x), pad, value=5.0).numpy(),
                    pad_ref(x, pad, "constant", 5.0), msg=f"pad const one-side {dev}")
        both_devices(body)

    def test_constant_three_dims(self):
        x = np.random.RandomState(51).randn(2, 3, 4, 5).astype("float32")
        pad = (1, 1, 2, 0, 0, 1)              # last three dims
        def body(dev):
            self.ac(F.pad(torch.array(x), pad, value=2.5).numpy(),
                    pad_ref(x, pad, "constant", 2.5), msg=f"pad const 3d {dev}")
        both_devices(body)

    def test_pad_keyword_alias(self):
        x = self.x; amounts = (1, 2)
        def body(dev):
            self.ac(F.pad(torch.array(x), pad=amounts, value=-3.0).numpy(),
                    pad_ref(x, amounts, "constant", -3.0), msg=f"pad keyword {dev}")
        both_devices(body)

    def test_bool_constant_fill(self):
        x = np.array([[True, False], [False, True]], dtype="bool")
        amounts = (1, 2, 2, 1)
        def body(dev):
            got = F.pad(torch.array(x), amounts, value=True).numpy()
            self.assertEqual(np.asarray(got).dtype, np.dtype("bool"))
            self.ac(got, pad_ref(x, amounts, "constant", True),
                    msg=f"pad bool fill {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# F.pad -- replicate / reflect / circular
# ---------------------------------------------------------------------------
class TestPadModes(Base):
    def setUp(self):
        self.x = np.random.RandomState(52).randn(1, 2, 4, 5).astype("float32")

    def test_replicate_asymmetric(self):
        x = self.x; pad = (1, 2, 3, 1)
        def body(dev):
            self.ac(F.pad(torch.array(x), pad, mode="replicate").numpy(),
                    pad_ref(x, pad, "replicate"), msg=f"pad replicate {dev}")
        both_devices(body)

    def test_replicate_last_dim(self):
        x = self.x; pad = (2, 3)
        def body(dev):
            self.ac(F.pad(torch.array(x), pad, mode="replicate").numpy(),
                    pad_ref(x, pad, "replicate"), msg=f"pad replicate 1d {dev}")
        both_devices(body)

    def test_reflect_symmetric(self):
        x = self.x; pad = (1, 1, 1, 1)        # reflect needs pad < dim
        def body(dev):
            self.ac(F.pad(torch.array(x), pad, mode="reflect").numpy(),
                    pad_ref(x, pad, "reflect"), msg=f"pad reflect {dev}")
        both_devices(body)

    def test_reflect_asymmetric(self):
        x = self.x; pad = (2, 1, 3, 2)        # all < dim (W=5,H=4)
        def body(dev):
            self.ac(F.pad(torch.array(x), pad, mode="reflect").numpy(),
                    pad_ref(x, pad, "reflect"), msg=f"pad reflect asym {dev}")
        both_devices(body)

    def test_circular_symmetric(self):
        x = self.x; pad = (1, 1, 1, 1)
        def body(dev):
            self.ac(F.pad(torch.array(x), pad, mode="circular").numpy(),
                    pad_ref(x, pad, "circular"), msg=f"pad circular {dev}")
        both_devices(body)

    def test_circular_asymmetric(self):
        x = self.x; pad = (2, 1, 1, 2)
        def body(dev):
            self.ac(F.pad(torch.array(x), pad, mode="circular").numpy(),
                    pad_ref(x, pad, "circular"), msg=f"pad circular asym {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# nn.ZeroPad2d / nn.ConstantPad2d
# ---------------------------------------------------------------------------
class TestPadModulesZeroConstant(Base):
    def setUp(self):
        self.x = np.random.RandomState(53).randn(2, 3, 4, 5).astype("float32")

    def test_zeropad2d_int(self):
        x = self.x
        def body(dev):
            self.ac(nn.ZeroPad2d(2)(torch.array(x)).numpy(),
                    pad_ref(x, (2, 2, 2, 2), "constant", 0.0), msg=f"ZeroPad2d int {dev}")
        both_devices(body)

    def test_zeropad2d_tuple(self):
        x = self.x; pad = (1, 2, 3, 0)        # torch order (l,r,t,b)
        def body(dev):
            self.ac(nn.ZeroPad2d(pad)(torch.array(x)).numpy(),
                    pad_ref(x, pad, "constant", 0.0), msg=f"ZeroPad2d tuple {dev}")
        both_devices(body)

    def test_constantpad2d_tuple(self):
        x = self.x; pad = (2, 1, 0, 3)
        def body(dev):
            self.ac(nn.ConstantPad2d(pad, 4.0)(torch.array(x)).numpy(),
                    pad_ref(x, pad, "constant", 4.0), msg=f"ConstantPad2d {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# nn.ConstantPad1d / nn.ConstantPad3d
# ---------------------------------------------------------------------------
class TestConstantPadModulesExtraDimensions(Base):
    def test_constantpad1d_tuple(self):
        x = np.random.RandomState(55).randn(2, 3, 4).astype("float32")
        pad = (1, 2)
        def body(dev):
            self.ac(nn.ConstantPad1d(pad, -2.5)(torch.array(x)).numpy(),
                    pad_ref(x, pad, "constant", -2.5), msg=f"ConstantPad1d {dev}")
        both_devices(body)

    def test_constantpad3d_tuple(self):
        x = np.random.RandomState(56).randn(1, 2, 3, 4, 5).astype("float32")
        pad = (1, 0, 2, 1, 0, 2)
        def body(dev):
            self.ac(nn.ConstantPad3d(pad, 1.25)(torch.array(x)).numpy(),
                    pad_ref(x, pad, "constant", 1.25), msg=f"ConstantPad3d {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# nn.ReflectionPad2d / nn.ReplicationPad2d   (int + 4-tuple)
# REGRESSION: tuple padding used to crash in __init__ (tuple < int). Fixed.
# ---------------------------------------------------------------------------
class TestReflectionReplicationModules(Base):
    def setUp(self):
        self.x = np.random.RandomState(54).randn(1, 2, 4, 5).astype("float32")

    def test_reflectionpad2d_int(self):
        x = self.x
        def body(dev):
            self.ac(nn.ReflectionPad2d(1)(torch.array(x)).numpy(),
                    pad_ref(x, (1, 1, 1, 1), "reflect"), msg=f"ReflectionPad2d int {dev}")
        both_devices(body)

    def test_reflectionpad2d_tuple(self):
        x = self.x; pad = (2, 1, 3, 2)
        def body(dev):
            self.ac(nn.ReflectionPad2d(pad)(torch.array(x)).numpy(),
                    pad_ref(x, pad, "reflect"), msg=f"ReflectionPad2d tuple {dev}")
        both_devices(body)

    def test_replicationpad2d_int(self):
        x = self.x
        def body(dev):
            self.ac(nn.ReplicationPad2d(2)(torch.array(x)).numpy(),
                    pad_ref(x, (2, 2, 2, 2), "replicate"), msg=f"ReplicationPad2d int {dev}")
        both_devices(body)

    def test_replicationpad2d_tuple(self):
        x = self.x; pad = (1, 2, 0, 1)
        def body(dev):
            self.ac(nn.ReplicationPad2d(pad)(torch.array(x)).numpy(),
                    pad_ref(x, pad, "replicate"), msg=f"ReplicationPad2d tuple {dev}")
        both_devices(body)

    def test_replicationpad2d_matches_torch_doc_example(self):
        # torch docs: nn.ReplicationPad2d((1,1,2,0)) on arange(9).reshape(1,1,3,3).
        x = np.arange(9).reshape(1, 1, 3, 3).astype("float32")
        pad = (1, 1, 2, 0)
        expected = np.array([[0, 0, 1, 2, 2],
                             [0, 0, 1, 2, 2],
                             [0, 0, 1, 2, 2],
                             [3, 3, 4, 5, 5],
                             [6, 6, 7, 8, 8]], dtype="float32")
        def body(dev):
            g = nn.ReplicationPad2d(pad)(torch.array(x)).numpy().reshape(5, 5)
            self.ac(g, expected, msg=f"ReplicationPad2d doc example {dev}")
            # cross-check the np reference agrees with torch's documented output
            self.ac(pad_ref(x, pad, "replicate").reshape(5, 5), expected,
                    msg=f"pad_ref replicate doc {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
