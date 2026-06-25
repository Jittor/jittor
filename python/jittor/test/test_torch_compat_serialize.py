"""Torch-grade serialization / state_dict round-trip tests for ``import jittor as torch``.

Part of the torch-grade test-suite rewrite (sibling of ``test_torch_compat_ops.py``).
Locks the *semantics* of the save/load and state_dict layer that ``jittor`` exposes
under the torch API: numeric values must survive a round-trip with zero error, module
parameters/buffers must be enumerable and reloadable, and numpy<->Var conversions must
be faithful. Runs on BOTH CPU and CUDA when the build has it.

Independent reference = the original numpy arrays (and a freshly-built module's forward
output), never jittor compared against itself.

Genuine jittor/torch divergences are ``@unittest.skip``-ped with a ``REASON:``:
  * ``torch.save``/``torch.load`` round-trip downcasts float64->float32 and
    int64->int32 (the portable-pickle reload goes through ``jt.array`` which narrows
    wide dtypes; the on-disk numpy data is the correct dtype, only the reload narrows).

Temp files are written under ``tempfile.mkdtemp()`` ($TMPDIR or /tmp) and cleaned up.

Run:  python -m jittor.test.test_torch_compat_serialize
      python -m pytest python/jittor/test/test_torch_compat_serialize.py
"""
import os
import shutil
import tempfile
import unittest
import numpy as np
import jittor as torch          # the whole point: jittor IS torch here
import jittor as jt
import jittor.nn as nn

# Exercise CPU always; add CUDA when the build has it. NPU(ACL) reports has_cuda too.
_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    """Run ``fn(device_name)`` once per available device under the right flag scope."""
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jt_serial_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def ac(self, got, ref, atol=1e-6, rtol=1e-6, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)

    def ae(self, got, ref, msg=""):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref), err_msg=msg)


# ---------------------------------------------------------------------------
# torch.save / torch.load
# ---------------------------------------------------------------------------
class TestSaveLoad(Base):
    def test_save_load_tensor_zero_error(self):
        x = np.random.RandomState(0).randn(3, 4, 5).astype("float32")
        def body(dev):
            p = self.path(f"t_{dev}.pkl")
            torch.save(torch.tensor(x), p)
            y = torch.load(p)
            self.ac(y.numpy(), x, atol=0, rtol=0, msg=f"tensor roundtrip {dev}")
        both_devices(body)

    def test_save_load_dict_of_tensors(self):
        rs = np.random.RandomState(1)
        a = rs.randn(2, 3).astype("float32")
        b = rs.randn(4).astype("float32")
        def body(dev):
            d = {"a": torch.tensor(a), "b": torch.tensor(b), "step": 7}
            p = self.path(f"d_{dev}.pkl")
            torch.save(d, p)
            e = torch.load(p)
            self.ac(e["a"].numpy(), a, atol=0, rtol=0, msg=f"dict a {dev}")
            self.ac(e["b"].numpy(), b, atol=0, rtol=0, msg=f"dict b {dev}")
            self.assertEqual(e["step"], 7, f"non-tensor value survives {dev}")
        both_devices(body)

    def test_save_load_nested_list(self):
        rs = np.random.RandomState(2)
        a = rs.randn(2, 2).astype("float32")
        b = rs.randn(3).astype("float32")
        def body(dev):
            obj = {"layers": [torch.tensor(a), torch.tensor(b)], "name": "net"}
            p = self.path(f"nested_{dev}.pkl")
            torch.save(obj, p)
            e = torch.load(p)
            self.ac(e["layers"][0].numpy(), a, atol=0, rtol=0, msg=f"nested0 {dev}")
            self.ac(e["layers"][1].numpy(), b, atol=0, rtol=0, msg=f"nested1 {dev}")
            self.assertEqual(e["name"], "net", f"nested name {dev}")
        both_devices(body)

    def test_save_load_module_state_dict_then_forward(self):
        # The real-world contract: persist a module, reload into a fresh one,
        # the reloaded module must produce identical output.
        rs = np.random.RandomState(3)
        x = rs.randn(2, 3, 8, 8).astype("float32")
        def body(dev):
            src = nn.Conv2d(3, 5, 3, padding=1)
            out_src = src(torch.tensor(x)).numpy()
            p = self.path(f"sd_{dev}.pkl")
            torch.save(src.state_dict(), p)
            sd = torch.load(p)
            dst = nn.Conv2d(3, 5, 3, padding=1)
            dst.load_state_dict(sd)
            out_dst = dst(torch.tensor(x)).numpy()
            self.ac(out_dst, out_src, atol=1e-5, msg=f"module roundtrip forward {dev}")
        both_devices(body)

    def test_save_load_dtypes_preserved(self):
        # Narrow dtypes survive the portable-pickle round-trip exactly.
        def body(dev):
            for dt in ["float32", "float16", "int32", "int16", "int8", "uint8", "bool"]:
                a = (np.arange(6) % 2).astype(dt) if dt == "bool" \
                    else np.arange(6).astype(dt)
                p = self.path(f"dt_{dt}_{dev}.pkl")
                torch.save(torch.from_numpy(a), p)
                y = torch.load(p)
                yn = y.numpy()
                self.assertEqual(yn.dtype.name, dt, f"dtype {dt} preserved {dev}")
                self.ae(yn, a, msg=f"dtype {dt} values {dev}")
        both_devices(body)

    def test_save_load_wide_dtypes_preserved(self):
        # fixed: _from_portable now reloads via from_numpy (preserves float64/int64)
        # instead of jt.array (which narrowed wide dtypes -> silent checkpoint downcast)
        def body(dev):
            for dt in ["float64", "int64"]:
                a = np.arange(6).astype(dt)
                p = self.path(f"wide_{dt}_{dev}.pkl")
                torch.save(torch.from_numpy(a), p)
                y = torch.load(p)
                self.assertEqual(y.numpy().dtype.name, dt,
                                 f"wide dtype {dt} preserved {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# module.state_dict / load_state_dict
# ---------------------------------------------------------------------------
class TestStateDict(Base):
    def test_state_dict_keys_and_values(self):
        def body(dev):
            m = nn.Conv2d(2, 3, 3)
            sd = m.state_dict()
            self.assertIn("weight", sd, f"weight key {dev}")
            self.assertIn("bias", sd, f"bias key {dev}")
            self.assertEqual(tuple(sd["weight"].shape), (3, 2, 3, 3),
                             f"weight shape {dev}")
            self.assertEqual(tuple(sd["bias"].shape), (3,), f"bias shape {dev}")
        both_devices(body)

    def test_state_dict_to_numpy(self):
        def body(dev):
            m = nn.Conv2d(2, 3, 3)
            sd = m.state_dict(to="numpy")
            self.assertIsInstance(sd["weight"], np.ndarray, f"numpy weight {dev}")
            # numpy view must equal the live Var
            live = dict(m.named_parameters())["weight"].numpy()
            self.ac(sd["weight"], live, atol=0, rtol=0, msg=f"state_dict numpy {dev}")
        both_devices(body)

    def test_load_state_dict_from_numpy_roundtrip(self):
        # state_dict(to="numpy") -> load_state_dict must restore the exact values.
        rs = np.random.RandomState(10)
        x = rs.randn(1, 2, 6, 6).astype("float32")
        def body(dev):
            src = nn.Conv2d(2, 4, 3)
            sd = src.state_dict(to="numpy")
            dst = nn.Conv2d(2, 4, 3)
            dst.load_state_dict(sd)
            for k, v in dst.named_parameters():
                self.ac(v.numpy(), sd[k], atol=0, rtol=0,
                        msg=f"param {k} restored {dev}")
            self.ac(dst(torch.tensor(x)).numpy(), src(torch.tensor(x)).numpy(),
                    atol=1e-5, msg=f"forward after load_state_dict {dev}")
        both_devices(body)

    def test_load_state_dict_modifies_target(self):
        # Loading must actually change the target's weights (not silently no-op).
        def body(dev):
            a = nn.Conv2d(2, 3, 3)
            b = nn.Conv2d(2, 3, 3)
            wa = dict(a.named_parameters())["weight"].numpy()
            wb = dict(b.named_parameters())["weight"].numpy()
            # random init makes them differ; after load they must match a.
            self.assertFalse(np.allclose(wa, wb), f"precondition differ {dev}")
            b.load_state_dict(a.state_dict())
            wb2 = dict(b.named_parameters())["weight"].numpy()
            self.ac(wb2, wa, atol=0, rtol=0, msg=f"load changed weights {dev}")
        both_devices(body)

    def test_state_dict_load_state_dict_multilayer(self):
        # A small Sequential-like stack: keys are dotted submodule paths.
        rs = np.random.RandomState(11)
        x = rs.randn(2, 3, 8, 8).astype("float32")

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
                self.bn = nn.BatchNorm2d(4)
                self.conv2 = nn.Conv2d(4, 2, 3, padding=1)

            def execute(self, x):
                return self.conv2(self.bn(self.conv1(x)))

        def body(dev):
            src = Net(); src.eval()
            sd = src.state_dict()
            keys = set(sd.keys())
            self.assertTrue(any(k.startswith("conv1.") for k in keys),
                            f"dotted conv1 key {dev}: {sorted(keys)}")
            self.assertTrue(any("bn" in k and "running_mean" in k for k in keys),
                            f"bn running_mean in state_dict {dev}")
            out_src = src(torch.tensor(x)).numpy()
            dst = Net(); dst.eval()
            dst.load_state_dict(sd)
            out_dst = dst(torch.tensor(x)).numpy()
            self.ac(out_dst, out_src, atol=1e-4, msg=f"multilayer roundtrip {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# named_parameters / parameters / named_buffers
# ---------------------------------------------------------------------------
class TestParamsBuffers(Base):
    def test_named_parameters_conv(self):
        def body(dev):
            m = nn.Conv2d(2, 3, 3)
            names = [k for k, _ in m.named_parameters()]
            self.assertIn("weight", names, f"weight param {dev}")
            self.assertIn("bias", names, f"bias param {dev}")
            for _, v in m.named_parameters():
                self.assertIsInstance(v, jt.Var, f"param is Var {dev}")
        both_devices(body)

    def test_parameters_count_matches_named(self):
        def body(dev):
            m = nn.Conv2d(3, 5, 3)
            self.assertEqual(len(list(m.parameters())),
                             len(list(m.named_parameters())),
                             f"parameters vs named_parameters count {dev}")
        both_devices(body)

    def test_batchnorm_buffers_vs_parameters(self):
        # running_mean/running_var/num_batches_tracked are BUFFERS, not parameters;
        # weight/bias are parameters. They must be disjoint.
        def body(dev):
            m = nn.BatchNorm2d(4)
            pnames = [k for k, _ in m.named_parameters()]
            bnames = [k for k, _ in m.named_buffers()]
            self.assertIn("weight", pnames, f"bn weight is param {dev}")
            self.assertIn("bias", pnames, f"bn bias is param {dev}")
            self.assertTrue(any("running_mean" in b for b in bnames),
                            f"running_mean is buffer {dev}: {bnames}")
            self.assertTrue(any("running_var" in b for b in bnames),
                            f"running_var is buffer {dev}: {bnames}")
            # buffers must NOT leak into parameters
            self.assertFalse(any("running_mean" in p for p in pnames),
                             f"running_mean not in params {dev}")
            self.assertFalse(any("running_var" in p for p in pnames),
                             f"running_var not in params {dev}")
        both_devices(body)

    def test_named_buffers_values_are_vars(self):
        def body(dev):
            m = nn.BatchNorm2d(3)
            for name, v in m.named_buffers():
                self.assertIsInstance(v, jt.Var, f"buffer {name} is Var {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# from_numpy / .numpy()
# ---------------------------------------------------------------------------
class TestFromNumpy(Base):
    def test_from_numpy_roundtrip_values(self):
        rs = np.random.RandomState(20)
        a = rs.randn(3, 4).astype("float32")
        def body(dev):
            v = torch.from_numpy(a)
            self.assertIsInstance(v, jt.Var, f"from_numpy -> Var {dev}")
            self.ac(v.numpy(), a, atol=0, rtol=0, msg=f"from_numpy roundtrip {dev}")
        both_devices(body)

    def test_from_numpy_dtype_preserved_common(self):
        def body(dev):
            for dt in ["float32", "float16", "int32", "int16", "int8", "uint8", "bool"]:
                a = (np.arange(6) % 2).astype(dt) if dt == "bool" \
                    else np.arange(6).astype(dt)
                v = torch.from_numpy(a)
                self.assertEqual(v.numpy().dtype.name, dt,
                                 f"from_numpy dtype {dt} {dev}")
        both_devices(body)

    def test_from_numpy_wide_dtypes(self):
        # Unlike plain jt.array (which narrows), the torch-compat from_numpy
        # PRESERVES float64 and int64 (matching torch). This is the counterpart
        # to the skipped save/load wide-dtype case.
        def body(dev):
            for dt in ["float64", "int64"]:
                a = np.arange(6).astype(dt)
                v = torch.from_numpy(a)
                self.assertEqual(v.numpy().dtype.name, dt,
                                 f"from_numpy wide dtype {dt} {dev}")
                self.ae(v.numpy(), a, msg=f"from_numpy wide values {dt} {dev}")
        both_devices(body)

    def test_numpy_preserves_shape_and_multidim(self):
        rs = np.random.RandomState(21)
        a = rs.randn(2, 3, 4, 5).astype("float32")
        def body(dev):
            v = torch.from_numpy(a)
            out = v.numpy()
            self.assertEqual(out.shape, a.shape, f"numpy shape {dev}")
            self.ac(out, a, atol=0, rtol=0, msg=f"numpy 4d values {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
