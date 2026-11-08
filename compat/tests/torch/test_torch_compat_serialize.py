"""Torch-grade serialization / state_dict round-trip tests for ``import torch``.

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

Run:  python -m pytest compat/tests/torch/test_torch_compat_serialize.py
      python -m pytest compat/tests/torch/test_torch_compat_serialize.py
"""

from _helpers import capability as _test_capability
import json
import os
import struct
import shutil
import tempfile
import unittest
import numpy as np
import torch
import jittor as jt
import torch.nn as nn

# The legacy cuda sweep label also exercises the registered ACL/ROCm backend.
_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if _test_capability.any_accelerator_enabled(backend=jt) else [])


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
class TestSafetensorsHardening(Base):
    """Aliased tensors are refused and corrupt files fail at open, as safetensors does."""

    def setUp(self):
        super().setUp()
        try:
            import safetensors
            import safetensors.torch as st
        except ImportError:
            self.skipTest("safetensors is not installed")
        self.st, self.error = st, safetensors.SafetensorError

    def _write(self, name, header, payload):
        path = self.path(name)
        with open(path, "wb") as handle:
            handle.write(self._raw(header, payload))
        return path

    def test_shared_storage_is_refused_and_distinct_tensors_round_trip(self):
        def body(dev):
            weight = torch.randn(3, 4, device=dev)
            layer = torch.nn.Linear(4, 4).to(dev)
            for name, tensors in (("same", {"a": weight, "b": weight}),
                                  ("tied", {"embed": layer.weight, "head": layer.weight})):
                path = self.path(f"{name}_{dev}.safetensors")
                with self.assertRaisesRegex(RuntimeError, "share memory"):
                    self.st.save_file(tensors, path)
                self.assertFalse(os.path.exists(path), f"{name} wrote a file {dev}")
                with self.assertRaisesRegex(RuntimeError, "share memory"):
                    self.st.save(tensors)
            path = self.path(f"distinct_{dev}.safetensors")
            self.st.save_file({"a": weight, "b": weight.clone()}, path)
            loaded = self.st.load_file(path)
            self.ae(loaded["a"].numpy(), weight.numpy(), f"distinct a {dev}")
            self.ae(loaded["b"].numpy(), weight.numpy(), f"distinct b {dev}")
        both_devices(body)

    def test_corrupt_headers_raise_safetensor_error_at_open(self):
        from safetensors import safe_open
        payload = np.arange(6, dtype=np.float32).tobytes()
        header = {"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}
        raw_header = json.dumps(header).encode("utf-8")
        cases = {
            "header too small": b"\x01\x02",
            "header too large": struct.pack("<Q", 10 ** 9) + raw_header + payload,
            "invalid JSON": struct.pack("<Q", 5) + b"{abc}" + payload,
            "expected an object": struct.pack("<Q", 2) + b"[]" + payload,
            "invalid UTF-8": struct.pack("<Q", 2) + b"\xff\xfe" + payload,
            "not fully covered": struct.pack("<Q", len(raw_header)) + raw_header + payload[:-2],
            "invalid tensor info": self._raw({"w": {"dtype": "F32", "shape": [2, 3],
                                                    "data_offsets": [24, 0]}}, payload),
        }
        for message, data in cases.items():
            with self.subTest(message=message):
                path = self.path("corrupt.safetensors")
                with open(path, "wb") as handle:
                    handle.write(data)
                with self.assertRaises(self.error):
                    safe_open(path, framework="pt", device="cpu")
                with self.assertRaises(self.error):
                    self.st.load(data)
        path = self._write("ok.safetensors", header, payload)
        with safe_open(path, framework="pt", device="cpu") as reader:
            self.ae(reader.get_tensor("w").numpy(), np.arange(6, dtype=np.float32).reshape(2, 3))

    def test_payload_truncated_after_open_fails_explicitly(self):
        from safetensors import safe_open
        payload = np.arange(6, dtype=np.float32).tobytes()
        path = self._write("shrinks.safetensors", {
            "w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}, payload)
        with safe_open(path, framework="pt", device="cpu") as reader:
            with open(path, "rb+") as handle:
                handle.seek(-2, os.SEEK_END)
                handle.truncate()
            with self.assertRaisesRegex(EOFError, "short safetensors payload.*'w'"):
                reader.get_tensor("w")

    @staticmethod
    def _raw(header, payload):
        raw = json.dumps(header).encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw + payload


class TestSaveLoad(Base):
    def _write_bfloat16_safetensor(self, name="weight"):
        values = np.asarray(
            [1.0, -2.5, 3.125, 0.0078125, 128.0, -0.25], dtype=np.float32)
        payload = (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()
        header = json.dumps({
            name: {
                "dtype": "BF16",
                "shape": [2, 3],
                "data_offsets": [0, len(payload)],
            }
        }, separators=(",", ":")).encode("utf-8")
        header += b" " * (-len(header) % 8)
        path = self.path("bf16.safetensors")
        with open(path, "wb") as handle:
            handle.write(struct.pack("<Q", len(header)))
            handle.write(header)
            handle.write(payload)
        return path, values.reshape(2, 3)

    def test_safetensors_bfloat16_load_honors_device(self):
        from safetensors import safe_open

        path, expected = self._write_bfloat16_safetensor()
        # The loader's default CPU contract must override a globally active CUDA
        # mode. This is the real-checkpoint path used by Transformers before a
        # module is explicitly moved to its execution device.
        with jt.flag_scope(use_cuda=1 if jt.has_cuda else 0):
            with safe_open(path, framework="pt", device="cpu") as handle:
                value = handle.get_tensor("weight")
                sliced = handle.get_slice("weight")[:]
            self.assertEqual(str(value.dtype), "torch.bfloat16")
            self.assertEqual(str(sliced.dtype), "torch.bfloat16")
            self.assertEqual(str(value.device), "cpu")
            self.assertEqual(str(sliced.device), "cpu")
            self.ac(value.float32().numpy(), expected, atol=0, rtol=0)
            self.ac(sliced.float32().numpy(), expected, atol=0, rtol=0)

        if jt.has_cuda:
            with safe_open(path, framework="pt", device="cuda") as handle:
                value = handle.get_tensor("weight")
            self.assertEqual(str(value.dtype), "torch.bfloat16")
            self.assertTrue(value.is_cuda)
            self.ac(value.float32().numpy(), expected, atol=0, rtol=0)

    def test_safetensors_bfloat16_save_preserves_dtype(self):
        from safetensors import safe_open
        from safetensors.torch import save_file

        expected = np.asarray(
            [[1.0, -2.5, 3.125], [0.0078125, 128.0, -0.25]], dtype=np.float32)
        value = torch.tensor(expected, dtype=torch.bfloat16, device="cpu")
        path = self.path("bf16-save.safetensors")
        save_file({"weight": value}, path)
        with safe_open(path, framework="pt", device="cpu") as handle:
            self.assertEqual(handle.get_dtype("weight"), "BF16")
            loaded = handle.get_tensor("weight")
        self.assertEqual(str(loaded.dtype), "torch.bfloat16")
        self.assertEqual(str(loaded.device), "cpu")
        self.ac(loaded.float32().numpy(), expected, atol=0, rtol=0)

    def test_safetensors_short_payload_fails_explicitly(self):
        from safetensors import safe_open

        path, _ = self._write_bfloat16_safetensor("truncated")
        with open(path, "rb+") as handle:
            handle.seek(-2, os.SEEK_END)
            handle.truncate()
        from safetensors import SafetensorError
        with self.assertRaisesRegex(
                (SafetensorError, EOFError),
                "(?:invalid tensor offsets|short safetensors payload).*truncated"):
            with safe_open(path, framework="pt", device="cpu") as handle:
                handle.get_tensor("truncated")

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
            src = nn.Conv2d(3, 5, 3, padding=1).to(dev)
            out_src = src(torch.tensor(x, device=dev)).numpy()
            p = self.path(f"sd_{dev}.pkl")
            torch.save(src.state_dict(), p)
            sd = torch.load(p)
            dst = nn.Conv2d(3, 5, 3, padding=1).to(dev)
            dst.load_state_dict(sd)
            out_dst = dst(torch.tensor(x, device=dev)).numpy()
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

    def test_save_syncs_pending_updates(self):
        def body(dev):
            p = self.path(f"pending_{dev}.pkl")
            x = torch.zeros(4)
            x.update(x + 3)
            torch.save({"x": x}, p)
            y = torch.load(p)["x"]
            self.ac(y.numpy(), np.full(4, 3, dtype=np.float32),
                    atol=0, rtol=0, msg=f"pending update saved {dev}")
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

    def test_state_dict_keeps_tied_weight_aliases(self):
        # PyTorch de-duplicates tied Parameters in named_parameters(), but keeps
        # every public attribute path in state_dict() for checkpoint compatibility.
        class TiedLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(3, 3, bias=False)
                self.decoder = nn.Linear(3, 3, bias=False)
                self.decoder.weight = self.encoder.weight

        def body(dev):
            model = TiedLinear()
            self.assertEqual(
                [name for name, _ in model.named_parameters()],
                ["encoder.weight"],
                f"tied parameter is enumerated once {dev}",
            )
            aliases = list(model.named_parameters(remove_duplicate=False))
            self.assertEqual(
                [name for name, _ in aliases],
                ["encoder.weight", "decoder.weight"],
                f"all tied parameter paths are enumerated {dev}",
            )
            self.assertIs(
                aliases[0][1], aliases[1][1],
                f"tied parameter paths preserve identity {dev}",
            )
            state = model.state_dict()
            self.assertEqual(
                set(state), {"encoder.weight", "decoder.weight"},
                f"tied state keys {dev}",
            )
            self.assertEqual(
                state["encoder.weight"].untyped_storage().data_ptr(),
                state["decoder.weight"].untyped_storage().data_ptr(),
                f"tied state aliases share storage {dev}",
            )
            weight = state["encoder.weight"]
            storage = weight.untyped_storage()
            self.assertEqual(weight.data_ptr(), storage.data_ptr(), dev)
            last = weight.view(-1)[-1]
            self.assertEqual(last.untyped_storage().data_ptr(), storage.data_ptr(),
                             f"view retains base storage {dev}")
            self.assertEqual(
                last.data_ptr(),
                storage.data_ptr() + last.storage_offset() * weight.element_size(),
                f"view data pointer reflects storage offset {dev}")
            self.assertEqual(
                last.data_ptr() + weight.element_size(),
                storage.data_ptr() + storage.nbytes(),
                f"full view covers its root storage {dev}",
            )

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
            src = nn.Conv2d(2, 4, 3).to(dev)
            sd = src.state_dict(to="numpy")
            dst = nn.Conv2d(2, 4, 3).to(dev)
            dst.load_state_dict(sd)
            for k, v in dst.named_parameters():
                self.ac(v.numpy(), sd[k], atol=0, rtol=0,
                        msg=f"param {k} restored {dev}")
            input_tensor = torch.tensor(x, device=dev)
            self.ac(dst(input_tensor).numpy(), src(input_tensor).numpy(),
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

    def test_load_state_dict_preserves_target_dtype(self):
        def body(dev):
            # torch.nn, not jittor.nn: ``.data`` is the torch frontend's
            # detached alias here, where the native one is the numpy view.
            dst = torch.nn.Linear(4, 3)
            dst.weight.data = dst.weight.data.bfloat16()
            dst.bias.data = dst.bias.data.bfloat16()
            src = {
                "weight": torch.ones_like(dst.weight.float()),
                "bias": torch.zeros_like(dst.bias.float()),
            }
            dst.load_state_dict(src)
            params = dict(dst.named_parameters())
            self.assertEqual(params["weight"].dtype, torch.bfloat16, f"weight dtype {dev}")
            self.assertEqual(params["bias"].dtype, torch.bfloat16, f"bias dtype {dev}")

        both_devices(body)

    def test_load_state_dict_assign_replaces_tied_parameter(self):
        class TiedBias(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.decoder = nn.Linear(1, 2, bias=False)
                self.bias = nn.Parameter(torch.zeros(2, device=device))
                self.decoder.bias = self.bias

            def tie_weights(self):
                self.decoder.bias = self.bias

        def body(dev):
            model = TiedBias(dev)
            canonical = torch.tensor([-3.0, 0.5], device=dev)
            duplicate = torch.zeros(2, device=dev)

            model.load_state_dict(
                {"bias": canonical}, strict=False, assign=True)
            self.assertIsNot(model.bias, model.decoder.bias, dev)
            self.ac(model.bias.numpy(), [-3.0, 0.5], atol=0, rtol=0, msg=dev)

            model.decoder.load_state_dict(
                {"bias": duplicate}, strict=False, assign=True)
            self.ac(model.bias.numpy(), [-3.0, 0.5], atol=0, rtol=0, msg=dev)
            self.ac(model.decoder.bias.numpy(), [0.0, 0.0], atol=0, rtol=0, msg=dev)

            model.tie_weights()
            self.assertIs(model.bias, model.decoder.bias, dev)
            self.ac(model.decoder.bias.numpy(), [-3.0, 0.5], atol=0, rtol=0, msg=dev)

        both_devices(body)

    def test_load_state_dict_assign_promotes_frontend_tensor_parameter(self):
        # Transformers' low-memory loader can leave a plain frontend Tensor in
        # a parameter slot before assign=True installs the checkpoint value.
        # The replacement must use nn.Parameter rather than calling Tensor with
        # a requires_grad keyword.
        def body(dev):
            model = nn.Linear(2, 1).to(dev)
            frontend_weight = torch.tensor([[0.0, 0.0]], device=dev)
            frontend_weight.requires_grad_(True)
            model.weight = frontend_weight
            source = torch.tensor([[2.0, 3.0]], device=dev)
            model.load_state_dict({"weight": source}, strict=False, assign=True)
            self.assertIsInstance(model.weight, nn.Parameter)
            self.assertTrue(model.weight.requires_grad)
            self.ac(model.weight.numpy(), [[2.0, 3.0]], atol=0, rtol=0, msg=dev)

        both_devices(body)

    def test_load_state_dict_assign_does_not_mutate_shared_source(self):
        class Roles(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.p1 = nn.Parameter(torch.zeros(2, device=device))
                self.p2 = nn.Parameter(torch.zeros(2, device=device))
                self.register_buffer(
                    "buf", torch.zeros(2, device=device), persistent=True)
                # The module-owned name remains the stable buffer contract even
                # if a prior replacement lost the per-Var role marker.
                self.buf.is_buffer = False
                self.buf._is_torch_parameter = True

        def body(dev):
            source = torch.tensor([4.0, 5.0], device=dev)
            source.requires_grad_(False)
            before = (
                source.requires_grad,
                getattr(source, "is_buffer", False),
                getattr(source, "_is_torch_parameter", False),
                source.is_meta,
            )
            model = Roles(dev)
            model.load_state_dict(
                {"p1": source, "p2": source, "buf": source}, assign=True)

            after = (
                source.requires_grad,
                getattr(source, "is_buffer", False),
                getattr(source, "_is_torch_parameter", False),
                source.is_meta,
            )
            self.assertEqual(after, before, dev)
            self.assertIsNot(model.p1, source, dev)
            self.assertIsNot(model.p2, source, dev)
            self.assertIs(model.buf, source, dev)
            self.assertIsNot(model.p1, model.p2, dev)
            self.assertIsNot(model.p1, model.buf, dev)
            self.assertTrue(isinstance(model.p1, nn.Parameter), dev)
            self.assertTrue(isinstance(model.p2, nn.Parameter), dev)
            self.assertFalse(isinstance(model.buf, nn.Parameter), dev)
            self.assertTrue(model.p1.requires_grad, dev)
            self.assertTrue(model.p2.requires_grad, dev)
            self.assertFalse(model.buf.requires_grad, dev)
            self.assertEqual([name for name, _ in model.named_buffers()], ["buf"])
            values = (model.p1, model.p2, model.buf)
            # Check all shared aliases before numpy readback: producing the
            # first array may migrate their common storage to the host.
            for value in values:
                value.sync()
                self.assertEqual(value.dtype, source.dtype, dev)
                self.assertEqual(value.device.type, dev, dev)
                self.assertEqual(value.is_cuda, dev == "cuda", dev)
                self.assertEqual(
                    value.location(), "device" if dev == "cuda" else "cpu", dev)
            for value in values:
                self.ac(value.numpy(), [4.0, 5.0], atol=0, rtol=0, msg=dev)

        both_devices(body)

    def test_load_state_dict_assign_shares_checkpoint_storage_and_roles(self):
        def body(dev):
            for source_parameter, target_role in (
                    (False, "parameter"), (True, "parameter"),
                    (False, "buffer"), (True, "buffer")):
                with self.subTest(device=dev, source_parameter=source_parameter,
                                  target_role=target_role):
                    source = torch.tensor([4.0, 5.0], dtype=torch.float64, device=dev)
                    if source_parameter:
                        source = nn.Parameter(source, requires_grad=False)
                    else:
                        source.requires_grad_(target_role == "buffer")
                    source._assign_alias_marker = "checkpoint"
                    model = nn.Module()
                    old = torch.empty(2, device="meta")
                    if target_role == "parameter":
                        old = nn.Parameter(old, requires_grad=True)
                        model.weight = old
                        model.unnamed_tie = old
                        key = "weight"
                    else:
                        model.register_buffer("buf", old)
                        key = "buf"
                    model.load_state_dict({key: source}, strict=False, assign=True)
                    value = getattr(model, key)
                    self.assertIsNot(value, old)
                    if source_parameter or target_role == "buffer":
                        self.assertIs(value, source)
                    else:
                        self.assertIsNot(value, source)
                    expected_parameter = target_role == "parameter" or source_parameter
                    self.assertEqual(isinstance(value, nn.Parameter), expected_parameter)
                    self.assertEqual(key in dict(model.named_parameters()), expected_parameter)
                    self.assertEqual(key in dict(model.named_buffers()), not expected_parameter)
                    self.assertEqual(value.requires_grad, target_role == "parameter" or not source_parameter)
                    expected_source_grad = (target_role == "parameter" if source_parameter
                                            else target_role == "buffer")
                    self.assertEqual(source.requires_grad, expected_source_grad)
                    self.assertEqual(source._assign_alias_marker, "checkpoint")
                    self.assertFalse(value.is_meta)
                    for tensor in (source, value):
                        tensor.sync()
                        self.assertEqual(tensor.dtype, torch.float64)
                        self.assertEqual(tensor.device.type, dev)
                        self.assertEqual(jt.core.dispatch_context([tensor])[0], dev)
                    if target_role == "parameter":
                        self.assertIs(model.unnamed_tie, old)
                        self.assertTrue(old.is_meta)
                    # Observe storage sharing through public mutations rather
                    # than relying on the separate Storage.data_ptr contract.
                    with torch.no_grad():
                        source.add_(2.0)
                    self.ac(value.detach().cpu().numpy(), [6.0, 7.0], atol=0, rtol=0)
                    with torch.no_grad():
                        value.add_(3.0)
                    self.ac(source.detach().cpu().numpy(), [9.0, 10.0], atol=0, rtol=0)
                    self.ac(value.detach().cpu().numpy(), [9.0, 10.0], atol=0, rtol=0)

        both_devices(body)

    def test_load_state_dict_assign_preserves_checkpoint_buffer_and_frozen_parameter(self):
        def body(dev):
            model = nn.Module()
            model.frozen = nn.Parameter(torch.zeros(2, device=dev), requires_grad=False)
            model.register_buffer("buf", torch.zeros(2, device=dev))
            source = torch.tensor([2.0, 3.0], dtype=torch.float64, device=dev,
                                  requires_grad=True)
            model.load_state_dict({"frozen": source, "buf": source}, assign=True)
            self.assertFalse(model.frozen.requires_grad, dev)
            self.assertTrue(model.buf.requires_grad, dev)
            self.assertTrue(source.requires_grad, dev)
            self.assertIsInstance(model.frozen, nn.Parameter, dev)
            self.assertNotIsInstance(model.buf, nn.Parameter, dev)
            self.assertEqual([name for name, _ in model.named_buffers()], ["buf"])
            for value in (model.frozen, model.buf):
                self.assertEqual(value.dtype, torch.float64, dev)
                self.assertEqual(value.device.type, dev, dev)
                self.ac(value.detach().cpu().numpy(), [2.0, 3.0], atol=0, rtol=0, msg=dev)

        both_devices(body)

    def test_load_state_dict_copy_into_meta_is_noop(self):
        def body(dev):
            model = nn.Module()
            model.weight = nn.Parameter(torch.empty(2, device="meta"))
            model.register_buffer("buf", torch.empty(2, device="meta"))
            old_weight, old_buffer = model.weight, model.buf
            self.assertTrue(old_weight.is_meta, dev)
            self.assertTrue(old_buffer.is_meta, dev)
            for value in (old_weight, old_buffer):
                self.assertEqual(value.device.type, "meta", dev)
                self.assertFalse(value.is_cpu, dev)
                self.assertFalse(value.is_cuda, dev)
            source = torch.tensor([4.0, 5.0], device=dev)
            with self.assertWarnsRegex(UserWarning, "meta.*no-op"):
                model.load_state_dict({"weight": source, "buf": source})
            self.assertIs(model.weight, old_weight, dev)
            self.assertIs(model.buf, old_buffer, dev)
            self.assertTrue(model.weight.is_meta, dev)
            self.assertTrue(model.buf.is_meta, dev)
            self.assertFalse(source.is_meta, dev)
            self.assertFalse(source.requires_grad, dev)

        both_devices(body)

    def test_load_state_dict_assign_nested_container_slots(self):
        from collections import OrderedDict

        def body(dev):
            model = nn.Module()
            model.numeric = nn.Sequential(nn.Linear(2, 1, bias=False))
            model.named = nn.Sequential(OrderedDict([
                ("projection", nn.Linear(2, 1, bias=False)),
            ]))
            # The same old holder is reachable through four independent
            # container paths; assignment must replace every named slot.
            shared = nn.Parameter(torch.empty((1, 2), device="meta"))
            model.numeric[0].weight = shared
            model.named[0].weight = shared
            model.listed = nn.ParameterList([shared])
            model.keyed = nn.ParameterDict({"projection": shared})
            sources = {
                name: torch.tensor([[index, index + 1]], dtype=torch.float64, device=dev)
                for index, name in enumerate((
                    "numeric.0.weight", "named.projection.weight",
                    "listed.0", "keyed.projection",
                ), start=1)
            }
            result = model.load_state_dict(sources, assign=True)
            self.assertEqual(result.missing_keys, [], dev)
            self.assertEqual(result.unexpected_keys, [], dev)
            parameters = dict(model.named_parameters())
            self.assertEqual(set(parameters), set(sources), dev)
            self.assertEqual(len({id(value) for value in parameters.values()}), 4, dev)
            for name, source in sources.items():
                value = parameters[name]
                self.assertIsNot(value, shared, name)
                self.assertIsInstance(value, nn.Parameter, name)
                self.assertFalse(value.is_meta, name)
                self.assertEqual(value.dtype, source.dtype, name)
                self.assertEqual(value.device.type, dev, name)
                self.assertTrue(value.requires_grad, name)
                self.assertFalse(source.requires_grad, name)
                self.ac(value.detach().cpu().numpy(), source.numpy(),
                        atol=0, rtol=0, msg=name)
            self.assertTrue(shared.is_meta, dev)

        both_devices(body)

    def test_meta_tied_bias_survives_until_checkpoint_retie(self):
        class TiedBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Linear(1, 2, bias=False)
                self.bias = nn.Parameter(torch.zeros(2))
                self.decoder.bias = self.bias

            def roberta_tie_weights(self):
                if self.decoder.bias.device.type == "meta":
                    self.decoder.bias = self.bias
                    return "decoder_from_bias"
                self.bias = self.decoder.bias
                return "bias_from_decoder"

        def body(dev):
            real = torch.tensor([3.0], device=dev)
            with torch.device("meta"):
                self.assertFalse(real.is_meta, dev)
                model = TiedBias()

            accelerate_model = TiedBias()
            accelerate_model.bias = nn.Parameter(
                accelerate_model.bias.to(torch.device("meta")))
            accelerate_model.decoder.bias = accelerate_model.bias

            for route, candidate in (
                ("device_context", model),
                ("accelerate_parameter_to", accelerate_model),
            ):
                label = f"{dev}:{route}"
                self.assertTrue(candidate.bias.is_meta, label)
                self.assertTrue(candidate.decoder.bias.is_meta, label)
                self.assertFalse(candidate.bias.is_cpu, label)
                self.assertFalse(candidate.bias.is_cuda, label)
                self.assertTrue(candidate.state_dict()["bias"].is_meta, label)

                canonical = torch.tensor([-0.75, 0.5], device=dev)
                candidate.load_state_dict(
                    {"bias": canonical}, strict=False, assign=True)

                self.assertFalse(candidate.bias.is_meta, label)
                self.assertTrue(candidate.decoder.bias.is_meta, label)
                self.assertIsNot(candidate.bias, candidate.decoder.bias, label)
                self.assertEqual(
                    candidate.roberta_tie_weights(), "decoder_from_bias", label)
                self.assertIs(candidate.bias, candidate.decoder.bias, label)
                self.ac(candidate.decoder.bias.numpy(), [-0.75, 0.5],
                        atol=0, rtol=0, msg=label)
    def test_load_state_dict_preserves_target_device(self):
        def body(dev):
            dst = torch.nn.Linear(4, 3)
            before = {
                name: (int(value.placement_backend), int(value.device_id))
                for name, value in dst.named_parameters()
            }
            source = {
                name: value.detach().cpu()
                for name, value in torch.nn.Linear(4, 3).state_dict().items()
            }
            dst.load_state_dict(source)
            after = {
                name: (int(value.placement_backend), int(value.device_id))
                for name, value in dst.named_parameters()
            }
            self.assertEqual(after, before, f"target placement {dev}")
            output = dst(torch.ones((2, 4)))
            output.sync()
            self.assertEqual(
                int(output.placement_backend),
                next(iter(before.values()))[0],
                f"forward placement {dev}",
            )

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
            src = Net().to(dev); src.eval()
            sd = src.state_dict()
            keys = set(sd.keys())
            self.assertTrue(any(k.startswith("conv1.") for k in keys),
                            f"dotted conv1 key {dev}: {sorted(keys)}")
            self.assertTrue(any("bn" in k and "running_mean" in k for k in keys),
                            f"bn running_mean in state_dict {dev}")
            input_tensor = torch.tensor(x, device=dev)
            out_src = src(input_tensor).numpy()
            dst = Net().to(dev); dst.eval()
            dst.load_state_dict(sd)
            out_dst = dst(input_tensor).numpy()
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

    def test_frombuffer_common_dtypes_and_offset(self):
        raw = np.arange(10, dtype=np.int32).tobytes()

        def body(dev):
            v = torch.frombuffer(raw, dtype=torch.int32, count=4, offset=8)
            self.assertEqual(v.numpy().dtype.name, "int32",
                             f"frombuffer dtype {dev}")
            self.ae(v.numpy(), np.arange(2, 6, dtype=np.int32),
                    msg=f"frombuffer values {dev}")
        both_devices(body)

    def test_frombuffer_requires_grad_registers_leaf(self):
        raw = np.asarray([1.0, 2.0, 3.0], dtype=np.float32).tobytes()

        def body(dev):
            # An unrelated live optimizer must not hide standalone input leaves.
            parameter = torch.nn.Parameter(torch.tensor([0.0]))
            optimizer = torch.optim.SGD([parameter], lr=0.1)
            v = torch.frombuffer(raw, dtype=torch.float32, requires_grad=True)
            self.assertFalse(v.is_stop_grad(), f"requires_grad leaf {dev}")
            loss = (v * v).sum()
            loss.backward()
            self.assertIsNotNone(optimizer)
            self.assertIsNotNone(v.grad, f"frombuffer grad populated {dev}")
            self.ac(v.grad.numpy(), np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
                    msg=f"frombuffer grad {dev}")
        both_devices(body)


class TestStateDictTorchKeywords(unittest.TestCase):
    """``Module.state_dict`` accepts Torch's keyword arguments.

    Wrapper modules forward them straight through -- ms-swift's tuner base
    calls ``self.base_model.state_dict(destination=..., prefix=...,
    keep_vars=...)`` -- so rejecting them broke saving an adapted model even
    though the forward and backward passes were fine.
    """

    def _module(self):
        return jt.nn.Sequential(jt.nn.Linear(4, 4), jt.nn.ReLU(), jt.nn.Linear(4, 2))

    def test_prefix_is_applied_to_every_key(self):
        module = self._module()
        plain = set(module.state_dict().keys())
        self.assertTrue(plain)
        prefixed = set(module.state_dict(prefix="base_model.").keys())
        self.assertEqual(prefixed, {"base_model." + key for key in plain})

    def test_destination_is_updated_and_returned(self):
        module = self._module()
        destination = {"already": 1}
        returned = module.state_dict(destination=destination)
        self.assertIs(returned, destination)
        self.assertIn("already", returned)
        for key in module.state_dict():
            self.assertIn(key, returned)

    def test_keep_vars_false_detaches_without_changing_values(self):
        module = jt.nn.Linear(4, 4)
        live = module.state_dict()["weight"]
        detached = module.state_dict(keep_vars=False)["weight"]
        self.assertIs(live, module.weight)
        self.assertIsNot(detached, module.weight)
        np.testing.assert_array_equal(detached.numpy(), live.numpy())
        # No gradient reaches the parameter through a detached entry.
        through_detached = jt.grad((detached * 2).sum(), module.weight, retain_graph=True)
        self.assertEqual(float(np.abs(through_detached.numpy()).max()), 0.0)
        through_live = jt.grad((live * 2).sum(), module.weight)
        self.assertEqual(float(np.abs(through_live.numpy()).max()), 2.0)

    def test_default_still_returns_live_vars(self):
        module = jt.nn.Linear(4, 4)
        self.assertIs(module.state_dict()["weight"], module.weight)


if __name__ == "__main__":
    unittest.main(verbosity=2)
