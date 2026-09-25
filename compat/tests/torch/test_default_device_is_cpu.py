"""A tensor with no `device=` belongs on the CPU, as in torch.

The shim reported the default device from `jt.flags.use_cuda`, which conflates
two things torch keeps apart: *the accelerator is enabled* and *the accelerator
is the default device*. CUDA being available never moves torch's default off
the CPU; only `set_default_device` does.

What the conflation cost: MiniMax-H3's VAE does `latent = latent.float().cpu()`
and then builds its normalisation constants with a plain `torch.tensor(...)`.
Under torch both sit on the CPU. Here the constants landed on cuda:0 and the
subtract died inside `device_copy_op` with "Expected all tensor inputs on the
same backend and device" -- 245 s into a generation, in code that is ordinary
torch.
"""
import unittest

import jittor as jt
import torch


def _name(tensor):
    return str(getattr(tensor, "device", "?")).split(":")[0]


class TestDefaultDeviceIsCpu(unittest.TestCase):
    def setUp(self):
        # `set_default_device` toggles `use_cuda` as part of its job, so a test
        # that calls it leaks that flag process-wide unless the flag is put back
        # too. The state-leak gate catches exactly this, and it caught this file.
        self._use_cuda = int(jt.flags.use_cuda)

    def tearDown(self):
        torch.set_default_device("cpu")
        jt.flags.use_cuda = self._use_cuda

    def test_tensor_without_device_is_cpu(self):
        with jt.flag_scope(use_cuda=1 if jt.has_cuda else 0):
            self.assertEqual(_name(torch.tensor([1.0, 2.0])), "cpu")

    def test_get_default_device_is_cpu(self):
        with jt.flag_scope(use_cuda=1 if jt.has_cuda else 0):
            self.assertEqual(str(torch.get_default_device()), "cpu")

    def test_a_cpu_tensor_and_a_fresh_constant_combine(self):
        # The H3 VAE shape, reduced.
        with jt.flag_scope(use_cuda=1 if jt.has_cuda else 0):
            latent = torch.ones((1, 4, 2, 2, 2)).float().cpu()
            mean = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(1, 4, 1, 1, 1)
            self.assertEqual(_name(latent - mean), "cpu")

    @unittest.skipUnless(jt.has_cuda, "no CUDA device")
    def test_an_explicit_device_still_wins(self):
        with jt.flag_scope(use_cuda=1):
            self.assertEqual(_name(torch.tensor([1.0], device="cuda")), "cuda")

    @unittest.skipUnless(jt.has_cuda, "no CUDA device")
    def test_set_default_device_still_moves_it(self):
        # The point is not "always CPU"; it is "CPU until asked otherwise".
        with jt.flag_scope(use_cuda=1):
            torch.set_default_device("cuda")
            self.assertEqual(_name(torch.tensor([1.0])), "cuda")
            self.assertEqual(str(torch.get_default_device()).split(":")[0], "cuda")
            torch.set_default_device("cpu")
            self.assertEqual(_name(torch.tensor([1.0])), "cpu")

    @unittest.skipUnless(jt.has_cuda, "no CUDA device")
    def test_a_device_context_still_wins(self):
        # `with torch.device(d):` outranks the default, as in torch.
        with jt.flag_scope(use_cuda=1):
            with torch.device("cuda"):
                self.assertEqual(_name(torch.tensor([1.0])), "cuda")

    @unittest.skipUnless(jt.has_cuda, "no CUDA device")
    def test_like_factories_still_inherit(self):
        # `*_like` takes the source's device, not the default.
        with jt.flag_scope(use_cuda=1):
            source = torch.ones(3, device="cuda")
            self.assertEqual(_name(torch.zeros_like(source)), "cuda")


class _PackedDiff(torch.nn.Module):
    """Transformers' packed-sequence check, `masking_utils.py`, reduced."""

    def forward(self, position_ids):
        first = position_ids[:, :1] - 1
        diff = torch.diff(position_ids, prepend=first, dim=-1)
        return torch.cat([diff, position_ids], dim=-1)


class TestTheDefaultIsOnlyForConstructors(unittest.TestCase):
    """The default device answers *where a new tensor goes*, nothing more.

    It is not where an op puts its result. When every module call entered an
    ambient placement of "the default device", a whole forward ran under a CPU
    placement: `torch.diff(..., prepend=...)` on CUDA inputs built its
    concatenation buffer on the host and died in `setitem` -- Qwen3's first
    forward, every SD1.5 UNet step, BERT and ViT all failed on a GPU in
    ordinary `model.to("cuda")` code.
    """

    @unittest.skipUnless(jt.has_cuda, "no CUDA device")
    def test_a_forward_follows_its_inputs(self):
        with jt.flag_scope(use_cuda=1):
            ids = torch.arange(4, device="cuda").reshape(1, 4)
            out = _PackedDiff().to("cuda")(ids)
            self.assertEqual(_name(out), "cuda")
            self.assertEqual(out.cpu().tolist(), [[1, 1, 1, 1, 0, 1, 2, 3]])

    @unittest.skipUnless(jt.has_cuda, "no CUDA device")
    def test_a_bare_factory_inside_a_forward_is_still_cpu(self):
        # The other half: torch builds `torch.ones(2)` on the default device
        # whether or not it is called from a forward.
        class Factory(torch.nn.Module):
            def forward(self, x):
                return torch.ones(2)

        with jt.flag_scope(use_cuda=1):
            out = Factory()(torch.ones(2, device="cuda"))
            self.assertEqual(_name(out), "cpu")

    def test_a_module_built_without_a_device_is_cpu(self):
        with jt.flag_scope(use_cuda=1 if jt.has_cuda else 0):
            self.assertEqual(_name(torch.nn.Linear(2, 2).weight), "cpu")

    @unittest.skipUnless(jt.has_cuda, "no CUDA device")
    def test_optimizer_state_follows_its_parameters(self):
        with jt.flag_scope(use_cuda=1):
            model = torch.nn.Linear(4, 2).to("cuda")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            before = model.weight.detach().clone()
            model(torch.ones(3, 4, device="cuda")).sum().backward()
            optimizer.step()
            self.assertEqual(_name(model.weight), "cuda")
            self.assertFalse(torch.equal(model.weight.detach().cpu(),
                                         before.cpu()))


if __name__ == "__main__":
    unittest.main()
