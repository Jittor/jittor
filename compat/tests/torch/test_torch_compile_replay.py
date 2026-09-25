"""torch.compile(mode="reduce-overhead") replays the module's captured graph.

PyTorch's answer to per-call host overhead is to record a step once and replay
it, and ``reduce-overhead`` is how a user opts into that. Jittor maps it onto
`jittor._runtime.graph_replay`: an SD1.5 sampling loop with the UNet compiled
this way went 1030 -> 487 ms against PyTorch's 438. What has to hold is the
answer -- for every new input, whatever structure the module returns -- and
everything a caller does with a compiled module besides calling it.
"""

import dataclasses
import unittest

import numpy as np
import torch
from torch import nn

import jittor as jt
from _helpers import capability as _test_capability


@dataclasses.dataclass
class _Output:
    sample: object = None


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 16)
        self.l2 = nn.Linear(16, 8)

    def forward(self, x, timestep, *, context=None):
        # The diffusers timestep path: a host 0-d tensor moved to the device
        # inside the module, which a device recording copies on every launch.
        t = timestep[None].to(x.device).expand(x.shape[0]).to(x.dtype)
        h = torch.relu(self.l1(x) + context)
        y = self.l2(h) * t[:, None]
        return {"out": _Output(sample=y), "both": (h, torch.cat([y, y], 1))}


class _Case:
    """The tests, run once per device by the classes below."""

    device = "cpu"

    def setUp(self):
        torch.manual_seed(0)
        self.model = _Net().to(self.device).eval()
        self.compiled = torch.compile(self.model, mode="reduce-overhead")
        self.context = torch.randn(1, 16, device=self.device)

    def _feeds(self, n=8):
        rng = np.random.RandomState(1)
        return [(torch.tensor(rng.randn(4, 8).astype("float32"), device=self.device),
                 torch.tensor(1000 - 50 * i)) for i in range(n)]

    def _eager(self, x, t):
        flag = jt.flags.auto_graph_replay
        jt.flags.auto_graph_replay = 0
        try:
            with torch.no_grad():
                out = self.model(x, t, context=self.context)
            return out["out"].sample.cpu().numpy(), out["both"][1].cpu().numpy()
        finally:
            jt.flags.auto_graph_replay = flag

    def test_every_call_answers_for_its_own_input(self):
        feeds = self._feeds()
        expected = [self._eager(x, t) for x, t in feeds]
        with torch.no_grad():
            # No read between calls: the host runs ahead of the device, which
            # is when a recording that reuses a buffer too early goes wrong.
            results = [self.compiled(x, t, context=self.context) for x, t in feeds]
        jt.gc()
        for got, (want_y, want_cat) in zip(results, expected):
            self.assertIsInstance(got["out"], _Output)
            self.assertIsInstance(got["out"].sample, torch.Tensor)
            self.assertEqual(str(got["out"].sample.device), str(feeds[0][0].device))
            np.testing.assert_allclose(got["out"].sample.cpu().numpy(), want_y,
                                       rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(got["both"][1].cpu().numpy(), want_cat,
                                       rtol=1e-5, atol=1e-5)
        replay = self.compiled._replay
        self.assertIsNone(replay.refused)
        self.assertEqual(replay.stats["captured"], 1)
        self.assertEqual(replay.stats["replayed"], len(feeds))

    def test_a_call_that_records_gradients_runs_the_module(self):
        x = torch.randn(4, 8, device=self.device, requires_grad=True)
        t = torch.tensor(3)
        self.compiled(x, t, context=self.context)["out"].sample.sum().backward()
        want = x.detach().clone().requires_grad_(True)
        self.model(want, t, context=self.context)["out"].sample.sum().backward()
        np.testing.assert_allclose(x.grad.cpu().numpy(), want.grad.cpu().numpy(),
                                   rtol=1e-5, atol=1e-6)
        self.assertEqual(self.compiled._replay.stats["replayed"], 0)

    def test_it_stands_in_for_the_module(self):
        from torch._dynamo.eval_frame import OptimizedModule
        self.assertIsInstance(self.compiled, OptimizedModule)
        self.assertIsInstance(self.compiled, nn.Module)
        self.assertIs(self.compiled._orig_mod, self.model)
        self.assertIs(self.compiled.l1, self.model.l1)
        self.assertEqual(sorted(self.compiled.state_dict()),
                         sorted("_orig_mod." + k for k in self.model.state_dict()))
        # Compiling twice does not stack wrappers.
        self.assertIs(torch.compile(self.compiled, mode="reduce-overhead"), self.compiled)

    def test_other_modes_and_callables_run_as_written(self):
        self.assertIs(torch.compile(self.model), self.model)
        self.assertIs(torch.compile(self.model, mode="default"), self.model)
        self.assertIs(torch.compile(mode="max-autotune-no-cudagraphs")(self.model), self.model)
        f = lambda v: v * 2
        self.assertIs(torch.compile(f, mode="reduce-overhead"), f)
        wrapped = torch.compile(mode="reduce-overhead")(self.model)
        self.assertIs(wrapped._orig_mod, self.model)


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestCompileReplayCuda(_Case, unittest.TestCase):
    device = "cuda"

    def test_the_step_is_recorded_as_one_launch(self):
        feeds = self._feeds(10)
        expected = [self._eager(x, t) for x, t in feeds]
        with torch.no_grad():
            results = [self.compiled(x, t, context=self.context) for x, t in feeds]
        # Returns the pools' cached blocks to the driver: a recording that
        # still pointed at a workspace it had freed would now fault.
        jt.gc()
        with torch.no_grad():
            results += [self.compiled(x, t, context=self.context) for x, t in feeds]
        for got, (want_y, _) in zip(results, expected + expected):
            np.testing.assert_allclose(got["out"].sample.cpu().numpy(), want_y,
                                       rtol=1e-5, atol=1e-5)
        replay = self.compiled._replay
        self.assertIsNone(replay._graph_refused)
        self.assertGreater(replay.stats["graph"], len(feeds))


    def test_a_recording_owns_the_workspaces_it_used(self):
        # An fp16 cuDNN convolution takes a workspace from the temporary pool
        # and frees it when it returns -- in the middle of the recording. The
        # recorded kernel keeps the address, so the block must stay with the
        # recording: `jt.gc()` otherwise handed it back to the driver and the
        # next launch died with an illegal address.
        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                              nn.Conv2d(64, 64, 3, padding=1)).to("cuda", torch.float16).eval()
        compiled = torch.compile(model, mode="reduce-overhead")
        feeds = [torch.randn(2, 64, 64, 64, device="cuda", dtype=torch.float16)
                 for _ in range(8)]
        jt.flags.auto_graph_replay = 0
        try:
            with torch.no_grad():
                expected = [model(x).float().cpu().numpy() for x in feeds]
                for x, want in zip(feeds, expected):
                    got = compiled(x).float().cpu().numpy()
                    jt.gc()
                    np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-2)
        finally:
            jt.flags.auto_graph_replay = 1
        self.assertIsNone(compiled._replay._graph_refused)
        self.assertGreater(compiled._replay.stats["graph"], 0)


class TestCompileReplayCpu(_Case, unittest.TestCase):
    device = "cpu"



if __name__ == "__main__":
    unittest.main()
