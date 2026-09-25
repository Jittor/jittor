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
        self.assertIs(torch.compile(f), f)
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


class _Training:
    """`torch.compile(train_step, mode="reduce-overhead")` replays the whole step.

    Every test runs the same step on an eager twin and asks for the same
    numbers: the losses, the parameters, and the books the optimizer keeps.
    """

    device = "cpu"

    def _run(self, compiled, optimizer="adamw", steps=10, between=None):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 4)).to(self.device)
        if optimizer == "adamw":
            opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        elif optimizer == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        else:
            opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
        rng = np.random.RandomState(0)
        data = [(torch.tensor(rng.randn(16, 8).astype("float32"), device=self.device),
                 torch.tensor(rng.randn(16, 4).astype("float32"), device=self.device))
                for _ in range(steps)]

        def step(x, y):
            opt.zero_grad(set_to_none=True)
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
            return loss

        run = torch.compile(step, mode="reduce-overhead") if compiled else step
        losses = []
        for i, (x, y) in enumerate(data):
            losses.append(float(run(x, y)))
            sched.step()
            if between is not None:
                between(model)
        params = [p.detach().cpu().numpy() for p in model.parameters()]
        return losses, params, opt, run

    def _compare(self, **kwargs):
        want = self._run(False, **kwargs)
        got = self._run(True, **kwargs)
        np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=1e-6)
        for g, w in zip(got[1], want[1]):
            np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-6)
        return got[2], want[2], got[3]

    def test_an_adamw_step_is_replayed_with_its_schedule(self):
        opt, eager_opt, run = self._compare()
        if self.device == "cuda":
            # Only the fused update reads its step and learning rate on the
            # device; the per-parameter one bakes them in and is refused, and
            # then runs as written -- which the comparison above covers.
            self.assertIsNone(run.refused)
            self.assertGreater(run.stats["replayed"], 0)
        else:
            self.assertIn("bakes its step count", run.refused)
        # The step counts the optimizer reports advance on every replay.
        self.assertEqual(opt.n_step, eager_opt.n_step)
        self.assertEqual(opt.param_groups[0]["param_steps"] if "param_steps" in opt.param_groups[0]
                         else None,
                         eager_opt.param_groups[0]["param_steps"]
                         if "param_steps" in eager_opt.param_groups[0] else None)

    def test_reading_parameters_between_steps_changes_nothing(self):
        # Reading a parameter migrates it to the host; a device recording that
        # kept its old address would update memory that is no longer it.
        self._compare(between=lambda model: [p.detach().cpu() for p in model.parameters()])

    def test_an_optimizer_it_cannot_replay_runs_as_written(self):
        opt, _, run = self._compare(optimizer="rmsprop")
        self.assertIsNotNone(run.refused)

    def test_an_sgd_step_is_replayed_with_its_schedule(self):
        # The fused SGD reads its rate on the device, and its momentum buffers
        # are written in place, so a replay continues them.
        opt, eager_opt, run = self._compare(optimizer="sgd")
        self.assertEqual(opt.n_step, eager_opt.n_step)
        if self.device == "cuda":
            self.assertIsNone(run.refused)
            self.assertGreater(run.stats["replayed"], 0)


class TestCompileTrainingCpu(_Training, unittest.TestCase):
    device = "cpu"


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestCompileTrainingCuda(_Training, unittest.TestCase):
    device = "cuda"

    def test_a_gradient_scaler_skips_overflows_on_the_device(self):
        # The scaler decides on the device, not by reading its found-inf back:
        # the fused AdamW skips a flagged step in the kernel, and the scale
        # backs off and grows as arithmetic on device state the capture keeps.
        # One batch in four is scaled up until its gradients overflow.
        rng = np.random.default_rng(0)
        data = [(torch.tensor(rng.standard_normal((32, 64)).astype("float32")
                              * (1e10 if i == 2 else 1), device="cuda"),
                 torch.tensor(rng.standard_normal((32, 16)).astype("float32"), device="cuda"))
                for i in range(4)]
        init = [rng.standard_normal(shape).astype("float32") * 0.1
                for shape in ((256, 64), (256,), (16, 256), (16,))]

        def run(compiled):
            model = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 16)).cuda()
            with torch.no_grad():
                for p, value in zip(model.parameters(), init):
                    p.copy_(torch.tensor(value, device="cuda"))
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 100, growth_interval=3)

            def step(x, y):
                opt.zero_grad(set_to_none=True)
                loss = ((model(x) - y) ** 2).mean()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                return loss
            f = torch.compile(step, mode="reduce-overhead") if compiled else step
            scales = []
            for i in range(12):
                f(*data[i % 4])
                scales.append(scaler.get_scale())
            return scales, [p.detach().cpu().numpy() for p in model.parameters()], f
        want_scales, want_params, _ = run(False)
        got_scales, got_params, f = run(True)
        self.assertEqual(got_scales, want_scales)
        self.assertLess(min(want_scales), 2.0 ** 100)       # it did skip
        for g, w in zip(got_params, want_params):
            np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-6)
        self.assertIsNone(f.refused)
        self.assertEqual(f.stats["captured"], 1)

    def test_the_step_is_recorded_as_one_launch(self):
        _, _, run = self._compare(steps=12)
        self.assertIsNone(run._graph_refused)
        self.assertGreater(run.stats["graph"], 0)





if __name__ == "__main__":
    unittest.main()
