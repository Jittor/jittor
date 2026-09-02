"""DDP really synchronises: two ranks that see different data stay identical.

Before 7.02 ``DistributedDataParallel`` was a plain forwarding wrapper -- no
bucket, no autograd hook, no initial broadcast, and ``no_sync()`` was a
``nullcontext``. Jittor all-reduces only inside ``opt.step(loss)``, while the
torch-idiomatic ``loss.backward(); opt.step()`` fills gradients through
``Var.backward`` and never touched MPI. On N ranks that trained N different
models from N different random initialisations and reported nothing, so 7.01
made it refuse on >1 rank rather than diverge quietly.

The criterion in the plan is exactly what ``test_two_ranks_stay_identical``
checks: after ``loss.backward(); opt.step()`` the two ranks' parameters agree.

Run: ``pytest tests/compat/torch/test_torch_ddp_grad_sync.py``. The outer
process starts ``mpirun -np 2`` and the assertions run inside the ranks.
"""
import os
import unittest

import numpy as np

import jittor as torch
import jittor as jt

from _helpers.child_process import run_mpi_python


_HAS_MPI = bool(getattr(jt.compile_extern, "has_mpi", False))
_INSIDE = bool(jt.compile_extern.inside_mpi()) if _HAS_MPI else False
_RANKS = 2


def _spread(var):
    """How far this rank's copy is from the mean over all ranks.

    Zero for every rank exactly when every rank holds the same value, which is
    the property these tests are about. Using a collective to check it avoids
    needing a side channel between the ranks.
    """
    return float(jt.abs(var - var.mpi_all_reduce("mean")).max().item())


def _model_spread(module):
    return max(_spread(p) for p in module.parameters())


def _train(ddp, module, steps=2, no_sync=False):
    opt = torch.optim.SGD(module.parameters(), lr=0.1)
    rs = np.random.RandomState(7 + jt.rank)      # different data per rank
    for _ in range(steps):
        x = jt.array(rs.randn(5, 4).astype("float32"))
        y = jt.array(rs.randn(5, 3).astype("float32"))
        opt.zero_grad()
        loss = ((ddp(x) - y) ** 2).mean()
        if no_sync:
            with ddp.no_sync():
                loss.backward()
        else:
            loss.backward()
        opt.step()
    jt.sync_all(True)


def _fresh_ddp():
    # Deliberately DIFFERENT init on each rank, so an absent broadcast shows up.
    jt.set_global_seed(1234 + jt.rank)
    module = torch.nn.Linear(4, 3)
    return torch.nn.parallel.DistributedDataParallel(module), module


@unittest.skipIf(not _HAS_MPI, "this Jittor build has no MPI")
@unittest.skipIf(not _INSIDE, "runs inside mpirun; see TestLaunch below")
class TestDDPInsideMpi(unittest.TestCase):
    def test_construction_broadcasts_rank0_parameters(self):
        _ddp, module = _fresh_ddp()
        self.assertEqual(jt.world_size, _RANKS)
        self.assertAlmostEqual(_model_spread(module), 0.0, places=6,
                               msg="each rank ran its own random init; DDP has "
                                   "to broadcast rank 0's weights")

    def test_two_ranks_stay_identical(self):
        # The plan's acceptance criterion for 7.02.
        ddp, module = _fresh_ddp()
        before = np.concatenate([p.numpy().ravel() for p in module.parameters()])
        _train(ddp, module)
        self.assertAlmostEqual(_model_spread(module), 0.0, places=6)
        after = np.concatenate([p.numpy().ravel() for p in module.parameters()])
        # Not a vacuous pass: the step has to have moved the weights, or two
        # untouched copies would agree for the wrong reason.
        self.assertGreater(np.abs(after - before).max(), 1e-4)

    def test_without_the_all_reduce_the_ranks_do_diverge(self):
        # Proves the check above can actually detect divergence. `no_sync()`
        # is the supported way to switch the all-reduce off, and it is what
        # the whole class used to do unconditionally.
        ddp, module = _fresh_ddp()
        _train(ddp, module, no_sync=True)
        self.assertGreater(
            _model_spread(module), 1e-4,
            "with the gradient all-reduce disabled the ranks must drift "
            "apart -- if they do not, the data is not actually different and "
            "test_two_ranks_stay_identical proves nothing")

    def test_no_sync_accumulates_locally_then_rejoins(self):
        # Gradient accumulation: several micro-batches under no_sync(), one
        # collective at the end. After the closing synchronised backward every
        # rank must hold the same gradient again.
        ddp, module = _fresh_ddp()
        rs = np.random.RandomState(11 + jt.rank)
        opt = torch.optim.SGD(module.parameters(), lr=0.1)
        opt.zero_grad()
        for micro in range(2):
            x = jt.array(rs.randn(5, 4).astype("float32"))
            y = jt.array(rs.randn(5, 3).astype("float32"))
            loss = ((ddp(x) - y) ** 2).mean()
            with ddp.no_sync():
                self.assertFalse(ddp.require_backward_grad_sync)
                loss.backward()
            self.assertTrue(ddp.require_backward_grad_sync)
        # The accumulated local gradients differ between ranks at this point.
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertGreater(max(_spread(g) for g in grads), 1e-6)
        # One more, synchronised, backward brings them back together.
        x = jt.array(rs.randn(5, 4).astype("float32"))
        y = jt.array(rs.randn(5, 3).astype("float32"))
        loss = ((ddp(x) - y) ** 2).mean()
        loss.backward()
        jt.sync_all(True)
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        self.assertAlmostEqual(max(_spread(g) for g in grads), 0.0, places=6)

    def test_the_gradient_is_synchronised_before_backward_returns(self):
        # torch does this in autograd hooks so that whatever runs between
        # backward() and step() -- clipping, norm logging -- sees the
        # synchronised gradient. Checking it here rather than after step()
        # is the whole point.
        ddp, module = _fresh_ddp()
        # An optimizer exists in every real training loop and is what publishes
        # the parameters as backward leaves; the point being checked is that
        # the gradient is already synchronised *before* its step() runs.
        opt = torch.optim.SGD(module.parameters(), lr=0.1)
        opt.zero_grad()
        rs = np.random.RandomState(3 + jt.rank)
        x = jt.array(rs.randn(5, 4).astype("float32"))
        y = jt.array(rs.randn(5, 3).astype("float32"))
        loss = ((ddp(x) - y) ** 2).mean()
        loss.backward()
        jt.sync_all(True)
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertAlmostEqual(max(_spread(g) for g in grads), 0.0, places=6)


@unittest.skipIf(not _HAS_MPI, "this Jittor build has no MPI")
@unittest.skipIf(_INSIDE, "this is the launcher; the ranks run the class above")
class TestLaunch(unittest.TestCase):
    def test_run_the_rank_tests_under_mpirun(self):
        completed = run_mpi_python(
            _RANKS, ["-m", "pytest", "-q", "-p", "no:cacheprovider",
                     os.path.abspath(__file__)],
            timeout=900, merge_stderr=True)
        self.assertEqual(completed.returncode, 0, completed.stdout[-6000:])


if __name__ == "__main__":
    unittest.main()
