"""FSDP2 properties that only exist on more than one rank.

Single-rank FSDP proves very little here: ``common._in_true_distributed()`` is
false on one process, so ``_init_true_fsdp_state`` returns without sharding
anything and the whole true-FSDP path never runs. Everything below therefore
runs under ``mpirun -np 2``.

CUDA is required, not a preference: jittor's MPI extern provides all_reduce,
broadcast and reduce but **no all_gather** (``extern/mpi/ops/``), so the shard
gather has no CPU implementation at all and FSDP2 multi-rank only exists on
NCCL.

Run: ``pytest tests/compat/torch/test_torch_fsdp2_multirank.py``. The outer
process launches the ranks; the assertions run inside them.
"""
import os
import unittest

import numpy as np

import jittor as torch
import jittor as jt

from _helpers.child_process import run_mpi_python
from _helpers.common import selected_device_types


_HAS_MPI = bool(getattr(jt.compile_extern, "has_mpi", False))
_INSIDE = bool(jt.compile_extern.inside_mpi()) if _HAS_MPI else False
_HAS_CUDA = bool(getattr(jt, "has_cuda", 0))
#: The launcher below starts a second, GPU-using job. A session that asked for
#: CPU must not have GPUs taken out from under it -- and on a shared machine
#: those are somebody else's cards, since this process has no say in which
#: `CUDA_VISIBLE_DEVICES` the child inherits.
_want = selected_device_types()
_CUDA_SELECTED = _HAS_CUDA and (_want is None or "cuda" in _want)
_RANKS = 2


def _spread(var):
    """Distance from the mean over ranks; 0 exactly when all ranks agree."""
    return float(jt.abs(var - var.mpi_all_reduce("mean")).max().item())


@unittest.skipIf(not _HAS_MPI, "this Jittor build has no MPI")
@unittest.skipIf(not _INSIDE, "runs inside mpirun; see TestLaunch below")
@unittest.skipIf(not _HAS_CUDA, "FSDP2 shard gather is NCCL-only")
class TestFSDP2InsideMpi(unittest.TestCase):
    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1
        self.assertEqual(jt.world_size, _RANKS)

    def tearDown(self):
        jt.flags.use_cuda = self._use_cuda

    # ---- fully_shard(mesh=) ------------------------------------------------

    def test_a_mesh_covering_the_whole_world_is_accepted(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("cuda", (_RANKS,), mesh_dim_names=("dp",))
        module = torch.nn.Linear(8, 8, bias=False)
        self.assertIs(fully_shard(module, mesh=mesh), module)

    def test_a_multi_dimensional_mesh_is_refused(self):
        # Stored and ignored before: shard.py shards by _world_size() whatever
        # the mesh says, so a 2-D plan silently became a 1-D shard over every
        # rank -- each parameter split the wrong number of ways.
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("cuda", (1, _RANKS), mesh_dim_names=("dp", "tp"))
        with self.assertRaises(NotImplementedError) as caught:
            fully_shard(torch.nn.Linear(8, 8, bias=False), mesh=mesh)
        self.assertIn("8.08", str(caught.exception))

    def test_a_mesh_smaller_than_the_world_is_refused(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp",))
        with self.assertRaises(NotImplementedError) as caught:
            fully_shard(torch.nn.Linear(8, 8, bias=False), mesh=mesh)
        message = str(caught.exception)
        self.assertIn("mesh", message.lower())

    def test_omitting_mesh_still_works(self):
        from torch.distributed.fsdp import fully_shard

        module = torch.nn.Linear(8, 8, bias=False)
        self.assertIs(fully_shard(module), module)

    # ---- clip_grad_norm_ ---------------------------------------------------

    def _sharded_model_with_grads(self):
        from torch.distributed.fsdp import fully_shard

        jt.set_global_seed(3)
        module = torch.nn.Linear(16, 16, bias=False)
        fully_shard(module)
        opt = torch.optim.SGD(module.parameters(), lr=0.0)
        rs = np.random.RandomState(5 + jt.rank)
        x = jt.array(rs.randn(4, 16).astype("float32"))
        opt.zero_grad()
        ((module(x) ** 2).mean()).backward()
        jt.sync_all(True)
        return module, opt

    def test_the_clipping_norm_is_the_whole_models_not_this_ranks_slice(self):
        # Each rank holds a different slice of the same logical gradient, so a
        # rank-local norm is always too small and every rank scales by its own,
        # different coefficient. The norm has to agree across ranks.
        module, _opt = self._sharded_model_with_grads()
        total = torch.nn.utils.clip_grad_norm_(list(module.parameters()), 1e9)
        self.assertGreater(float(total.item()), 0.0)
        self.assertAlmostEqual(_spread(total.reshape((1,))), 0.0, places=4)

    def test_it_matches_the_norm_of_the_gathered_gradient(self):
        # The value, not just its agreement: the sharded norm must equal the
        # norm of the full gradient assembled from every rank's slice.
        from jittor.compat import collectives

        module, _opt = self._sharded_model_with_grads()
        params = list(module.parameters())
        total = torch.nn.utils.clip_grad_norm_(params, 1e9)

        squares = []
        for p in params:
            g = p.grad
            if g is None:
                continue
            squares.append(float((g.cast("float64") ** 2).sum().item()))
        local = jt.array(np.array([sum(squares)], dtype="float64"))
        reference = float(jt.sqrt(collectives._reduce_scalar(local, "sum")).item())
        self.assertAlmostEqual(float(total.item()), reference, places=3)

    def test_a_non_sharded_model_is_not_reduced_across_ranks(self):
        # DDP's ranks already hold the same averaged gradient; reducing there
        # would count one norm N times and clip far too hard.
        jt.set_global_seed(11)
        module = torch.nn.Linear(16, 16, bias=False)
        opt = torch.optim.SGD(module.parameters(), lr=0.0)
        x = jt.array(np.random.RandomState(2).randn(4, 16).astype("float32"))
        opt.zero_grad()
        ((module(x) ** 2).mean()).backward()
        jt.sync_all(True)
        params = list(module.parameters())
        total = float(torch.nn.utils.clip_grad_norm_(params, 1e9).item())
        squares = sum(float((p.grad.cast("float64") ** 2).sum().item())
                      for p in params if p.grad is not None)
        self.assertAlmostEqual(total, float(np.sqrt(squares)), places=3)


@unittest.skipIf(not _HAS_MPI, "this Jittor build has no MPI")
@unittest.skipIf(_INSIDE, "this is the launcher; the ranks run the class above")
@unittest.skipIf(not _CUDA_SELECTED,
                 "FSDP2 shard gather is NCCL-only, and this session did not "
                 "select cuda -- launching one anyway would take GPUs the "
                 "session did not ask for")
class TestLaunch(unittest.TestCase):
    def test_run_the_rank_tests_under_mpirun(self):
        completed = run_mpi_python(
            _RANKS, ["-m", "pytest", "-q", "-p", "no:cacheprovider",
                     os.path.abspath(__file__)],
            timeout=2400, merge_stderr=True)
        self.assertEqual(completed.returncode, 0, completed.stdout[-8000:])


if __name__ == "__main__":
    unittest.main()
