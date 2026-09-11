"""Running a broad slice of the Torch API leaves no device memory behind.

The existing memory contracts each drive one operation. A per-operation leak
that is small enough to look like allocator noise in isolation is only visible
once many different kernels have run: each one allocates its own workspace, and
a single kernel that keeps a reference holds its block for the life of the
process.

This is deliberately a *residue* contract, not a peak one. Peak usage is a
property of the operation being measured; residue after an explicit collection
is a property of the framework, and it is the one that decides whether a long
run survives. `jt.gc()` is what returns cached blocks to the driver, so the
comparison is baseline-after-collection against baseline-after-collection --
comparing raw `nvidia-smi` numbers would only re-measure the caching allocator.
"""

import functools
import unittest

import numpy as np
import torch


def _cuda_available():
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def requires_cuda(target):
    """Probe when the case runs, not when the file is imported.

    ``unittest.skipUnless(_cuda_available(), ...)`` evaluated the probe in a
    decorator argument, which runs at *collection* -- where this suite forbids
    backend work -- and froze one answer for the whole process.
    """
    if isinstance(target, type):
        setup = target.setUpClass.__func__

        @classmethod
        def checked_setup(cls):
            if not _cuda_available():
                raise unittest.SkipTest('cuda is required for device memory residue contracts')
            setup(cls)

        target.setUpClass = checked_setup
        return target

    @functools.wraps(target)
    def checked(*args, **kwargs):
        if not _cuda_available():
            raise unittest.SkipTest('cuda is required for device memory residue contracts')
        return target(*args, **kwargs)

    return checked


def _collect():
    import jittor as jt
    jt.gc()


def _device_bytes():
    import jittor as jt
    jt.sync_all()
    _collect()
    return int(jt.flags.stat_allocator_total_alloc_byte
               - jt.flags.stat_allocator_total_free_byte)


#: Operations driven for residue. Chosen to span distinct kernel families --
#: elementwise, reduction, matmul, structural, comparison and autograd -- since
#: a shared workspace leak would otherwise hide behind whichever family the
#: test happened to pick.
def _drain(value):
    """Force materialisation without adding a reduction of our own.

    Several operations already return a rank-0 result, and reducing one of
    those currently fails (KI-OPS-004), so an unconditional ``.sum()`` here
    would measure that defect instead of memory.
    """
    return float(value.item() if value.ndim == 0 else value.sum().item())


def _operations(a, b):
    yield a + b
    yield a * b
    yield a - b
    yield a / (b.abs() + 1.0)
    yield a.exp()
    yield a.abs().log()
    yield a.sqrt() if False else a.abs().sqrt()
    yield a.tanh()
    yield a.sigmoid()
    yield a.sum()
    yield a.mean()
    yield a.max()
    yield a.min()
    yield a.sum(dim=-1)
    yield a @ b.transpose(0, 1)
    yield a.transpose(0, 1).contiguous()
    yield a.reshape(-1)
    yield torch.cat([a, b], dim=0)
    yield torch.stack([a, b], dim=0)
    yield a[:2, :3]
    yield a.clone()
    yield (a > b).float()
    yield torch.softmax(a, dim=-1)
    yield torch.cumsum(a, dim=-1)
    yield torch.sort(a, dim=-1)[0]
    yield a.to(torch.float64)
    yield a.to(torch.float16).to(torch.float32)


@requires_cuda
class TestApiSurfaceMemoryResidue(unittest.TestCase):
    """A broad pass over the API returns to its own starting point."""

    #: Bytes tolerated between two collected baselines. Not zero: the allocator
    #: keeps small bookkeeping blocks whose count depends on which kernels ran.
    residue_tolerance = 8 * 1024 * 1024

    def _tensors(self):
        rng = np.random.RandomState(20260909)
        a = torch.tensor(rng.rand(64, 64).astype("float32")).cuda()
        b = torch.tensor(rng.rand(64, 64).astype("float32") + 1.0).cuda()
        return a, b

    def test_a_broad_pass_leaves_no_residue(self):
        a, b = self._tensors()
        for value in _operations(a, b):        # warm every kernel first, so the
            _drain(value)                      # baseline already owns their code
        baseline = _device_bytes()

        for _ in range(3):
            for value in _operations(a, b):
                _drain(value)
        after = _device_bytes()

        self.assertLess(
            after - baseline, self.residue_tolerance,
            "device memory grew by {0} bytes across three broad passes".format(
                after - baseline),
        )

    def test_autograd_passes_leave_no_residue(self):
        """Backward allocates a graph; releasing the graph must release it too."""
        a, _b = self._tensors()

        def one_pass():
            x = a.clone().requires_grad_(True)
            loss = (x * x).sum() + x.tanh().sum()
            loss.backward()
            return float(loss.item())

        one_pass()
        baseline = _device_bytes()
        for _ in range(5):
            one_pass()
        after = _device_bytes()

        self.assertLess(
            after - baseline, self.residue_tolerance,
            "autograd left {0} bytes behind over five passes".format(after - baseline),
        )


if __name__ == "__main__":
    unittest.main()
