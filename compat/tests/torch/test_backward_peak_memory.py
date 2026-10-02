"""A Torch-mode backward peaks at one layer's worth, however deep the model.

Under the Torch frontend almost every op is built from Python, and a Python
Function's backward (LayerNorm, GELU, ...) creates Python-held gradients while
``jt.grad`` is still building the backward graph. An auto-flush landing there
treated each of them as a result, computed it and kept it -- with the forward
activations it reads -- until construction ended. Each flush pinned another
slice: twelve residual/LayerNorm/GELU blocks peaked at 2534 MB against 460 MB
fully lazy (PyTorch 1417 MB), and the step jumped rather than grew, from
172 MB at four blocks to 1309 MB at six, where construction first crossed the
128-operator flush threshold. ViT-B/16 training at batch 64 ran out of a
24 GB card that PyTorch finishes in 9 GB.

The peak is a process-wide maximum, so each measurement is its own process.
"""

import textwrap
import unittest

from _helpers import capability as _test_capability
from _helpers.child_process import run_python_child

import jittor as jt


_SCRIPT = textwrap.dedent("""
    import jittor as jt
    import torch
    jt.flags.auto_flush_ops = {flush}

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = torch.nn.LayerNorm(768)
            self.fc1 = torch.nn.Linear(768, 3072)
            self.fc2 = torch.nn.Linear(3072, 768)
        def forward(self, x):
            return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln(x))))

    model = torch.nn.Sequential(*[Block() for _ in range(12)]).to("cuda")
    # ViT-B/16 widths at batch 16: activations are ~10 MB, past the
    # 8 MB a flush waits for, as in the real model.
    x = torch.randn(16 * 197, 768, device="cuda")
    jt.sync_all(True)
    with jt.flag_scope(profile_memory_enable=2):
        model(x).square().mean().backward()
        jt.sync_all(True)
        peak = int(jt.get_max_memory_info().split("[!@#div1!@#]")[0])
    print("PEAK", peak)
""")


def _peak(flush):
    finished = run_python_child(["-c", _SCRIPT.format(flush=flush)],
                                env={"JITTOR_TORCH_SHIM": "1"}, timeout=1800)
    for line in finished.stdout.splitlines():
        if line.startswith("PEAK "):
            return int(line.split()[1])
    raise AssertionError(
        "no result for auto_flush_ops=%d:\nstdout:\n%s\nstderr:\n%s"
        % (flush, finished.stdout[-2000:], finished.stderr[-2000:]))


@unittest.skipUnless(
    _test_capability.check_accelerator("cuda", backend=jt).enabled,
    "auto-flush only runs on CUDA")
class TestBackwardPeakDoesNotDependOnTheFlush(unittest.TestCase):
    def test_the_default_flush_does_not_pin_the_backward(self):
        default, lazy = _peak(128), _peak(0)
        self.assertLessEqual(
            default, 1.25 * lazy,
            "backward peak %.0f MB with auto_flush_ops=128 against %.0f MB "
            "fully lazy" % (default / 2**20, lazy / 2**20))


@unittest.skipUnless(
    _test_capability.check_accelerator("cuda", backend=jt).enabled,
    "the pools count device memory only")
class TestMaxMemoryAllocatedSeesInsideABatch(unittest.TestCase):
    """``max_memory_allocated`` is a high-water mark, not a sample.

    It used to be sampled from Python at the moments some memory API was
    called, so a peak reached and released inside one batch never showed:
    training steps that filled a 22 GB card reported 0.1-1.3 GB.
    """

    def test_a_peak_released_within_the_batch_is_reported(self):
        import torch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.max_memory_allocated()
        big = torch.randn(8192, 8192, device="cuda")      # 256 MiB
        (big * 2).sum().item()
        del big
        torch.cuda.synchronize()
        self.assertGreaterEqual(torch.cuda.max_memory_allocated() - base,
                                8192 * 8192 * 4)
        self.assertGreaterEqual(
            torch.cuda.memory_stats()["allocated_bytes.all.peak"] - base,
            8192 * 8192 * 4)


if __name__ == "__main__":
    unittest.main()
