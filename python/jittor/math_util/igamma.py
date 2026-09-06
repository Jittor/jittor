"""Incomplete gamma with shared mathematical source and a CUDA launch owner."""

from pathlib import Path

from jittor.backends.cuda.kernels.math.igamma import igamma as _igamma_cuda


_SHARED_HEADER = (Path(__file__).parent / "src" / "igamma.h").read_text(encoding="utf8")


def igamma(alpha, x):
    return _igamma_cuda(alpha, x, _SHARED_HEADER)
