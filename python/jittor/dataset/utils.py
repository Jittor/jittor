# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers:
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import jittor as jt
import numpy as np
from collections.abc import Sequence, Mapping
from PIL import Image
import functools
import time

from jittor_utils import LOG

def get_random_list(n):
    return list(np.random.permutation(range(n)))

def get_order_list(n):
    return [i for i in range(n)]


def collate_batch(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    real_size = len(batch)
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, jt.Var):
        temp_data = jt.stack([data for data in batch], 0)
        return temp_data
    if elem_type is np.ndarray:
        temp_data = np.stack([data for data in batch], 0)
        return temp_data
    elif np.issubdtype(elem_type, np.integer):
        return np.int32(batch)
    elif isinstance(elem, int):
        return np.int32(batch)
    elif isinstance(elem, float):
        return np.float32(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, Mapping):
        return {key: collate_batch([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple):
        transposed = zip(*batch)
        return tuple(collate_batch(samples) for samples in transposed)
    elif isinstance(elem, Sequence):
        transposed = zip(*batch)
        return [collate_batch(samples) for samples in transposed]
    elif isinstance(elem, Image.Image):
        temp_data = np.stack([np.array(data) for data in batch], 0)
        return temp_data
    else:
        raise TypeError(f"Not support type <{elem_type.__name__}>")

class HookTimer:
    """Accumulate the wall time spent inside ``getattr(obj, attr)``.

    **Opt in, and put it back.** Constructing one no longer installs it: use it
    as a context manager, or call :meth:`install` / :meth:`uninstall` around
    the region you want measured.

    ``HookTimer(PIL.Image, "open")`` used to install itself in ``__init__``,
    from the top level of ``jittor.dataset.dataset`` -- so ``import
    jittor.dataset`` replaced ``PIL.Image.open``, process-wide, for every
    library in the process, with no way to undo it. And it replaced it with the
    timer OBJECT, which is not a function: ``inspect.signature``,
    ``functools.wraps`` and pickling all stopped working on ``PIL.Image.open``
    for code that had never heard of jittor. What installs now is a
    ``functools.wraps``-ed function, so the attribute keeps looking like what
    it replaced.
    """

    def __init__(self, obj, attr, install=False):
        self.obj = obj
        self.attr = attr
        #: kept for callers that reach for the un-timed callable
        self.origin = getattr(obj, attr)
        self.duration = 0.0
        self._wrapper = None
        # nesting depth, so an inner `with` does not un-hook the outer one
        self._depth = 0
        if install:
            self.install()

    @property
    def installed(self):
        return self._wrapper is not None

    def install(self):
        """Wrap the attribute. Nests; returns self so ``with`` works."""
        self._depth += 1
        if self.installed:
            return self
        origin = getattr(self.obj, self.attr)
        self.origin = origin

        @functools.wraps(origin)
        def timed(*args, **kw):
            start = time.time()
            try:
                return origin(*args, **kw)
            finally:
                # in a finally, so a raising call is still accounted for and a
                # single failure cannot silently stop the clock for good
                self.duration += time.time() - start

        timed._jittor_hook_timer = self
        self._wrapper = timed
        setattr(self.obj, self.attr, timed)
        return self

    def uninstall(self):
        """Put the original attribute back. Idempotent.

        If something else replaced the attribute after us, leave that in place
        rather than clobbering it, and say so -- restoring blindly would delete
        the other patch without a word.
        """
        if self._depth == 0:
            return
        self._depth -= 1
        if self._depth:
            return
        wrapper = self._wrapper
        if wrapper is None:
            return
        self._wrapper = None
        current = getattr(self.obj, self.attr, None)
        if current is wrapper:
            setattr(self.obj, self.attr, self.origin)
            return
        LOG.w(f"HookTimer: {self.obj!r}.{self.attr} was replaced by someone "
              f"else while hooked; leaving the newer value in place")

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, tb):
        self.uninstall()
        return False

    def __call__(self, *args, **kw):
        start = time.time()
        try:
            return self.origin(*args, **kw)
        finally:
            self.duration += time.time() - start

