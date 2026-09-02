# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""What ``import jittor`` is allowed to change in a process that imports it.

Task 5.20. Two import-time statements reached outside jittor and could not be
undone, and neither said anything when it fired:

* ``jittor/misc/tensor_ops.py`` called ``set_global_seed(<wall clock>)`` at
  module level, which reseeds Python's ``random``, numpy's global RNG and
  cupy's. A caller who seeded numpy *before* importing jittor -- including a
  caller who never mentions jittor and merely imports a library that does --
  had their seed silently replaced with a different one on every run.

* ``jittor/dataset/dataset.py`` called ``HookTimer(PIL.Image, "open")`` at
  module level, which replaced ``PIL.Image.open`` for the whole process, with
  a non-function object (so ``inspect.signature``, ``functools.wraps`` and
  pickling stopped working on it) and no uninstall.

Each check runs in a child process, because this one has already imported
jittor and cannot observe its own import.

Run::  python -m pytest tests/core/test_import_side_effects.py
"""

import unittest

from _helpers.child_process import run_child_script


def run_probe(source):
    done = run_child_script(source, text=True, merge_stderr=True,
                            name="import_side_effects")
    return done.returncode, done.stdout


NUMPY_SEED_SURVIVES = '''
import numpy as np
np.random.seed(0)
expected = np.random.rand(5).tolist()

np.random.seed(0)
import jittor as jt              # must not touch numpy's global RNG
got = np.random.rand(5).tolist()

assert got == expected, "import jittor reseeded numpy: %r != %r" % (got, expected)
print("DONE")
'''

STDLIB_RANDOM_SEED_SURVIVES = '''
import random
random.seed(0)
expected = [random.random() for _ in range(5)]

random.seed(0)
import jittor as jt              # must not touch random's global state
got = [random.random() for _ in range(5)]

assert got == expected, "import jittor reseeded random: %r != %r" % (got, expected)
print("DONE")
'''

JITTOR_STILL_SEEDS_ITSELF = '''
import jittor as jt
seed = jt.get_seed()
assert isinstance(seed, int), seed
# a per-process seed is still set at import; it is just jittor's own
assert jt.random((4,)).shape[0] == 4
print("SEED", seed)
print("DONE")
'''

SET_GLOBAL_SEED_STILL_REACHES_OUT = '''
import random
import numpy as np
import jittor as jt

jt.set_global_seed(1234, different_seed_for_mpi=False)
a_np = np.random.rand(4).tolist()
a_py = [random.random() for _ in range(4)]
jt.set_global_seed(1234, different_seed_for_mpi=False)
assert np.random.rand(4).tolist() == a_np
assert [random.random() for _ in range(4)] == a_py
print("DONE")
'''

PIL_OPEN_IS_UNTOUCHED = '''
import inspect
import pickle
import PIL.Image

original = PIL.Image.open
original_signature = inspect.signature(original)

import jittor.dataset            # must not replace PIL.Image.open

assert PIL.Image.open is original, "import jittor.dataset replaced PIL.Image.open"
assert inspect.isfunction(PIL.Image.open) or inspect.isbuiltin(PIL.Image.open)
assert inspect.signature(PIL.Image.open) == original_signature
# picklable by reference, which a bare instance of a hook class is not
assert pickle.loads(pickle.dumps(PIL.Image.open)) is original
print("DONE")
'''

PIL_HOOK_IS_OPT_IN_AND_REVERSIBLE = '''
import functools
import inspect
import io as _io
import PIL.Image
import jittor as jt
import jittor.dataset

original = PIL.Image.open
buf = _io.BytesIO()
PIL.Image.new("RGB", (4, 4)).save(buf, format="PNG")

with jt.dataset.time_image_open() as timer:
    timer.duration = 0.0
    assert PIL.Image.open is not original, "the hook did not install"
    # ...but it still looks and behaves like the function it replaced
    assert inspect.signature(PIL.Image.open) == inspect.signature(original)
    assert PIL.Image.open.__name__ == original.__name__
    assert PIL.Image.open.__wrapped__ is original
    buf.seek(0)
    image = PIL.Image.open(buf)
    image.load()
    assert image.size == (4, 4)
    assert timer.duration > 0.0

assert PIL.Image.open is original, "the hook did not uninstall"
print("DONE")
'''

PIL_HOOK_NESTS = '''
import PIL.Image
import jittor as jt
import jittor.dataset

original = PIL.Image.open
with jt.dataset.time_image_open():
    hooked = PIL.Image.open
    with jt.dataset.time_image_open():
        assert PIL.Image.open is hooked
    # the inner exit must not un-hook the outer block
    assert PIL.Image.open is hooked
assert PIL.Image.open is original
print("DONE")
'''


class TestImportDoesNotReseedForeignRngs(unittest.TestCase):
    def _ok(self, source):
        code, out = run_probe(source)
        self.assertEqual(code, 0, out)
        self.assertIn("DONE", out)

    def test_import_does_not_reseed_numpy(self):
        self._ok(NUMPY_SEED_SURVIVES)

    def test_import_does_not_reseed_stdlib_random(self):
        self._ok(STDLIB_RANDOM_SEED_SURVIVES)

    def test_import_still_seeds_jittors_own_rng(self):
        self._ok(JITTOR_STILL_SEEDS_ITSELF)

    def test_set_global_seed_still_seeds_all_of_them_when_asked(self):
        self._ok(SET_GLOBAL_SEED_STILL_REACHES_OUT)


class TestImportDoesNotPatchPil(unittest.TestCase):
    def _ok(self, source):
        code, out = run_probe(source)
        self.assertEqual(code, 0, out)
        self.assertIn("DONE", out)

    def test_importing_jittor_dataset_leaves_pil_image_open_alone(self):
        self._ok(PIL_OPEN_IS_UNTOUCHED)

    def test_the_timer_is_opt_in_and_puts_pil_back(self):
        self._ok(PIL_HOOK_IS_OPT_IN_AND_REVERSIBLE)

    def test_nested_timer_scopes_do_not_unhook_early(self):
        self._ok(PIL_HOOK_NESTS)


if __name__ == "__main__":
    unittest.main()
