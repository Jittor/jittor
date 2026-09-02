"""Loading a real ``.pt`` archive must honour the saved strides (task 7.15).

``torch.save`` does not flatten a tensor before writing it. It writes the whole
*storage* once and records, per tensor, ``(storage_offset, size, stride)``.  A
tensor that is a view -- a transpose, a slice, a column of a fused QKV matrix,
one head of an attention weight -- therefore travels as a non-contiguous
description of a larger buffer.

The reader in ``compat/torch/serialization.py`` used to take ``stride`` and drop
it: it sliced ``storage[offset:offset + numel]`` and reshaped that to ``size``.
For a contiguous tensor that is right.  For every view it silently reads a
*different* set of elements and reports success -- the checkpoint loads, the
shapes match, the numbers are wrong.

These tests build the archive by hand, in torch's on-disk format, so they need
no real torch installed. ``_write_archive`` documents that format.
"""
import io
import os
import pickle
import shutil
import tempfile
import unittest
import zipfile
from collections import OrderedDict

import numpy as np

import jittor as torch


class _Global(object):
    """A name to emit as a pickle ``GLOBAL``, e.g. ``torch.FloatStorage``.

    Callable only because ``pickle.save_reduce`` insists the reconstructor be
    callable; it is never actually called.
    """

    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.__name__ = name

    def __call__(self, *args, **kwargs):  # pragma: no cover - never invoked
        raise AssertionError("marker is a name, not a function")


class _StorageRef(object):
    """The tensor payload, written once to ``data/<key>`` and referenced by id."""

    def __init__(self, key, array, storage_type):
        self.key = key
        self.array = array
        self.storage_type = storage_type


class _Rebuild(object):
    """A tensor, pickled the way torch pickles one: a call to _rebuild_tensor_v2."""

    def __init__(self, storage, offset, size, stride):
        self.args = (storage, offset, tuple(size), tuple(stride), False,
                     OrderedDict())

    def __reduce__(self):
        return (_Global("torch._utils", "_rebuild_tensor_v2"), self.args)


# The C pickler has no overridable ``save``; the pure-Python one does.
class _Pickler(pickle._Pickler):
    """torch's pickling rules: storages by persistent id, names as GLOBAL."""

    def persistent_id(self, obj):
        if isinstance(obj, _StorageRef):
            return ("storage", _Global("torch", obj.storage_type), obj.key,
                    "cpu", obj.array.size)
        return None

    def save(self, obj, save_persistent_id=True):
        if isinstance(obj, _Global):
            self.write(pickle.GLOBAL
                       + (obj.module + "\n" + obj.name + "\n").encode("ascii"))
            self.memoize(obj)
            return
        return pickle._Pickler.save(self, obj, save_persistent_id)


def _write_archive(path, obj, storages):
    """Write ``obj`` as torch does: ``data.pkl`` plus one file per storage."""
    buf = io.BytesIO()
    _Pickler(buf, protocol=2).dump(obj)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())
        for storage in storages:
            zf.writestr("archive/data/" + storage.key, storage.array.tobytes())


class TestLoadHonoursSavedStrides(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jt_stride_")
        # 0..5 as float32, the storage every case below views differently.
        self.base = np.arange(6, dtype=np.float32)
        self.storage = _StorageRef("0", self.base, "FloatStorage")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _load(self, offset, size, stride):
        path = os.path.join(self.tmp, "ckpt.pt")
        _write_archive(path,
                       {"w": _Rebuild(self.storage, offset, size, stride)},
                       [self.storage])
        return torch.load(path)["w"]

    def test_a_contiguous_tensor_still_loads(self):
        # base.reshape(2, 3): offset 0, stride (3, 1)
        got = self._load(0, (2, 3), (3, 1))
        np.testing.assert_array_equal(got.numpy(), self.base.reshape(2, 3))

    def test_a_transposed_view_loads_transposed(self):
        # base.reshape(2, 3).T: size (3, 2), stride (1, 3), offset 0.
        # Dropping the stride reshapes to [[0,1],[2,3],[4,5]] -- same shape,
        # every element in the wrong place.
        got = self._load(0, (3, 2), (1, 3))
        np.testing.assert_array_equal(got.numpy(), self.base.reshape(2, 3).T)

    def test_a_column_slice_loads_its_own_columns(self):
        # base.reshape(2, 3)[:, 1:]: size (2, 2), stride (3, 1), offset 1.
        # Dropping the stride yields [[1,2],[3,4]] instead of [[1,2],[4,5]]:
        # the second row comes from the wrong place in the storage.
        got = self._load(1, (2, 2), (3, 1))
        np.testing.assert_array_equal(got.numpy(), self.base.reshape(2, 3)[:, 1:])

    def test_a_strided_row_selection_loads_every_other_row(self):
        # base.reshape(3, 2)[::2]: size (2, 2), stride (4, 1), offset 0.
        got = self._load(0, (2, 2), (4, 1))
        np.testing.assert_array_equal(got.numpy(), self.base.reshape(3, 2)[::2])

    def test_a_scalar_view_reads_its_own_element(self):
        got = self._load(4, (), ())
        # jittor has no 0-d Var yet (task 2.05), so shape is (1,) here; what
        # this pins is *which* element of the storage was read.
        self.assertEqual(got.numpy().reshape(-1).tolist(), [4.0])

    def test_an_expanded_view_repeats_the_element(self):
        # torch's expand() records stride 0; every row is the same element.
        got = self._load(2, (3, 2), (0, 1))
        np.testing.assert_array_equal(
            got.numpy(), np.array([[2., 3.], [2., 3.], [2., 3.]], np.float32))

    def test_a_description_that_runs_past_the_storage_is_refused(self):
        # Silently short-reading here is how a truncated or mismatched
        # checkpoint used to become wrong numbers.
        with self.assertRaises(pickle.UnpicklingError) as caught:
            self._load(0, (3, 3), (3, 1))
        self.assertIn("storage", str(caught.exception).lower())

    def test_a_negative_stride_is_refused(self):
        with self.assertRaises(pickle.UnpicklingError):
            self._load(5, (3,), (-1,))


if __name__ == "__main__":
    unittest.main()
