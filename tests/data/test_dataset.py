# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from jittor.dataset.dataset import ImageFolder, Dataset
import jittor.transform as transform

import jittor as jt
import unittest
import os
import numpy as np
import random
import pytest

from _helpers.child_process import run_child_script as _run_child_script
from _helpers.child_process import run_mpi_python

pass_this_test = False
msg = ""
mid = 0
if hasattr(os, "uname") and os.uname()[1] == "jittor-ce":
    mid = 1
try:
    traindir = ["/data1/cjld/imagenet/train/","/home/cjld/imagenet/train/"][mid]
    assert os.path.isdir(traindir)
except Exception as e:
    pass_this_test = True
    msg = str(e)

@unittest.skipIf(pass_this_test, f"can not run imagenet dataset test: {msg}")
class TestDataset(unittest.TestCase):
    def test_multi_workers(self):
        check_num_batch = 10
        tc_data = []

        def get_dataset():
            dataset = ImageFolder(traindir).set_attrs(batch_size=256, shuffle=False)
            dataset.set_attrs(transform = transform.Compose([
                transform.Resize(224),
                transform.ImageNormalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ]), num_workers=0)
            return dataset

        dataset = get_dataset()

        for i, data in enumerate(dataset):
            print("get batch", i)
            tc_data.append(data)
            if i==check_num_batch: break

        def check(num_workers, epoch=1):
            dataset = get_dataset().set_attrs(num_workers=num_workers)

            random.seed(0)

            for _ in range(epoch):
                for i, (images, labels) in enumerate(dataset):
                    print("compare", i)
                    assert np.allclose(images.data, tc_data[i][0].data), \
                         (images.sum(), tc_data[i][0].sum(), images.shape, 
                         tc_data[i][0].shape)
                    assert np.allclose(labels.data, tc_data[i][1].data)
                    if i==check_num_batch: break
            # dataset.terminate()
        check(1)
        check(2)
        check(4,2)

    def test_collate_batch(self):
        from jittor.dataset.utils import collate_batch
        batch = collate_batch([(1,1),(1,2),(1,3)])
        assert isinstance(batch[0], np.ndarray)
        assert isinstance(batch[1], np.ndarray)


class YourDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=10240)

    def __getitem__(self, k):
        self.tmp = None
        x = jt.array(k)
        y = x
        for i in range(10):
            for j in range(i+2):
                y = y + j - j
            y.stop_fuse()
        return x, y

        
class YourDataset2(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=16)

    def __getitem__(self, k):
        return np.random.rand(2)


class YourDataset3(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=16)

    def __getitem__(self, k):
        return random.randint(0,1000)


class YourDataset4(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=160)

    def __getitem__(self, k):
        return jt.rand(2)


class YourDataset5(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=160)

    def __getitem__(self, k):
        return { "a":np.array([1,2,3]) }

class TestDataset2(unittest.TestCase):
    def test_dataset_use_jittor(self):
        dataset = YourDataset().set_attrs(batch_size=256, shuffle=True, num_workers=4)
        dataset.tmp = jt.array([1,2,3,4,5])
        dataset.tmp.sync()
        for x, y in dataset:
            # dataset.display_worker_status()
            pass


    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_dataset_use_jittor_cuda(self):
        self.test_dataset_use_jittor()

class TestDatasetSeed(unittest.TestCase):
    def test_np(self):

        dataset = YourDataset2().set_attrs(batch_size=1, shuffle=True, num_workers=4)
        for _ in range(10):
            dd = []
            for d in dataset:
                dd.append(d.numpy())
            for i in range(len(d)):
                for j in range(i+1, len(d)):
                    assert not np.allclose(dd[i], dd[j])

    def test_py_native(self):
        import random

        jt.set_global_seed(0)
        dataset = YourDataset3().set_attrs(batch_size=1, shuffle=True, num_workers=4)
        for _ in range(10):
            dd = []
            for d in dataset:
                dd.append(d.numpy())
            for i in range(len(d)):
                for j in range(i+1, len(d)):
                    assert not np.allclose(dd[i], dd[j])

    def test_jtrand(self):
        import random

        jt.set_global_seed(0)
        dataset = YourDataset4().set_attrs(batch_size=1, shuffle=True, num_workers=4)
        for _ in range(10):
            dd = []
            for d in dataset:
                dd.append(d.numpy())
            for i in range(len(d)):
                for j in range(i+1, len(d)):
                    assert not np.allclose(dd[i], dd[j])

    def test_dict(self):
        import random

        jt.set_global_seed(0)
        dataset = YourDataset5().set_attrs(batch_size=1, shuffle=True, num_workers=4)
        for _ in range(10):
            dd = []
            for d in dataset:
                # breakpoint()
                assert isinstance(d, dict)
                assert isinstance(d['a'], jt.Var)
                np.testing.assert_allclose(d['a'].numpy(), [[1,2,3]])

    # Downloads an external dataset archive; see the note in
    # tests/models/test_resnet.py.
    @pytest.mark.network
    def test_cifar(self):
        from jittor.dataset.cifar import CIFAR10
        a = CIFAR10()
        a.set_attrs(batch_size=16)
        for imgs, labels in a:
            print(imgs.shape, labels.shape)
            assert imgs.shape == [16,32,32,3,]
            assert labels.shape == [16,]
            break

    def test_tensor_dataset(self):
        import jittor as jt
        from jittor.dataset import TensorDataset

        x = jt.array([1,2,3])
        y = jt.array([4,5,6])
        z = jt.array([7,8,9])

        dataset = TensorDataset(x, y, z)
        # dataset.set_attrs(batch_size=2)
        dataset.set_attrs(batch_size=1)

        for i,(a,b,c) in enumerate(dataset):
            # print(a,b,c)
            # print(a.shape)
            assert a.shape == [1]
            assert x[i] == a
            assert y[i] == b
            assert z[i] == c

    def test_children_died(self):
        if os.name == 'nt':
            # TODO: windows cannot pass this test now
            # don't know how to detect child died in windows
            # some clue: https://ikriv.com/blog/?p=1431
            return
        src = """
import jittor as jt
from jittor.dataset import Dataset
import numpy as np

class YourDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=160)

    def __getitem__(self, k):
        if k>100:
            while 1:
                pass
        return { "a":np.array([1,2,3]) }
if __name__ == "__main__":
    dataset = YourDataset()
    dataset.set_attrs(num_workers=2)

    for d in dataset:
        dataset.workers[0].p.kill()
        pass
"""
        r = run_child_script(src)
        s = r.stderr.decode()
        print(s)
        assert r.returncode != 0
        assert "SIGCHLD" in s
        assert "quick exit" in s


    @unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
    def test_dataset_shuffle_mpi(self):
        src = """
import jittor as jt
from jittor.dataset import Dataset
import numpy as np

class YourDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=160, shuffle=True)

    def __getitem__(self, k):
        return k

dataset = YourDataset()
dataset.set_attrs(num_workers=2)

for d in dataset:
    for a in d:
        print("CHECK: ", a.item())
"""
        fname = os.path.join(jt.flags.cache_path, "test_dataset_shuffle_mpi.py")
        with open(fname, 'w') as f:
            f.write(src)
        # mpirun starts both ranks itself, so neither inherits this process'
        # sys.path: without the helper's pinned PYTHONPATH they would import
        # the installed jittor and the assertion below would prove nothing.
        r = run_mpi_python(2, [fname], text=False)
        s = r.stdout.decode()
        # print(s)
        st = set([ l for l in s.splitlines() if l.startswith("CHECK:") ])
        assert r.returncode == 0
        # print(st)
        assert len(st) == 160, len(st)

    def test_children_died2(self):
        src = """
import jittor as jt
from jittor.dataset import Dataset
import numpy as np

class YourDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=160)

    def __getitem__(self, k):
        if k>100:
            while 1:
                pass
        return { "a":np.array([1,2,3]) }

if __name__ == "__main__":
    dataset = YourDataset()
    dataset.set_attrs(num_workers=2)

    for d in dataset:
        break
    dataset.terminate()
"""
        r = run_child_script(src)
        s = r.stderr.decode()
        print(s)
        assert r.returncode == 0
        

TINY_DATASET_WITH_FLAG_SRC = """
import os
import numpy as np
import jittor as jt
from jittor.dataset import Dataset
import jittor.dataset.dataset as dsmod

print("MP_LOG_V_TRUTHY=%s" % bool(dsmod.mp_log_v))

class Tiny(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=8, batch_size=4, shuffle=False, num_workers=2)
    def __getitem__(self, k):
        return np.array([k], dtype="float32")

if __name__ == "__main__":
    for batch in Tiny():
        pass
    print("DONE")
"""


# ---------------------------------------------------------------------------
#  helpers for tests that need a fresh interpreter
# ---------------------------------------------------------------------------
def run_child_script(src, extra_env=None, timeout=300):
    """Run ``src`` in a fresh interpreter that imports THIS tree's jittor.

    Kept as a named wrapper because several tests below call it; the pinning it
    used to do by hand now lives in ``_helpers.child_process``, which every
    child-launching test shares.
    """
    return _run_child_script(src, env=extra_env, timeout=timeout,
                             directory=jt.flags.cache_path, name="dataset_child")


class TestChildScriptHelper(unittest.TestCase):
    def test_child_imports_this_tree(self):
        from pathlib import Path
        r = run_child_script(
            "import jittor, os\n"
            "print('JITTOR_AT=' + os.path.dirname(os.path.dirname(jittor.__file__)))\n")
        out = r.stdout.decode()
        assert r.returncode == 0, r.stderr.decode()[-3000:]
        line = [l for l in out.splitlines() if l.startswith("JITTOR_AT=")]
        assert line, out
        expected = str(Path(__file__).resolve().parents[2] / "python")
        self.assertEqual(line[0][len("JITTOR_AT="):], expected)


class TestWorkerLogFlag(unittest.TestCase):
    """``mp_log_v`` is a debug switch. ``os.environ.get(name, 0)`` returns the
    string ``"0"``, which is truthy, so ``mp_log_v=0`` used to turn the worker
    chatter ON -- the opposite of what it says."""

    SRC = TINY_DATASET_WITH_FLAG_SRC

    def _run(self, value):
        env = {} if value is None else {"mp_log_v": value}
        r = run_child_script(self.SRC, extra_env=env)
        out = r.stdout.decode()
        assert r.returncode == 0, out[-2000:] + r.stderr.decode()[-2000:]
        assert "DONE" in out, out[-2000:]
        truthy = [l for l in out.splitlines() if l.startswith("MP_LOG_V_TRUTHY=")]
        assert truthy, out
        return truthy[0].endswith("True"), out

    def test_zero_turns_the_log_off(self):
        truthy, out = self._run("0")
        self.assertFalse(truthy, "mp_log_v=0 must be falsy")
        assert "recv buffer" not in out, out[-2000:]

    def test_unset_turns_the_log_off(self):
        truthy, out = self._run(None)
        self.assertFalse(truthy, "unset mp_log_v must be falsy")
        assert "recv buffer" not in out, out[-2000:]

    def test_one_turns_the_log_on(self):
        truthy, out = self._run("1")
        self.assertTrue(truthy, "mp_log_v=1 must be truthy")
        assert "recv buffer" in out, out[-2000:]


class TestDatasetDeepcopy(unittest.TestCase):
    """``Dataset.__deepcopy__`` has to keep the memo contract: register the new
    object (not its ``id``) and thread the memo through the attribute copies.
    Otherwise a reference back to the dataset recurses forever and objects that
    two places share come back as two separate copies."""

    def _tiny(self):
        class Tiny(Dataset):
            def __init__(self):
                super().__init__()
                self.set_attrs(total_len=4, batch_size=2, shuffle=False)

            def __getitem__(self, k):
                return np.array([k], dtype="float32")
        return Tiny()

    @staticmethod
    def _stack_depth():
        import sys
        depth, frame = 0, sys._getframe()
        while frame is not None:
            depth += 1
            frame = frame.f_back
        return depth

    def test_reference_back_to_the_dataset_is_the_copy(self):
        import sys
        from copy import deepcopy
        ds = self._tiny()
        ds.sibling = ds
        # A correct copy needs only a handful of frames. Cap the limit just
        # above the current depth so a regression fails in milliseconds with
        # RecursionError instead of grinding to the default limit.
        previous = sys.getrecursionlimit()
        sys.setrecursionlimit(self._stack_depth() + 120)
        try:
            copied = deepcopy(ds)
        finally:
            sys.setrecursionlimit(previous)
        assert copied is not ds
        assert copied.sibling is copied, "a cycle must resolve to the copy"

    def test_shared_object_stays_shared(self):
        from copy import deepcopy
        ds = self._tiny()
        shared = {"payload": [1, 2, 3]}
        ds.extra = shared
        holder = {"ds": ds, "shared": shared}
        copied = deepcopy(holder)
        assert copied["ds"] is not ds
        assert copied["ds"].extra is copied["shared"], \
            "the memo must be threaded into the attribute copies"
        assert copied["ds"].extra is not shared

    def test_dataset_seen_twice_copies_once(self):
        from copy import deepcopy
        ds = self._tiny()
        copied = deepcopy({"a": ds, "b": ds})
        assert copied["a"] is copied["b"]
        assert isinstance(copied["a"], Dataset)

    def test_copy_still_works_and_iterates(self):
        from copy import deepcopy
        ds = self._tiny()
        copied = deepcopy(ds)
        assert copied.dataset is copied
        assert copied.total_len == 4
        assert len([b for b in copied]) == 2


class TestWorkerExceptionPropagation(unittest.TestCase):
    """An exception raised inside a worker must come back to the caller as an
    exception. It used to be delivered as SIGINT to the parent, which jittor's
    handler turns into an immediate process exit -- indistinguishable from the
    user pressing Ctrl-C, and impossible to catch."""

    SRC = """
import numpy as np
import jittor as jt
from jittor.dataset import Dataset

class Boom(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=64, batch_size=4, shuffle=False, num_workers=2)
    def __getitem__(self, k):
        if k == 7:
            raise ValueError("boom in worker")
        return np.array([k], dtype="float32")

if __name__ == "__main__":
    ds = Boom()
    try:
        for batch in ds:
            pass
    except Exception as e:
        print("RAISED=%s" % type(e).__name__)
        print("HAS_MESSAGE=%s" % ("boom in worker" in str(e)))
        print("HAS_TRACEBACK=%s" % ("__getitem__" in str(e)))
        print("HAS_VALUEERROR=%s" % ("ValueError" in str(e)))
    else:
        print("NO_EXCEPTION")
    print("STILL_ALIVE")
"""

    def test_worker_exception_reaches_the_caller(self):
        r = run_child_script(self.SRC)
        out = r.stdout.decode()
        err = r.stderr.decode()
        assert r.returncode == 0, (
            "the parent must survive to raise, got returncode %s\\n%s\\n%s"
            % (r.returncode, out[-2000:], err[-3000:]))
        assert "STILL_ALIVE" in out, out[-2000:] + err[-2000:]
        assert "NO_EXCEPTION" not in out, out[-2000:]
        assert "RAISED=RuntimeError" in out, out[-2000:]
        assert "HAS_MESSAGE=True" in out, out[-2000:]
        assert "HAS_TRACEBACK=True" in out, out[-2000:]
        assert "HAS_VALUEERROR=True" in out, out[-2000:]
        # and nothing pretends the user pressed Ctrl-C
        assert "Caught SIGINT" not in err, err[-2000:]


if __name__ == "__main__":
    unittest.main()
