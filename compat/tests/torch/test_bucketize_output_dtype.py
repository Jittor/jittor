"""Bucketize controls its public output dtype independently of integer sum."""

from functools import lru_cache
import json
import os
import subprocess
import unittest

import numpy as np
import jittor as jt
import torch

from jittor._runtime.fallback import forbid_backend_fallbacks
from _helpers.child_process import child_env
from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


_ORACLE = r'''
import json, sys, torch
assert vars(torch).get('_torch_owner') is None
assert torch._C.__file__.endswith(('.so', '.pyd'))
assert 'jittor' not in sys.modules
selected = sys.argv[1]
rows = {}
for name in ('float32', 'float64', 'int32', 'int64'):
    dtype = getattr(torch, name)
    values = torch.tensor([[-2, 0, 1, 3], [4, 5, 9, 11]], dtype=dtype, device=selected)
    boundaries = torch.tensor([-1, 1, 3, 4, 8], dtype=dtype, device=selected)
    for right in (False, True):
        for out_int32 in (False, True):
            result = torch.bucketize(values, boundaries, right=right, out_int32=out_int32)
            assert result.device.type == selected
            key = name + '/' + str(int(right)) + '/' + str(int(out_int32))
            rows[key] = {'dtype': str(result.dtype), 'values': result.cpu().tolist()}
sums = {}
for name in ('int32', 'int64'):
    value = torch.tensor([2**30, 2**30], dtype=getattr(torch, name), device=selected).sum()
    assert value.device.type == selected
    sums[name] = {'dtype': str(value.dtype), 'value': value.item()}
print('BUCKETIZE_ORACLE ' + json.dumps({'device': selected, 'rows': rows, 'sums': sums}))
'''


@lru_cache(maxsize=2)
def _reference(device):
    executable = os.environ.get('REAL_TORCH_PYTHON', '').strip()
    if not executable:
        raise unittest.SkipTest('REAL_TORCH_PYTHON is required for independent bucketize oracle')
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('JT_', 'JITTOR_', 'NCCL_'))}
    environment.pop('LD_LIBRARY_PATH', None)
    result = subprocess.run(
        [executable, '-c', _ORACLE, device],
        env=child_env(environment, inherit=False, repo_paths=False, without_torch_mode=True),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert result.returncode == 0, result.stdout
    records = [line.removeprefix('BUCKETIZE_ORACLE ') for line in result.stdout.splitlines()
               if line.startswith('BUCKETIZE_ORACLE ')]
    assert len(records) == 1, result.stdout
    reference = json.loads(records[0])
    assert reference['device'] == device and len(reference['rows']) == 16
    return reference


def _resident(value, device):
    value.sync()
    assert value.location() == ('cpu' if device == 'cpu' else 'device')
    assert value.device.type == device


class TestBucketizeOutputDtype(cu.JittorTestCase):
    def test_both_sides_and_output_dtypes_match_independent_torch(self, device):
        reference = _reference(device)
        with forbid_backend_fallbacks():
            for name in ('float32', 'float64', 'int32', 'int64'):
                dtype = getattr(torch, name)
                values = torch.tensor([[-2, 0, 1, 3], [4, 5, 9, 11]], dtype=dtype, device=device)
                boundaries = torch.tensor([-1, 1, 3, 4, 8], dtype=dtype, device=device)
                for value in (values, boundaries):
                    self.assertEqual(str(value.dtype), 'torch.' + name)
                    _resident(value, device)
                for right in (False, True):
                    for out_int32 in (False, True):
                        with self.subTest(input_dtype=name, right=right, out_int32=out_int32):
                            result = torch.bucketize(values, boundaries, right=right,
                                                    out_int32=out_int32)
                            _resident(result, device)
                            expected = reference['rows'][name + '/' + str(int(right)) +
                                                         '/' + str(int(out_int32))]
                            self.assertEqual(str(result.dtype), expected['dtype'])
                            self.assertEqual(str(result.dtype),
                                             'torch.int32' if out_int32 else 'torch.int64')
                            np.testing.assert_array_equal(result.cpu().numpy(), expected['values'])
            jt.sync_all(True)

    def test_default_integral_sum_stays_wide(self, device):
        reference = _reference(device)
        with forbid_backend_fallbacks():
            for name in ('int32', 'int64'):
                with self.subTest(input_dtype=name):
                    result = torch.tensor([2**30, 2**30], dtype=getattr(torch, name),
                                          device=device).sum()
                    _resident(result, device)
                    self.assertEqual(str(result.dtype), 'torch.int64')
                    self.assertEqual(str(result.dtype), reference['sums'][name]['dtype'])
                    self.assertEqual(result.item(), 2**31)
                    self.assertEqual(result.item(), reference['sums'][name]['value'])
            jt.sync_all(True)


instantiate_device_type_tests(TestBucketizeOutputDtype, globals(), only_for=('cpu', 'cuda'))
