"""Worker admission follows the caller's current CPU affinity."""
import os
import sys
import unittest

from _helpers.child_process import run_python_child


_PROBE = r'''
import os
from pathlib import Path
import jittor as jt
import jittor_utils

allowed = sorted(os.sched_getaffinity(0))
groups = {}
for cpu in allowed:
    path = Path('/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list' % cpu)
    if path.is_file():
        groups.setdefault(path.read_text(), []).append(cpu)
pair = next((cpus[:2] for cpus in groups.values() if len(cpus) >= 2), None)
if pair is None:
    print('PLACEMENT_SKIP')
    raise SystemExit(0)

jt.flags.async_execution = 0
jt.flags.auto_graph_replay = 0
probe = jittor_utils.compile_module(r"""
#include "bindings/pyjt/py_converter.h"
#include "runtime/async_exec.h"
namespace jittor {
// @pyjt(place)
bool place() { return async_exec_placed(); }
}
""", jt.compiler.cc_flags)
os.sched_setaffinity(0, set(pair))
before = set(Path('/proc/self/task').iterdir())
assert probe.place(), 'two same-domain CPUs must permit a worker'
created = set(Path('/proc/self/task').iterdir()) - before
assert len(created) == 1, created
tid = int(created.pop().name)
worker_mask = set(os.sched_getaffinity(tid))
assert len(worker_mask) == 1 and worker_mask <= set(pair), worker_mask

# The caller moves onto the worker's CPU and now permits no other CPU.
# A cached L3-domain identity alone incorrectly keeps admitting the worker.
os.sched_setaffinity(0, worker_mask)
assert not probe.place(), 'a singleton caller mask cannot place a second core'
os.sched_setaffinity(0, set(pair))
assert probe.place(), 'restoring a second CPU must permit the worker again'
assert set(os.sched_getaffinity(tid)) <= set(pair)
os.sched_setaffinity(0, set(allowed))
print('PLACEMENT_OK')
'''


@unittest.skipUnless(sys.platform.startswith('linux') and hasattr(os, 'sched_setaffinity'),
                     'environment prerequisite: Linux CPU affinity')
class TestAsyncWorkerAffinity(unittest.TestCase):
    def test_admission_rechecks_a_changed_mask_within_one_l3_domain(self):
        result = run_python_child(['-c', _PROBE], without_torch_mode=True,
                                  merge_stderr=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout)
        if 'PLACEMENT_SKIP' in result.stdout:
            self.skipTest('environment prerequisite: two allowed CPUs sharing L3')
        self.assertIn('PLACEMENT_OK', result.stdout)
