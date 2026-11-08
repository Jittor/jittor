"""A batch that failed must not keep failing the next sync.

The executor marks the outputs of the operator that threw, but the rest of the
batch never ran and used to stay pending. `sync_all` picked those up again, so
one failure was re-raised from an unrelated caller -- in a flag scope's exit
flush it then failed every later test in the process (KI-EXEC-017).
"""

import unittest

import numpy as np

import jittor as jt


class TestFailedBatchCleanup(unittest.TestCase):
    def test_sync_all_does_not_replay_a_failed_batch(self):
        source = jt.array(np.array([1., 2.], dtype="float32"))
        # No source: preparation refuses it, so it throws inside the batch
        # and the second operator in the same batch never runs.
        failing = jt.code(source.shape, source.dtype, [source])
        never_ran = jt.code(source.shape, source.dtype, [source],
                            cpu_src="out0 = in0 + 1;")
        with self.assertRaises(Exception):
            jt.sync([failing, never_ran])

        # Sweeping everything up must not retry the failed batch. This is the
        # call a scope exit and every later test make.
        jt.sync_all()

        fresh = jt.array(np.array([3., 4.], dtype="float32"))
        np.testing.assert_array_equal((fresh + 1).numpy(), [4., 5.])


if __name__ == "__main__":
    unittest.main()
