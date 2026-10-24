"""Complete CPU default RNG continuation; run separately under each runtime.

Random values are intentionally compared within each implementation, never
between distinct RNG algorithms. No accelerator state support is claimed.
"""
import contextlib
import random
import unittest

import numpy as np
import torch


CANDIDATE = hasattr(torch, "_torch_compat_install_context")
if CANDIDATE:
    import jittor as jt


class TestCpuRngState(unittest.TestCase):
    def setUp(self):
        self.scope = contextlib.ExitStack()
        if CANDIDATE:
            from jittor._runtime.fallback import forbid_backend_fallbacks
            self.scope.enter_context(jt.runtime.scope(use_cuda=0))
            self.scope.enter_context(forbid_backend_fallbacks())
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.saved = torch.get_rng_state()

    def tearDown(self):
        torch.set_rng_state(self.saved)
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        self.scope.__exit__(None, None, None)

    def draw(self):
        values = [torch.rand(7, device="cpu"), torch.randn(9, device="cpu"),
                  torch.rand(3, device="cpu"), torch.randn(5, device="cpu")]
        result = []
        for value in values:
            self.assertEqual(value.device.type, "cpu")
            array = value.numpy().copy()
            if CANDIDATE:
                self.assertEqual(getattr(value, "_var", value).location(), "cpu")
            result.append(array)
        return result

    def assertDrawsEqual(self, left, right):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            np.testing.assert_array_equal(a, b)

    def test_advanced_mixed_stream_roundtrip(self):
        torch.manual_seed(1701)
        fresh = torch.get_rng_state().numpy().copy()
        self.draw()
        state = torch.get_rng_state()
        self.assertEqual(state.dtype, torch.uint8)
        self.assertEqual(state.ndim, 1)
        self.assertEqual(state.device.type, "cpu")
        self.assertFalse(np.array_equal(fresh, state.numpy()))
        expected = self.draw()
        self.draw()
        torch.set_rng_state(state)
        self.assertDrawsEqual(self.draw(), expected)

    def test_restores_cpu_seed_metadata_without_reseeding_python_or_numpy(self):
        torch.manual_seed(71)
        self.draw()
        state = torch.get_rng_state()
        torch.manual_seed(92)
        py_before, np_before = random.getstate(), np.random.get_state()
        torch.set_rng_state(state)
        self.assertEqual(torch.initial_seed(), 71)
        self.assertEqual(random.getstate(), py_before)
        np_after = np.random.get_state()
        self.assertEqual(np_after[0], np_before[0])
        np.testing.assert_array_equal(np_after[1], np_before[1])
        self.assertEqual(np_after[2:], np_before[2:])
        if CANDIDATE:
            # Upstream get_seed() now reports the shared host stream's seed,
            # not the independently stored default seed for ACL streams.
            self.assertEqual(jt.get_seed(), torch.initial_seed())

    @unittest.skipUnless(CANDIDATE, "candidate rank-branch contract, not a real MPI test")
    def test_manual_seed_is_rank_independent_and_snapshot_restores_exact_seed(self):
        # Exercise the real CPU RNG with native rank branches. Changing this
        # attribute is not a process-group or actual multi-rank execution test.
        original_rank = jt.rank
        baseline = None
        try:
            for rank in (0, 1, 3):
                with self.subTest(rank=rank):
                    jt.rank = rank
                    torch.manual_seed(71)
                    self.assertEqual(torch.initial_seed(), 71)
                    first = self.draw()
                    if baseline is None:
                        baseline = first
                    else:
                        self.assertDrawsEqual(first, baseline)
                    state = torch.get_rng_state()
                    expected = self.draw()
                    torch.manual_seed(92)
                    self.draw()
                    torch.set_rng_state(state)
                    self.assertEqual(torch.initial_seed(), 71)
                    self.assertDrawsEqual(self.draw(), expected)
        finally:
            jt.rank = original_rank

    def test_invalid_states_do_not_change_stream(self):
        self.draw()
        before = torch.get_rng_state()
        for state in (torch.tensor([71], dtype=torch.int64, device="cpu"),
                      torch.tensor([], dtype=torch.uint8, device="cpu"),
                      torch.tensor(list(b"foreign state"), dtype=torch.uint8, device="cpu")):
            with self.subTest(state=state.shape):
                with self.assertRaises((TypeError, ValueError, RuntimeError)):
                    torch.set_rng_state(state)
                np.testing.assert_array_equal(torch.get_rng_state().numpy(), before.numpy())

    def test_fork_after_advancement_and_exception(self):
        self.draw()
        state = torch.get_rng_state()
        expected = self.draw()
        torch.set_rng_state(state)
        with self.assertRaisesRegex(RuntimeError, "test sentinel"):
            with torch.random.fork_rng(devices=[]):
                self.draw()
                raise RuntimeError("test sentinel")
        self.assertDrawsEqual(self.draw(), expected)

    def test_pending_draw_is_finished_before_snapshot(self):
        # Candidate random work is lazy; the snapshot must include this draw.
        pending = torch.randn(13, device="cpu")
        state = torch.get_rng_state()
        pending.numpy()
        expected = self.draw()
        torch.set_rng_state(state)
        self.assertDrawsEqual(self.draw(), expected)

    def test_restore_finishes_pending_draw_before_replacing_engine(self):
        state = torch.get_rng_state()
        expected_pending = torch.randn(13, device="cpu").numpy().copy()
        torch.set_rng_state(state)
        expected_next = self.draw()
        torch.set_rng_state(state)
        pending = torch.randn(13, device="cpu")
        torch.set_rng_state(state)
        np.testing.assert_array_equal(pending.numpy(), expected_pending)
        self.assertDrawsEqual(self.draw(), expected_next)

    @unittest.skipUnless(CANDIDATE, "native Jittor CPU state format contract")
    def test_native_owner_and_malformed_blob_rejection(self):
        self.draw()
        state = jt.core.get_cpu_rng_state()
        # Retained native legacy format, not Torch's canonical state owner.
        public_before = torch.get_rng_state().numpy().copy()
        header, seed, engine = state.rsplit("\n", 2)
        invalid = ["71", state.replace("JITTOR_CPU_RNG_V1", "JITTOR_CPU_RNG_V2"),
                   state + " ", state + "\n12", header + "\ninvalid\n" + engine,
                   header + "\n" + seed + "\n0",
                   header + "\n" + seed + "\n18446744073709551615"]
        for ordinal, bad in enumerate(invalid):
            with self.subTest(legacy_case=ordinal):
                with self.assertRaises((ValueError, RuntimeError)):
                    jt.core.set_cpu_rng_state(bad)
                self.assertTrue(jt.core.get_cpu_rng_state() == state,
                                "legacy invalid restore changed its owner")
                self.assertTrue(torch.get_rng_state().numpy().tobytes() == public_before.tobytes(),
                                "legacy invalid restore changed canonical state")
        expected = self.draw()
        jt.core.set_cpu_rng_state(state)
        self.assertDrawsEqual(self.draw(), expected)

    @unittest.skipUnless(CANDIDATE, "canonical native CPU state format contract")
    def test_canonical_owner_roundtrip_and_invalid_states_are_atomic(self):
        self.draw()
        state = jt.core.rng_state("cpu")
        public_before = torch.get_rng_state().numpy().copy()
        self.assertTrue(public_before.tobytes() == state.encode("ascii"),
                        "Torch state must use the canonical CPU owner")
        fields = state.split()
        self.assertTrue(len(fields) == 8, "canonical CPU state field count")
        self.assertTrue(fields[:3] == ["JITTOR_RNG_1", "cpu", "0"],
                        "canonical CPU header/backend/device")
        self.assertTrue(fields[4] == "0", "CPU state has no counter offset")
        low, high = int(fields[5]), int(fields[6])
        self.assertTrue(low <= int(fields[7]) <= high, "CPU engine state range")
        mutations = [(0, "JITTOR_RNG_2"), (1, "acl"), (2, "1"),
                     (3, "invalid"), (4, "-1"), (4, "1"),
                     (5, str(low + 1)), (6, str(high + 1)),
                     (7, str(low - 1)), (7, str(high + 1))]
        invalid = [b"", b"71", b"foreign state", bytes([255]),
                   " ".join(fields[:-1]).encode("ascii"),
                   (state + " trailing").encode("ascii")]
        for index, value in mutations:
            changed = list(fields)
            changed[index] = value
            invalid.append(" ".join(changed).encode("ascii"))
        try:
            for ordinal, raw in enumerate(invalid):
                with self.subTest(canonical_case=ordinal):
                    tensor = torch.tensor(list(raw), dtype=torch.uint8, device="cpu")
                    try:
                        with self.assertRaises((TypeError, ValueError, RuntimeError)):
                            torch.set_rng_state(tensor)
                        self.assertTrue(jt.core.rng_state("cpu") == state,
                                        "rejected state changed canonical owner")
                        self.assertTrue(torch.get_rng_state().numpy().tobytes() == public_before.tobytes(),
                                        "rejected state changed public bytes")
                    finally:
                        # Each malformed input is an independent control even
                        # if a regression wrongly accepts a previous state.
                        jt.core.set_rng_state("cpu", state)
        finally:
            # Do not let a failed negative control contaminate other tests.
            jt.core.set_rng_state("cpu", state)
        expected = self.draw()
        # The current canonical parser explicitly consumes trailing whitespace;
        # treat it as equivalent syntax, not the old legacy strict format.
        padded = torch.tensor(list((state + " \n\t").encode("ascii")),
                              dtype=torch.uint8, device="cpu")
        torch.set_rng_state(padded)
        self.assertTrue(jt.core.rng_state("cpu") == state,
                        "equivalent whitespace must restore canonical state")
        self.assertDrawsEqual(self.draw(), expected)


if __name__ == "__main__":
    unittest.main()
