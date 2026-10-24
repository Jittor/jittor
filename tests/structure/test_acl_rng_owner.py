"""Host source contracts for ACL aliases to the native runtime RNG owner.

This does not compile a retired header as a substitute owner, import Jittor,
or claim CANN numerical/device validation. State format is JITTOR_RNG_1;
older unmerged local JITTOR_ACL_RNG_V1 bytes are not cross-runtime compatible.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


def body(source, signature):
    return source.split(signature + " {", 1)[1].split("\n}", 1)[0]


class TestAclRngOwner(unittest.TestCase):
    def test_compatibility_spellings_delegate_to_one_native_runtime_owner(self):
        source = (ROOT / "backends/acl/src/backend.cc").read_text()
        self.assertIn('#include "runtime/rng_state.h"', source)
        delegates = {
            'string acl_get_rng_state(int device)': 'return rng_state("acl", device);',
            'void acl_set_rng_state(int device, const string& blob)': 'set_rng_state("acl", blob, device);',
            'vector<string> acl_get_rng_state_all()': 'return rng_states("acl");',
            'void acl_set_rng_state_all(const vector<string>& blobs)': 'set_rng_states("acl", blobs);',
            'void acl_manual_seed(int device, int64 seed)': 'rng_manual_seed("acl", static_cast<uint64>(seed), device);',
            'void acl_manual_seed_all(int64 seed)': 'rng_manual_seed_all("acl", static_cast<uint64>(seed));',
            'int64 acl_initial_seed(int device)': 'rng_initial_seed("acl", device)',
        }
        for signature, delegate in delegates.items():
            with self.subTest(signature=signature):
                actual = body(source, signature)
                self.assertIn(delegate, actual)
                for forbidden in ('sync_all(', 'ExecutorEntryScope', 'aclrt', 'rng_streams()', 'runtime_rng_state()'):
                    self.assertNotIn(forbidden, actual)
        for forbidden in ('AclRngStreams', 'reset_acl_rng_from_native_seed', 'initialize_acl_rng',
                          'acl_rng_prepare', 'acl_rng_advance', 'add_set_seed_callback'):
            self.assertNotIn(forbidden, source)

    def test_random_dropout_multinomial_all_reserve_from_same_core_owner(self):
        source = (ROOT / "backends/acl/src/acl_op_exec.cc").read_text()
        launch = source.split('static void exec_acl_random(Op *op)', 1)[1].split('static unordered_map', 1)[0]
        self.assertIn('reserve_acl_random(out->device_id, out->numel())', launch)
        self.assertLess(launch.index('reserve_acl_random'), launch.index('runner.run()'))
        for forbidden in ('current_seed', 'current_offset', 'acl_rng_prepare', 'acl_rng_advance', 'initialize_acl_rng'):
            self.assertNotIn(forbidden, source)
        for relative in ('backends/acl/kernels/native/dropout_op_acl.cc',
                         'backends/acl/kernels/native/multinomial_op_acl.cc'):
            self.assertIn('reserve_acl_random(', (ROOT / relative).read_text())

    def test_core_owner_protects_async_reservation_and_atomic_restore(self):
        source = (ROOT / 'src/runtime/rng_state.cc').read_text()
        reservation = body(source, 'AclRngSpan reserve_acl_random(int device, int64 elements)')
        self.assertIn('!inside_executor() && !on_async_worker()', reservation)
        self.assertLess(reservation.index('async_exec_wait()'), reservation.index('ExecutorEntryScope lock'))
        self.assertIn('runtime_rng_state()', source)
        self.assertIn('state.offset += elements', reservation)
        restore = body(source, 'void set_rng_states(const string& backend, const vector<string>& states)')
        self.assertLess(restore.index('states.size() !='), restore.index('sync_all(true)'))
        self.assertLess(restore.index('parse_state('), restore.index('sync_all(true)'))
        self.assertLess(restore.index('sync_all(true)'), restore.index('apply_state('))
        self.assertIn('JITTOR_RNG_1', source)
        self.assertNotIn('JITTOR_ACL_RNG_V1', source)

    def test_legacy_signed_bindings_preserve_validation_and_seed_bits(self):
        source = (ROOT / 'backends/acl/src/backend.cc').read_text()
        for signature in ('void acl_manual_seed(int device, int64 seed)', 'void acl_manual_seed_all(int64 seed)'):
            actual = body(source, signature)
            self.assertLess(actual.index('seed < 0'), actual.index('rng_manual_seed'))
            self.assertIn('std::invalid_argument', actual)
        initial = body(source, 'int64 acl_initial_seed(int device)')
        self.assertIn('static_assert(sizeof(result) == sizeof(seed)', initial)
        self.assertIn('std::memcpy(&result, &seed, sizeof(result))', initial)
        self.assertIn('return result;', initial)


if __name__ == '__main__':
    unittest.main()
