"""Native history/explicit-stop regression, not a Torch-oracle contract.

One native interpreter/device; CPU/CUDA selected by JITTOR_TEST_DEVICES.
No framework is imported by source/discovery checks. Native CPU before/after
runs must retain every RED. No metadata override or second AD engine.
"""
import os
import sys
import unittest


class TestNativePermanentStopHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global np, jt
        import numpy as np
        import jittor as jt

    def setUp(self):
        self.device = os.environ.get('JITTOR_TEST_DEVICES', 'cpu')
        self.assertTrue(self.device in ('cpu', 'cuda'), 'select one CPU/CUDA backend')
        self.assertTrue('torch' not in sys.modules, 'native process must not load Torch')
        self.assertTrue(jt.autograd.get_policy() is jt.autograd.NATIVE,
                        'native policy required; do not silently change a frontend policy')
        if self.device == 'cuda':
            # This is an explicitly selected backend, not a reason to skip a RED.
            self.assertTrue(bool(jt.compiler.has_cuda), 'selected CUDA compiler unavailable')
        scope = jt.runtime.scope(use_cuda=int(self.device == 'cuda'),
            device_id=0 if self.device == 'cuda' else -1, backend_fallback='error')
        scope.__enter__(); self.addCleanup(scope.__exit__, None, None, None)
        self.assertTrue(jt.core.backend_fallback_count() == 0, 'preexisting fallback')
        self.addCleanup(lambda: self.assertTrue(jt.core.backend_fallback_count() == 0, 'fallback occurred'))

    @staticmethod
    def array(values):
        return jt.array(np.asarray(values, dtype=np.float32))

    @staticmethod
    def history(value):
        # All four spellings are the native query contract. This tuple retains
        # scalar metadata only, NOT a Var/Op or a synthetic producer reference.
        return (value.is_backward_leaf, int(value.grad_fn_node_id),
                int(value.grad_fn_op_id), value.grad_fn_name)

    def leaf(self, value, label):
        h = self.history(value)
        self.assertTrue(h[0] is True and h[1] == -1 and h[2] == -1 and h[3] == '', label)

    def nonleaf_record(self, h, label, identity=None):
        self.assertTrue(h[0] is False and h[1] >= 0 and h[2] > 0
                        and type(h[3]) is str and len(h[3]) > 0, label)
        if identity is not None:
            self.assertTrue(h[1] == identity, label + ': producer identity')

    def resident(self, value):
        value.sync()
        self.assertTrue(jt.core.dispatch_context([value]) ==
            (self.device, 0 if self.device == 'cuda' else -1), 'actual dispatch')
        self.assertTrue(value.location() == ('device' if self.device == 'cuda' else 'cpu'), 'actual residency')

    def full(self, value, expected, label):
        # Deliberately exercises the snapshot boundary that caused the real
        # late free. Never migrate the original held tensor to read its values.
        self.resident(value)
        with jt.no_grad():
            clone = value.clone()
            self.assertTrue(clone is not value, 'snapshot owns independent holder')
            host = clone.cpu()
            actual = np.array(host.numpy(), copy=True, order='C')
        self.resident(value)
        expected = np.asarray(expected, dtype=np.float32)
        self.assertTrue(actual.dtype == expected.dtype, label + ': dtype')
        self.assertTrue(actual.shape == expected.shape, label + ': full shape')
        self.assertTrue(bool(np.isfinite(actual).all() and np.isfinite(expected).all()), label + ': finite')
        self.assertTrue(bool(np.array_equal(actual, expected)), label + ': all exact binary values')

    def released_case(self, held_hidden, materialize, prequery):
        x = self.array([-.5, .25, 1.5])
        if held_hidden:
            hidden = x * 2
            y = hidden + 1
        else:
            y = x * 2 + 1
        before = self.history(y) if prequery else None
        if prequery:
            self.nonleaf_record(before, 'before execution')
        if materialize:
            jt.sync_all()
        gradient, = jt.grad(y.sum(), [x], retain_graph=False)
        after_backward = self.history(y)
        self.full(x, [-.5, .25, 1.5], 'original x')
        after_x_snapshot = self.history(y)
        self.full(y, [0., 1.5, 4.], 'original y')
        after_y_snapshot = self.history(y)
        self.full(gradient, [2., 2., 2.], 'real native VJP')
        after_grad_snapshot = self.history(y)
        # Values are checked before reporting history REDs, and all four
        # observation boundaries are retained separately in native metadata.
        identity = before[1] if prequery else after_backward[1]
        for stage, record in (('after_backward', after_backward),
                              ('after_x_snapshot', after_x_snapshot),
                              ('after_y_snapshot', after_y_snapshot),
                              ('after_gradient_snapshot', after_grad_snapshot)):
            with self.subTest(boundary=stage):
                self.nonleaf_record(record, 'released execution preserves history: ' + stage, identity)
        if held_hidden:
            self.assertTrue(hidden.requires_grad is True, 'held hidden remains trainable')

    def test_no_hidden_execution_and_snapshot_do_not_erase_history(self):
        # Eight independent call scopes. The false branch never binds H and
        # noprequery never reads y history/values/placement before real AD.
        for held_hidden in (False, True):
            for materialize in (False, True):
                for prequery in (False, True):
                    with self.subTest(held_hidden=held_hidden, materialize=materialize, prequery=prequery):
                        self.released_case(held_hidden, materialize, prequery)

    def test_held_hidden_explicit_stop_invalidates_and_cannot_revive(self):
        x = self.array([-.5, .25, 1.5]); hidden = x * 2; y = hidden + 1
        self.nonleaf_record(self.history(y), 'recorded hidden edge')
        jt.sync_all()
        self.nonleaf_record(self.history(y), 'held hidden edge after sync')
        self.assertTrue(hidden.stop_grad() is hidden, 'explicit stop original holder')
        self.assertTrue(hidden.is_stop_grad() is True, 'hidden permanently stopped')
        self.leaf(y, 'stopped sole differentiable input makes output native leaf')
        self.full(x, [-.5, .25, 1.5], 'hidden-stop x')
        self.full(hidden, [-1., .5, 3.], 'hidden-stop hidden')
        self.full(y, [0., 1.5, 4.], 'hidden-stop y')
        self.leaf(y, 'snapshot cannot revive explicit stopped edge')
        hidden.stop_grad()  # Already-stopped public call must remain invalid.
        jt.sync_all()
        self.leaf(y, 'repeated explicit stop cannot revive history')

    def test_mixed_inputs_stop_one_then_all(self):
        x = self.array([-.5, .25, 1.5]); hidden = x * 2
        z = self.array([.125, .5, -.75]); y = hidden + z
        identity = self.history(y)[1]
        self.nonleaf_record(self.history(y), 'mixed live inputs', identity)
        jt.sync_all(); hidden.stop_grad()
        self.nonleaf_record(self.history(y), 'one remaining live input', identity)
        self.full(y, [-.875, 1., 2.25], 'mixed one-stop values')
        self.nonleaf_record(self.history(y), 'snapshot retains remaining live input', identity)
        z.stop_grad()
        self.leaf(y, 'all actual inputs permanently stopped')
        self.full(hidden, [-1., .5, 3.], 'mixed stopped hidden')
        self.full(z, [.125, .5, -.75], 'mixed stopped z')
        self.full(y, [-.875, 1., 2.25], 'mixed all-stop values')
        self.leaf(y, 'full snapshots cannot revive stopped inputs')

    def test_output_stop_then_start_is_a_new_native_leaf(self):
        x = self.array([-.5, .25, 1.5]); y = x * 2 + 1
        self.nonleaf_record(self.history(y), 'output original history')
        y.stop_grad(); self.leaf(y, 'explicit output stop')
        self.assertTrue(y.start_grad() is y, 'start original public holder')
        self.assertTrue(y.requires_grad is True, 'new output leaf trainable')
        self.leaf(y, 'start_grad does not reconnect old producer')
        self.full(y, [0., 1.5, 4.], 'new leaf values')
        self.leaf(y, 'snapshot cannot reconnect old producer')
        gradient, = jt.grad((y * 3).sum(), [y], retain_graph=False)
        self.full(gradient, [3., 3., 3.], 'new native leaf VJP')
        self.leaf(y, 'new leaf remains a leaf after AD')

    def test_assign_to_new_leaf_does_not_keep_replaced_history(self):
        x = self.array([-.5, .25, 1.5]); y = x * 2 + 1
        self.nonleaf_record(self.history(y), 'pre-assignment history')
        replacement = self.array([2., 4., 8.]); self.leaf(replacement, 'fresh replacement leaf')
        self.assertTrue(y.assign(replacement) is y, 'assign original holder')
        self.leaf(y, 'new native assignment has no old producer')
        self.full(y, [2., 4., 8.], 'replacement values')
        self.full(x, [-.5, .25, 1.5], 'assignment original x unchanged')
        self.leaf(y, 'snapshot cannot revive replaced producer')

    def test_native_output_reversible_mask_restores_recorded_producer(self):
        x = self.array([-.5, .25, 1.5]); hidden = x * 2; y = hidden + 1
        identity = self.history(y)[1]
        self.nonleaf_record(self.history(y), 'before reversible mask', identity)
        y.requires_grad = False
        self.leaf(y, 'output reversible disabled mask')
        self.assertTrue(y.is_stop_grad() is False, 'reversible mask is not permanent stop')
        self.full(y, [0., 1.5, 4.], 'masked output values')
        self.leaf(y, 'snapshot must not bypass disabled mask')
        y.requires_grad = True
        self.nonleaf_record(self.history(y), 'restored original producer', identity)
        gradient, = jt.grad(y.sum(), [x], retain_graph=False)
        self.full(gradient, [2., 2., 2.], 'restored recorded native VJP')
        self.full(y, [0., 1.5, 4.], 'restored output values')
        self.nonleaf_record(self.history(y), 'restored output after snapshots', identity)
        self.assertTrue(hidden.requires_grad is True, 'retained hidden control remains live')


if __name__ == '__main__':
    unittest.main()
