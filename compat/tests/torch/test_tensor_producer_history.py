"""Public producer history survives graph execution/free.

One interpreter/device. Set JITTOR_HISTORY_RUNTIME=oracle for binary Torch;
candidate explicitly imports Jittor before Torch. Independent CPU Torch has
confirmed the complete fixed public fixture. CUDA requires separate real-device
verification; CPU or HOST/source inspection is not CUDA evidence.

32 linear scenarios = 2 dtypes x hidden-holder x materialize x prequery x
retain_graph. Extra negative controls are separately labelled, not L2 proof.
No model/config/network/files/arrays in failure diagnostics. No optimizer.
"""
import os
import unittest


class TestTensorProducerHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Lazy runtime imports permit stdlib-only source/discovery controls.
        global np, torch, jt
        import numpy as np
        runtime = os.environ.get('JITTOR_HISTORY_RUNTIME', 'candidate')
        if runtime not in ('oracle', 'candidate'):
            raise ValueError('select exactly oracle or candidate runtime')
        cls.oracle = runtime == 'oracle'
        jt = None
        if not cls.oracle:
            import jittor as jt
        import torch

    def setUp(self):
        self.device = os.environ.get('JITTOR_TEST_DEVICES', 'cpu')
        self.assertTrue(self.device in ('cpu', 'cuda'), 'select one CPU/CUDA device')
        if self.oracle:
            self.assertFalse(hasattr(torch, '_torch_compat_install_context'), 'independent Torch required')
            if self.device == 'cuda' and not torch.cuda.is_available():
                self.skipTest('accelerator prerequisite: independent PyTorch CUDA unavailable')
        else:
            if self.device == 'cuda' and not jt.compiler.has_cuda:
                self.skipTest('accelerator prerequisite: candidate CUDA compiler unavailable')
            scope = jt.runtime.scope(use_cuda=int(self.device == 'cuda'),
                device_id=0 if self.device == 'cuda' else -1, backend_fallback='error')
            scope.__enter__(); self.addCleanup(scope.__exit__, None, None, None)
            self.assertTrue(jt.core.backend_fallback_count() == 0, 'preexisting fallback')
            self.addCleanup(lambda: self.assertTrue(jt.core.backend_fallback_count() == 0, 'fallback occurred'))
        grad_scope = torch.enable_grad()
        grad_scope.__enter__(); self.addCleanup(grad_scope.__exit__, None, None, None)

    def tensor(self, dtype):
        return torch.tensor([-.5, .25, 1.5], dtype=dtype,
                            device=self.device, requires_grad=True)

    def materialize(self):
        if not self.oracle:
            jt.sync_all()  # Reproduce the real failing public graph sweep.
        elif self.device == 'cuda':
            torch.cuda.synchronize(0)  # Native operations were already eager.

    def resident(self, value):
        self.assertTrue(value.device.type == self.device, 'public device changed')
        if self.device == 'cuda':
            self.assertTrue(value.device.index == 0, 'logical CUDA0 required')
        if not self.oracle:
            native = getattr(value, '_var', value)
            native.sync()
            self.assertTrue(jt.core.dispatch_context([native]) ==
                (self.device, 0 if self.device == 'cuda' else -1), 'actual dispatch changed')
            self.assertTrue(native.location() == ('device' if self.device == 'cuda' else 'cpu'),
                            'actual residency changed')

    def host(self, value):
        # Used only AFTER the relevant backward and first metadata observation.
        self.resident(value)
        with torch.no_grad():
            clone = value.clone()
            self.assertTrue(clone is not value, 'snapshot must own a clone')
            result = np.array(clone.cpu().numpy(), copy=True, order='C')
        self.resident(value)
        return result

    def full(self, value, expected, dtype, label):
        actual = self.host(value)
        expected = np.asarray(expected, dtype='float32' if dtype == torch.float32 else 'float64')
        # Boolean assertions only: no ndarray/bytes diagnostics on failure.
        self.assertTrue(actual.dtype == expected.dtype, label + ': dtype')
        self.assertTrue(actual.shape == expected.shape, label + ': full shape')
        self.assertTrue(bool(np.isfinite(actual).all() and np.isfinite(expected).all()), label + ': finite')
        self.assertTrue(bool(np.array_equal(actual, expected)), label + ': all exact binary fixture values')

    @staticmethod
    def history(value):
        return (value.requires_grad, value.is_leaf, value.grad_fn)

    def nonleaf(self, history, label):
        requires, leaf, producer = history
        self.assertTrue(requires is True, label + ': requires_grad')
        self.assertTrue(leaf is False, label + ': is_leaf')
        self.assertTrue(producer is not None, label + ': grad_fn present')

    def stopped_leaf(self, value, label):
        requires, leaf, producer = self.history(value)
        self.assertTrue(requires is False, label + ': requires_grad')
        self.assertTrue(leaf is True, label + ': is_leaf')
        self.assertTrue(producer is None, label + ': grad_fn absent')

    def linear_case(self, dtype, hidden_holder, materialize, prequery, retain_graph):
        # Each case is a separate Python call: no previous case's hidden Var
        # can accidentally keep this case's x*2 intermediate alive.
        x = self.tensor(dtype)
        if hidden_holder:
            hidden = x * 2
            y = hidden + 1
        else:
            y = x * 2 + 1
        held = self.history(y) if prequery else None
        if prequery:
            self.nonleaf(held, 'linear before')
        # In the noprequery branch, NO y.is_leaf/grad_fn/history, host read,
        # resident(sync), or clone occurs before the actual backward.
        if materialize:
            self.materialize()
        y.sum().backward(retain_graph=retain_graph)
        after = self.history(y)  # Observe BEFORE any snapshot can affect graph lifetime.
        self.assertTrue(x.grad is not None, 'linear x gradient exists')
        self.full(x, [-.5, .25, 1.5], dtype, 'linear input unchanged')
        self.full(y, [0., 1.5, 4.], dtype, 'linear y')
        self.full(x.grad, [2., 2., 2.], dtype, 'linear x gradient')
        self.nonleaf(after, 'linear after backward')
        self.nonleaf(self.history(y), 'linear after snapshots')
        if prequery:
            # The real Python grad_fn wrapper is strongly retained; do not
            # require implementation class-name equality across frameworks.
            self.assertTrue(after[2] is held[2], 'held producer identity after backward')
            self.assertTrue(y.grad_fn is held[2], 'held producer identity after snapshots')
        if hidden_holder:
            self.nonleaf(self.history(hidden), 'retained hidden history')

    def test_linear_history_full_matrix(self):
        for dtype in (torch.float32, torch.float64):
            for hidden_holder in (False, True):
                for materialize in (False, True):
                    for prequery in (False, True):
                        for retain_graph in (False, True):
                            with self.subTest(dtype=str(dtype), hidden_holder=hidden_holder,
                                             materialize=materialize, prequery=prequery, retain_graph=retain_graph):
                                self.linear_case(dtype, hidden_holder, materialize, prequery, retain_graph)

    def test_detach_does_not_erase_original_history(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                x = self.tensor(dtype); y = x * 2 + 1; held = y.grad_fn
                detached = y.detach()
                self.assertTrue(detached is not y, 'detach separate object')
                self.stopped_leaf(detached, 'detach')
                y.sum().backward(); after = self.history(y)
                self.full(detached, [0., 1.5, 4.], dtype, 'detached values')
                self.full(x.grad, [2., 2., 2.], dtype, 'detach original gradient')
                self.nonleaf(after, 'detach original history')
                self.assertTrue(after[2] is held, 'detach original held producer')

    def test_explicit_permanent_stop_then_restore_is_new_leaf(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                x = self.tensor(dtype); y = x * 2 + 1; held = y.grad_fn
                # Public shared API counterpart of explicit permanent stop:
                # detach_ clears THIS tensor's history (not merely execution).
                self.assertTrue(y.detach_() is y, 'detach_ original object')
                self.stopped_leaf(y, 'explicit detach_')
                y.requires_grad_(True)
                self.assertTrue(y.requires_grad is True and y.is_leaf is True and y.grad_fn is None,
                                'restored detached tensor is a new leaf')
                y.sum().backward()
                self.assertTrue(x.grad is None, 'permanent stop must not reconnect original x')
                self.full(y.grad, [1., 1., 1.], dtype, 'restored detached leaf gradient')
                self.full(y, [0., 1.5, 4.], dtype, 'stopped tensor values')
                self.assertTrue(held is not None, 'held old producer does not reconnect detached y')

    def test_copy_replaces_producer_not_sticky_old_metadata(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                x = self.tensor(dtype); y = x * 2 + 1; old = y.grad_fn
                rhs = self.tensor(dtype)
                self.assertTrue(y.copy_(rhs) is y, 'copy_ original holder')
                copied = self.history(y); self.nonleaf(copied, 'differentiable copy history')
                self.assertTrue(copied[2] is not old, 'copy_ must replace old producer')
                y.sum().backward(); after = self.history(y)
                self.full(y, [-.5, .25, 1.5], dtype, 'copy values')
                self.assertTrue(x.grad is not None and rhs.grad is not None, 'copy full VJP inventory')
                self.full(x.grad, [0., 0., 0.], dtype, 'copy overwritten x gradient')
                self.full(rhs.grad, [1., 1., 1.], dtype, 'copy RHS gradient')
                self.nonleaf(after, 'copy after backward')
                self.assertTrue(after[2] is copied[2], 'copy current held producer remains')

    def test_reversible_leaf_requires_grad_restore_keeps_saved_graph(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                x = self.tensor(dtype); y = x * x * 2 + 1
                self.assertTrue(x.requires_grad_(False) is x, 'freeze original leaf')
                self.assertTrue(x.requires_grad_(True) is x, 'restore original leaf')
                self.assertTrue(x.is_leaf is True and x.grad_fn is None, 'restored source remains leaf')
                y.sum().backward(); after = self.history(y)
                self.full(x.grad, [-2., 1., 6.], dtype, 'restored source saved-graph gradient')
                self.full(y, [1.5, 1.125, 5.5], dtype, 'restored saved-graph values')
                self.nonleaf(after, 'reversible source restore output')

    def test_saved_tensor_release_error_separate_from_public_history(self):
        for dtype in (torch.float32, torch.float64):
            for retain_graph in (False, True):
                with self.subTest(dtype=str(dtype), retain_graph=retain_graph):
                    x = self.tensor(dtype); y = x * x * 2 + 1; held = y.grad_fn
                    loss = y.sum()
                    loss.backward(retain_graph=retain_graph)
                    after = self.history(y)
                    self.full(x.grad, [-2., 1., 6.], dtype, 'first saved-tensor VJP')
                    self.nonleaf(after, 'saved tensor graph after first backward')
                    self.assertTrue(after[2] is held, 'saved graph held public producer')
                    if retain_graph:
                        loss.backward()  # Consume the retained graph, then release.
                        self.full(x.grad, [-4., 2., 12.], dtype, 'two accumulated saved-tensor VJPs')
                    # Do NOT require a linear x*2 graph to throw on re-entry:
                    # its derivative may need no saved tensor in binary Torch.
                    with self.assertRaises(RuntimeError):
                        loss.backward()
                    self.nonleaf(self.history(y), 'release error must not falsify output history')
                    self.assertTrue(y.grad_fn is held, 'public producer survives released-error path')


if __name__ == '__main__':
    unittest.main()
