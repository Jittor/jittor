"""FSDP2 compatibility tests for ``import torch``.

Run:
    python -m pytest compat/tests/torch/test_torch_compat_fsdp2.py
"""
import abc
import textwrap
import unittest
import types
import weakref
from unittest import mock
import numpy as np
import pytest
import torch
import jittor as jt
from _helpers import capability as _test_capability
from _helpers.child_process import run_python_child
from jittor.compat import fsdp2 as canonical_fsdp
from jittor.compat.fsdp2 import grad_sync as fsdp_grad_sync
from jittor.compat.fsdp2 import shard as fsdp_shard
from jittor.compat.torch.installers.distributed import _backend_matches_active
from jittor.compat.torch.tensor_state import get_tensor_state

#: Keep this module on one xdist worker.
#:
#: These cases patch and read module-scope FSDP2 state and depend on running in
#: file order, which is the property ``--dist loadfile`` was chosen to preserve
#: (see agent/skills/gate-tier-budget §4). The smoke tier runs ``loadgroup``
#: instead, to stop one long file pinning a worker, and loadgroup distributes
#: *ungrouped* tests one by one -- so this class was being split across four
#: workers, each seeing a different subset in a different order.
#:
#: Measured: run on its own the file is deterministic (5 failed, 21 passed, in
#: both of two runs). Inside the smoke tier, two warm back-to-back runs of the
#: identical selection disagreed about three of its nodeids, in both
#: directions, and those three were the *only* conclusion differences in the
#: whole torch half. A tier that answers differently each time cannot satisfy
#: 0.15's "one round reports every failure", and it makes
#: ``tools/gate_conclusion_diff.py`` unusable as a criterion, because real lost
#: conclusions arrive mixed with this noise.
#:
#: The group costs nothing: the file is 2.7 s, far below the tier's makespan.
pytestmark = pytest.mark.xdist_group("fsdp2_compat_module_state")


class TestFSDP2Compat(unittest.TestCase):
    def test_parameter_trainability_uses_torch_requires_grad(self):
        parameter = torch.nn.Parameter(torch.ones(4))
        parameter.requires_grad_(False)

        self.assertFalse(parameter.requires_grad)
        self.assertFalse(parameter.is_stop_grad())
        self.assertFalse(fsdp_shard._parameter_requires_grad(parameter))

    def test_initial_shard_materialization_releases_full_parent(self):
        # Switching allocators while earlier tests still own Vars can corrupt
        # their storage during jt.gc(). A fresh process is part of this memory
        # test's contract, not merely test-order isolation.
        code = textwrap.dedent(
            """
            import gc
            import numpy as np
            import jittor as jt
            from jittor.compat.fsdp2 import shard as fsdp_shard

            with jt.flag_scope(
                    use_cuda=0, use_stat_allocator=1, use_sfrl_allocator=0):
                jt.sync_all(True)
                gc.collect()
                baseline = (
                    jt.flags.stat_allocator_total_alloc_byte
                    - jt.flags.stat_allocator_total_free_byte)
                full = jt.array(np.ones(
                    (4 * 1024 * 1024,), dtype=np.float32))
                full.sync()
                shard = fsdp_shard._materialize_initial_shard(
                    full[: full.shape[0] // 4])
                del full
                gc.collect()
                jt.gc()
                jt.sync_all(True)
                live_delta = (
                    jt.flags.stat_allocator_total_alloc_byte
                    - jt.flags.stat_allocator_total_free_byte
                    - baseline)
                assert tuple(shard.shape) == (1024 * 1024,)
                np.testing.assert_array_equal(
                    shard.numpy(), np.ones((1024 * 1024,), dtype="float32"))
                assert live_delta < 8 * 1024 * 1024, live_delta
                print("MATERIALIZATION_OK", live_delta)
            """
        )
        completed = run_python_child(
            ["-c", code], text=True, merge_stderr=True)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("MATERIALIZATION_OK", completed.stdout)

    def test_fsdp_var_methods_do_not_retain_temporary_full_parameter(self):
        code = textwrap.dedent(
            """
            import gc
            import types
            import numpy as np
            import jittor as jt
            import jittor.compat.torch
            from jittor.compat.fsdp2 import shard as fsdp_shard

            with jt.flag_scope(
                    use_cuda=0, use_stat_allocator=1, use_sfrl_allocator=0):
                jt.sync_all(True)
                gc.collect()
                baseline = (
                    jt.flags.stat_allocator_total_alloc_byte
                    - jt.flags.stat_allocator_total_free_byte)
                shard = jt.array(np.ones((1,), dtype=np.float32)).stop_grad()
                owner = types.SimpleNamespace(weight=shard)
                entry = fsdp_shard.common.StateRecord(
                    owner=owner, attr="weight", shard=shard, full_param=None,
                    requires_grad=False)
                state = fsdp_shard.common.StateRecord(
                    true_fsdp_initialized=True, true_fsdp_flat=False,
                    true_fsdp_unsharded=True, true_fsdp_params=(entry,))
                fsdp_shard._mark_fsdp_param_var(
                    shard, state, entry, "shard")
                full = jt.array(np.ones(
                    (4 * 1024 * 1024,), dtype=np.float32))
                full.sync()
                fsdp_shard._mark_fsdp_param_var(full, state, entry, "full")
                entry.full_param = full
                owner.weight = full
                assert full.to_local() is shard
                assert full.full_tensor() is full
                assert type(full.to_local).__name__ == "_ShardTensorMethod"
                assert "_local_tensor" not in getattr(full, "__dict__", {})
                fsdp_shard._reshard_module_params(
                    types.SimpleNamespace(_fsdp_state=state))
                del full
                gc.collect()
                jt.gc()
                jt.sync_all(True)
                live_delta = (
                    jt.flags.stat_allocator_total_alloc_byte
                    - jt.flags.stat_allocator_total_free_byte
                    - baseline)
                assert entry.full_param is None
                assert owner.weight is shard
                assert live_delta < 8 * 1024 * 1024, live_delta
                assert type(shard.to_local).__name__ == "_ShardTensorMethod"
                assert "_local_tensor" not in getattr(shard, "__dict__", {})
                print("FSDP_TEMP_FULL_RELEASE_OK", live_delta)
            """
        )
        completed = run_python_child(
            ["-c", code], text=True, merge_stderr=True)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("FSDP_TEMP_FULL_RELEASE_OK", completed.stdout)

    def test_fsdp_var_metadata_does_not_accumulate_replaced_vars(self):
        # Replacing a full/shard/gradient triple models one FSDP reshard cycle.
        # Keep the state alive while replacing its entries, just as the runtime
        # does, so stale Var metadata is the only possible retention root.
        code = textwrap.dedent(
            """
            import gc
            import types
            import numpy as np
            import jittor as jt
            import jittor.compat.torch
            from jittor.compat.fsdp2 import shard as fsdp_shard

            with jt.flag_scope(
                    use_cuda=0, use_stat_allocator=1, use_sfrl_allocator=0):
                owner = types.SimpleNamespace()
                entry = fsdp_shard.common.StateRecord(
                    owner=owner, attr="weight", shard=None, full_param=None,
                    requires_grad=True)
                state = fsdp_shard.common.StateRecord(
                    true_fsdp_initialized=True, true_fsdp_flat=False,
                    true_fsdp_unsharded=True, true_fsdp_params=(entry,),
                    true_fsdp_module=None)
                live = []
                for _ in range(12):
                    current = jt.array(
                        np.ones((128,), dtype=np.float32)).stop_grad()
                    entry.shard = current
                    owner.weight = current
                    fsdp_shard._mark_fsdp_param_var(
                        current, state, entry, "shard")
                    full = jt.array(
                        np.ones((128,), dtype=np.float32)).stop_grad()
                    entry.full_param = full
                    owner.weight = full
                    fsdp_shard._mark_fsdp_param_var(
                        full, state, entry, "full")
                    gradient = jt.array(
                        np.ones((128,), dtype=np.float32)).stop_grad()
                    fsdp_shard._mark_fsdp_param_var(
                        gradient, state, entry, "grad_shard")
                    entry.full_param = None
                    owner.weight = entry.shard
                    del current, full, gradient
                    gc.collect()
                    jt.gc()
                    jt.sync_all(True)
                    live.append(jt.liveness_info()["lived_vars"])
                assert len(set(live[4:])) == 1, live
                assert "_local_tensor" not in getattr(
                    entry.shard, "__dict__", {})
                print("FSDP_METADATA_REPLACEMENT_OK", live)
            """
        )
        completed = run_python_child(
            ["-c", code], text=True, merge_stderr=True)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("FSDP_METADATA_REPLACEMENT_OK", completed.stdout)

    def test_stale_gradient_method_resolves_only_a_current_gradient(self):
        _, state, entries, _ = self._fake_fsdp_state(([1.0, 2.0],))
        entry = entries[0]
        stale = fsdp_shard._mark_fsdp_param_var(
            jt.ones_like(entry.shard).stop_grad(), state, entry, "grad_shard")
        current = fsdp_shard._mark_fsdp_param_var(
            jt.zeros_like(entry.shard).stop_grad(), state, entry, "grad_shard")
        state.true_fsdp_last_grads = (current,)

        self.assertIs(stale.to_local(), current)
        state.true_fsdp_last_grads = ()
        with self.assertRaisesRegex(ReferenceError, "gradient has been released"):
            stale.to_local()

    def test_reshard_releases_only_frozen_full_parameters(self):
        _, state, entries, full = self._fake_fsdp_state(
            ([1.0, 2.0], [3.0, 4.0]))
        state.true_fsdp_unsharded = True
        entries[0].requires_grad = False
        entries[0].full_param = full[0]
        entries[1].requires_grad = True
        entries[1].full_param = full[1]
        for entry in entries:
            setattr(entry.owner, entry.attr, entry.full_param)

        fsdp_shard._reshard_module_params(
            types.SimpleNamespace(_fsdp_state=state))

        self.assertIs(getattr(entries[0].owner, entries[0].attr), entries[0].shard)
        self.assertIs(getattr(entries[1].owner, entries[1].attr), entries[1].shard)
        self.assertIsNone(entries[0].full_param)
        self.assertIs(entries[1].full_param, full[1])

    def test_flat_reshard_releases_frozen_full_buffer(self):
        _, state, entries, full = self._fake_flat_fsdp_state(
            ([1.0, 2.0], [3.0, 4.0]))
        state.true_fsdp_unsharded = True
        state.true_fsdp_flat_full_param = jt.concat(full)
        for entry, value in zip(entries, full):
            entry.requires_grad = False
            entry.full_param = value
            setattr(entry.owner, entry.attr, value)

        fsdp_shard._reshard_module_params(
            types.SimpleNamespace(_fsdp_state=state))

        self.assertTrue(all(entry.full_param is None for entry in entries))
        self.assertIsNone(state.true_fsdp_flat_full_param)

    def test_frozen_execute_syncs_before_reshard_and_gc(self):
        events = []
        state = types.SimpleNamespace(
            true_fsdp_initialized=True,
            true_fsdp_params=(types.SimpleNamespace(requires_grad=False),),
            reshard_after_forward=True,
        )
        module = types.SimpleNamespace(_fsdp_state=state)

        with mock.patch.object(
                fsdp_shard, "_unshard_module_params",
                side_effect=lambda value: events.append("unshard")), mock.patch.object(
                    fsdp_shard, "_reshard_module_params",
                    side_effect=lambda value: events.append("reshard")), mock.patch.object(
                        fsdp_shard.jt, "submit_pending",
                        side_effect=lambda *values, **kwargs: events.append("submit")), mock.patch.object(
                            fsdp_shard.jt, "gc",
                            side_effect=lambda: events.append("gc")):
            result = fsdp_shard._execute_with_true_fsdp(
                module, lambda: events.append("execute") or jt.ones(1))

        self.assertIsInstance(result, jt.Var)
        self.assertEqual(
            events, ["unshard", "execute", "submit", "reshard", "gc"])

    def test_frozen_execute_materializes_nested_output_without_input_grad(self):
        state = types.SimpleNamespace(
            true_fsdp_initialized=True,
            true_fsdp_params=(types.SimpleNamespace(requires_grad=False),),
            reshard_after_forward=True,
        )
        module = types.SimpleNamespace(_fsdp_state=state)
        inputs = jt.array([1.0, 2.0]).stop_grad()
        original = inputs * 3

        with mock.patch.object(
                fsdp_shard, "_unshard_module_params"), mock.patch.object(
                    fsdp_shard, "_reshard_module_params"), mock.patch.object(
                        fsdp_shard.jt, "gc"):
            result = fsdp_shard._execute_with_true_fsdp(
                module,
                lambda value: {
                    "tensor": original,
                    "nested": (original + 1, [original + 2]),
                },
                inputs,
            )

        np.testing.assert_allclose(result["tensor"].numpy(), [3.0, 6.0])
        np.testing.assert_allclose(result["nested"][0].numpy(), [4.0, 7.0])
        np.testing.assert_allclose(result["nested"][1][0].numpy(), [5.0, 8.0])
        self.assertFalse(result["tensor"].requires_grad)
        self.assertTrue(result["tensor"].is_stop_grad())
        self.assertIsNot(result["tensor"], original)

    def test_frozen_execute_preserves_graph_for_trainable_input(self):
        state = types.SimpleNamespace(
            true_fsdp_initialized=True,
            true_fsdp_params=(types.SimpleNamespace(requires_grad=False),),
            reshard_after_forward=True,
        )
        module = types.SimpleNamespace(_fsdp_state=state)
        inputs = jt.array([1.0, 2.0])
        original = inputs * 3

        with mock.patch.object(
                fsdp_shard, "_unshard_module_params"), mock.patch.object(
                    fsdp_shard, "_reshard_module_params"), mock.patch.object(
                        fsdp_shard.jt, "gc"):
            result = fsdp_shard._execute_with_true_fsdp(
                module, lambda value: original, inputs)

        self.assertIs(result, original)
        self.assertTrue(result.requires_grad)
        self.assertFalse(result.is_stop_grad())

    def test_trainable_execute_does_not_force_forward_sync(self):
        events = []
        state = types.SimpleNamespace(
            true_fsdp_initialized=True,
            true_fsdp_params=(types.SimpleNamespace(requires_grad=True),),
            reshard_after_forward=True,
        )
        module = types.SimpleNamespace(_fsdp_state=state)

        with mock.patch.object(
                fsdp_shard, "_unshard_module_params",
                side_effect=lambda value: events.append("unshard")), mock.patch.object(
                    fsdp_shard, "_reshard_module_params",
                    side_effect=lambda value: events.append("reshard")), mock.patch.object(
                        fsdp_shard.jt, "submit_pending") as submit, mock.patch.object(
                            fsdp_shard.jt, "gc") as collect:
            result = fsdp_shard._execute_with_true_fsdp(
                module, lambda: events.append("execute") or "output")

        self.assertEqual(result, "output")
        self.assertEqual(events, ["unshard", "execute", "reshard"])
        submit.assert_not_called()
        collect.assert_not_called()

    def _fake_fsdp_state(self, values):
        fsdp = canonical_fsdp

        owner = types.SimpleNamespace()
        entries = []
        full_params = []
        state = fsdp._common.StateRecord(
            true_fsdp_initialized=True,
            true_fsdp_flat=False,
            true_fsdp_rank=0,
            true_fsdp_world_size=1,
            true_fsdp_unsharded=False,
            true_fsdp_module=None,
            frontend_type=torch.Tensor,
        )
        for i, value in enumerate(values):
            full = torch.tensor(
                np.asarray(value, dtype="float32"), requires_grad=True)
            shard = torch.tensor(
                np.asarray(value, dtype="float32"), requires_grad=True)
            attr = f"param_{i}"
            entry = fsdp._common.StateRecord(
                name=attr,
                owner=owner,
                attr=attr,
                shape=tuple(shard.shape),
                dtype=shard.dtype,
                numel=int(shard.numel()),
                padded_numel=int(shard.numel()),
                shard_numel=int(shard.numel()),
                shard=shard,
                full_param=full,
                last_grad=None,
                requires_grad=True,
            )
            entries.append(entry)
            full_params.append(full)
            setattr(owner, attr, shard)
        state.true_fsdp_params = entries
        for entry in entries:
            fsdp._mark_fsdp_param_var(entry.shard, state, entry, "shard")
        return fsdp, state, entries, full_params

    def _fake_flat_fsdp_state(self, values):
        fsdp = canonical_fsdp

        owner = types.SimpleNamespace()
        arrays = [np.asarray(value, dtype="float32") for value in values]
        flat = torch.tensor(
            np.concatenate([value.reshape(-1) for value in arrays]),
            requires_grad=True,
        )
        state = fsdp._common.StateRecord(
            true_fsdp_initialized=True,
            true_fsdp_flat=True,
            true_fsdp_rank=0,
            true_fsdp_world_size=1,
            true_fsdp_unsharded=False,
            true_fsdp_module=None,
            frontend_type=torch.Tensor,
            true_fsdp_flat_total_numel=int(flat.numel()),
            true_fsdp_flat_padded_numel=int(flat.numel()),
            true_fsdp_flat_shard_numel=int(flat.numel()),
            true_fsdp_flat_shard=flat,
        )
        entries = []
        full_params = []
        offset = 0
        for i, value in enumerate(arrays):
            full = torch.tensor(value, requires_grad=True)
            attr = f"param_{i}"
            entry = fsdp._common.StateRecord(
                name=attr,
                owner=owner,
                attr=attr,
                shape=tuple(value.shape),
                dtype=full.dtype,
                numel=int(value.size),
                padded_numel=int(value.size),
                shard_numel=int(value.size),
                shard=None,
                full_param=full,
                flat_offset=offset,
                last_grad=None,
                requires_grad=True,
            )
            offset += int(value.size)
            entries.append(entry)
            full_params.append(full)
        state.true_fsdp_params = entries
        fsdp._mark_fsdp_param_var(flat, state, None, "flat_shard")
        fsdp._refresh_flat_entry_shards(state)
        for entry in entries:
            setattr(owner, entry.attr, entry.shard)
        return fsdp, state, entries, full_params

    def test_flat_and_nonflat_grad_sync_execute_real_slicing(self):
        values = ([1.0, 2.0], [3.0, 4.0])
        for factory in (self._fake_flat_fsdp_state, self._fake_fsdp_state):
            _, state, entries, full = factory(values)
            full_grads = [jt.ones_like(value) for value in full]
            with mock.patch.object(
                    canonical_fsdp._common, "_reduce_scatter_padded",
                    side_effect=lambda value, group=None: value):
                sharded = fsdp_grad_sync._sync_sharded_grads_from_full_grads(
                    state, full_grads)
            self.assertEqual(len(sharded), len(entries))
            for entry, grad in zip(entries, sharded):
                self.assertEqual(tuple(grad.shape), tuple(entry.shard.shape))
                self.assertIs(grad.to_local(), grad)
                self.assertEqual(tuple(grad.full_tensor().shape), entry.shape)
                self.assertNotIn("_local_tensor", getattr(grad, "__dict__", {}))
                np.testing.assert_array_equal(
                    grad.numpy(), np.ones(entry.shard.shape, dtype="float32"))

    def test_fsdp_optimizer_skips_unused_and_zero_clears_pending_grad(self):
        fsdp, _, entries, full = self._fake_fsdp_state(
            ([1.0, 2.0], [3.0, 4.0]))
        optimizer = torch.optim.AdamW(
            [entry.shard for entry in entries], lr=0.01, weight_decay=0.2)
        unused_before = entries[1].shard.numpy().copy()

        def local_sync(state, grads, **kwargs):
            return [grad.stop_grad() for grad in grads]

        with mock.patch.object(
                fsdp_grad_sync, "_sync_sharded_grads_from_full_grads",
                side_effect=local_sync):
            fsdp.fill_fsdp_optimizer_grads_from_grad_map(
                [optimizer], {id(full[0]): jt.ones_like(full[0])})

        self.assertIsNotNone(entries[0].shard.grad)
        self.assertIsNone(entries[1].shard.grad)
        optimizer.step()
        np.testing.assert_array_equal(entries[1].shard.numpy(), unused_before)
        self.assertNotIn(entries[1].shard, optimizer.state)
        self.assertIsNotNone(entries[0].shard.grad)

        optimizer.zero_grad(set_to_none=True)
        self.assertIsNone(entries[0].shard.grad)
        self.assertIsNone(entries[0].last_grad)
        used_before_empty = entries[0].shard.numpy().copy()
        optimizer.step()
        np.testing.assert_array_equal(entries[0].shard.numpy(), used_before_empty)
        self.assertEqual(optimizer.n_step, 1)
        jt.sync_all(True)

    def test_shared_fsdp_parameter_accumulates_once_for_two_optimizers(self):
        fsdp, _, entries, full = self._fake_fsdp_state(([1.0, 2.0],))
        first = torch.optim.AdamW([entries[0].shard], lr=0.01)
        second = torch.optim.AdamW([entries[0].shard], lr=0.01)

        def local_sync(state, grads, **kwargs):
            return [grad.stop_grad() for grad in grads]

        with mock.patch.object(
                fsdp_grad_sync, "_sync_sharded_grads_from_full_grads",
                side_effect=local_sync):
            grad = jt.ones_like(full[0])
            fsdp.fill_fsdp_optimizer_grads_from_grad_map(
                [first, second], {id(full[0]): grad})
            published = entries[0].shard.grad
            self.assertIs(first.param_groups[0]["grads"][0], published)
            self.assertIs(second.param_groups[0]["grads"][0], published)
            fsdp.fill_fsdp_optimizer_grads_from_grad_map(
                [first, second], {id(full[0]): grad})

        self.assertIs(entries[0].shard.grad, published)
        np.testing.assert_allclose(
            published.numpy(), np.full(2, 2.0, dtype="float32"),
            atol=0.0, rtol=0.0)
        self.assertIs(first.param_groups[0]["grads"][0], published)
        self.assertIs(second.param_groups[0]["grads"][0], published)
        jt.sync_all(True)

    def test_fsdp_adamw_two_steps_keep_flat_and_nonflat_trainable(self):
        for factory in (self._fake_fsdp_state, self._fake_flat_fsdp_state):
            fsdp, state, entries, full = factory(([1.0, 2.0], [3.0, 4.0]))
            optimizer = torch.optim.AdamW(
                [entry.shard for entry in entries], lr=0.01)

            def local_sync(current_state, grads, **kwargs):
                return [grad.reshape(entry.shard.shape).stop_grad()
                        for entry, grad in zip(current_state.true_fsdp_params, grads)]

            with mock.patch.object(
                    fsdp_grad_sync, "_sync_sharded_grads_from_full_grads",
                    side_effect=local_sync):
                sum((param * param).sum() for param in full).backward()
                optimizer.step()
            self.assertTrue(all(not entry.shard.is_stop_grad()
                                for entry in entries))
            if state.true_fsdp_flat:
                self.assertFalse(state.true_fsdp_flat_shard.is_stop_grad())

            optimizer.zero_grad(set_to_none=True)
            before_second = [entry.shard.numpy().copy() for entry in entries]
            for entry in entries:
                entry.full_param = entry.shard.reshape(entry.shape) * 1.0
            with mock.patch.object(
                    fsdp_grad_sync, "_sync_sharded_grads_from_full_grads",
                    side_effect=local_sync):
                sum((entry.full_param * entry.full_param).sum()
                    for entry in entries).backward()
                optimizer.step()
            self.assertEqual(optimizer.n_step, 2)
            self.assertTrue(all(not entry.shard.is_stop_grad()
                                for entry in entries))
            for before, entry in zip(before_second, entries):
                self.assertGreater(
                    float(np.abs(entry.shard.numpy() - before).max()), 0.0)
            jt.sync_all(True)

    def test_flat_fsdp_optimizer_materializes_before_refresh(self):
        _, state, entries, full = self._fake_flat_fsdp_state(
            ([1.0, 2.0], [3.0, 4.0]))
        optimizer = torch.optim.AdamW(
            [entry.shard for entry in entries], lr=0.01)

        def local_sync(current_state, grads, **kwargs):
            return [grad.reshape(entry.shard.shape).stop_grad()
                    for entry, grad in zip(current_state.true_fsdp_params, grads)]

        with mock.patch.object(
                fsdp_grad_sync, "_sync_sharded_grads_from_full_grads",
                side_effect=local_sync):
            canonical_fsdp.fill_fsdp_optimizer_grads_from_grad_map(
                [optimizer], {
                    id(param): jt.ones_like(param) for param in full
                })

        locations = []
        original_refresh = fsdp_shard._refresh_flat_entry_shards

        def checked_refresh(current_state):
            locations.append(current_state.true_fsdp_flat_shard.location())
            return original_refresh(current_state)

        with mock.patch.object(
                fsdp_shard, "_refresh_flat_entry_shards",
                side_effect=checked_refresh):
            canonical_fsdp.optimizer_step(optimizer)

        self.assertEqual(len(locations), 1)
        self.assertNotEqual(locations[0], "none")
        jt.sync_all(True)

    def test_sharded_sgd_helper_keeps_parameters_trainable(self):
        for factory in (self._fake_fsdp_state, self._fake_flat_fsdp_state):
            fsdp, state, entries, _ = factory(([1.0, 2.0], [3.0, 4.0]))
            module = types.SimpleNamespace(_fsdp_state=state)

            def fake_sync(*args, **kwargs):
                if state.true_fsdp_flat:
                    state.true_fsdp_last_flat_grad = jt.ones_like(
                        state.true_fsdp_flat_shard).stop_grad()
                return [jt.ones_like(entry.shard).stop_grad()
                        for entry in entries]

            for _ in range(2):
                before = [entry.shard.numpy().copy() for entry in entries]
                with mock.patch.object(
                        fsdp_grad_sync, "sync_sharded_grads", side_effect=fake_sync):
                    fsdp.sharded_sgd_step(module, jt.array(0.0), lr=0.1)
                self.assertTrue(all(not entry.shard.is_stop_grad()
                                    for entry in entries))
                for old, entry in zip(before, entries):
                    self.assertGreater(
                        float(np.abs(entry.shard.numpy() - old).max()), 0.0)
            jt.sync_all(True)

    def test_fsdp_sgd_momentum_state_is_serialized(self):
        fsdp, _, entries, full = self._fake_fsdp_state(([1.0, 2.0],))
        optimizer = torch.optim.SGD(
            [entries[0].shard], lr=0.1, momentum=0.9)

        def local_sync(state, grads, **kwargs):
            return [grad.stop_grad() for grad in grads]

        with mock.patch.object(
                fsdp_grad_sync, "_sync_sharded_grads_from_full_grads",
                side_effect=local_sync):
            fsdp.fill_fsdp_optimizer_grads_from_grad_map(
                [optimizer], {id(full[0]): jt.ones_like(full[0])})
            optimizer.step()
        state_dict = optimizer.state_dict()
        self.assertEqual(set(state_dict["state"]), {0})
        self.assertIn("momentum_buffer", state_dict["state"][0])
        self.assertIn(entries[0].shard, optimizer.state)
        jt.sync_all(True)

    def test_flat_fsdp_preserves_frozen_parameter_state(self):
        fsdp, state, entries, full = self._fake_flat_fsdp_state(
            ([1.0, 2.0], [3.0, 4.0]))
        entries[0].requires_grad = True
        entries[1].requires_grad = False
        full[1].stop_grad()
        fsdp._refresh_flat_entry_shards(state)
        self.assertFalse(entries[0].shard.is_stop_grad())
        self.assertTrue(entries[1].shard.is_stop_grad())
        frozen_before = entries[1].shard.numpy().copy()
        optimizer = torch.optim.AdamW(
            [entry.shard for entry in entries], lr=0.01, weight_decay=0.2)

        def local_sync(current_state, grads, **kwargs):
            return [grad.reshape(entry.shard.shape).stop_grad()
                    for entry, grad in zip(current_state.true_fsdp_params, grads)]

        with mock.patch.object(
                fsdp_grad_sync, "_sync_sharded_grads_from_full_grads",
                side_effect=local_sync):
            fsdp.fill_fsdp_optimizer_grads_from_grad_map(
                [optimizer], {id(full[0]): jt.ones_like(full[0])})
            optimizer.step()
        self.assertTrue(entries[1].shard.is_stop_grad())
        np.testing.assert_array_equal(entries[1].shard.numpy(), frozen_before)
        self.assertNotIn(entries[1].shard, optimizer.state)
        self.assertEqual(optimizer.param_groups[0]["_torch_steps"], [1, 0])
        jt.sync_all(True)

    def test_unresharded_full_grad_is_visible_and_controls_step(self):
        fsdp, state, entries, full = self._fake_fsdp_state(([1.0, 2.0],))
        state.true_fsdp_unsharded = True
        fsdp._mark_fsdp_param_var(full[0], state, entries[0], "full")
        setattr(entries[0].owner, entries[0].attr, full[0])
        optimizer = torch.optim.SGD([entries[0].shard], lr=0.1)

        def local_sync(current_state, grads, **kwargs):
            return [grad.stop_grad() for grad in grads]

        with mock.patch.object(
                fsdp_grad_sync, "_sync_sharded_grads_from_full_grads",
                side_effect=local_sync):
            fsdp.fill_fsdp_optimizer_grads_from_grad_map(
                [optimizer], {id(full[0]): jt.ones_like(full[0])})
        self.assertIsNotNone(full[0].grad)
        public_grad = full[0].grad
        optimizer.zero_grad(set_to_none=False)
        self.assertIs(full[0].grad, public_grad)
        np.testing.assert_array_equal(
            public_grad.numpy(), np.zeros(2, dtype="float32"))
        with mock.patch.object(
                fsdp_grad_sync, "_sync_sharded_grads_from_full_grads",
                side_effect=local_sync):
            fsdp.fill_fsdp_optimizer_grads_from_grad_map(
                [optimizer], {id(full[0]): jt.ones_like(full[0])})
        self.assertIs(full[0].grad, public_grad)
        full[0].grad.mul_(0.5)
        optimizer.step()
        np.testing.assert_allclose(
            entries[0].shard.numpy(), np.array([0.95, 1.95], dtype="float32"),
            atol=1e-6, rtol=1e-6)
        jt.sync_all(True)

    def test_unresharded_full_manual_grad_creates_optimizer_slot(self):
        fsdp, state, entries, full = self._fake_fsdp_state(([1.0, 2.0],))
        state.true_fsdp_unsharded = True
        fsdp._mark_fsdp_param_var(full[0], state, entries[0], "full")
        setattr(entries[0].owner, entries[0].attr, full[0])
        optimizer = torch.optim.SGD([entries[0].shard], lr=0.1)

        manual = jt.ones_like(full[0]).stop_grad()
        full[0].grad = manual
        self.assertNotIn("grads", optimizer.param_groups[0])
        optimizer.step()
        np.testing.assert_allclose(
            entries[0].shard.numpy(), np.array([0.9, 1.9], dtype="float32"),
            atol=1e-6, rtol=1e-6)
        self.assertIs(full[0].grad, manual)
        jt.sync_all(True)

    def test_flat_fsdp_dynamic_requires_grad_persists_across_refresh(self):
        fsdp, state, entries, full = self._fake_flat_fsdp_state(
            ([1.0, 2.0], [3.0, 4.0]))
        state.true_fsdp_unsharded = True
        for entry, param in zip(entries, full):
            entry.full_param = param
            fsdp._mark_fsdp_param_var(param, state, entry, "full")
            setattr(entry.owner, entry.attr, param)

        full[0].requires_grad_(False)
        self.assertFalse(entries[0].requires_grad)
        self.assertFalse(entries[0].shard.requires_grad)
        self.assertFalse(state.true_fsdp_flat_shard.is_stop_grad())
        fsdp._refresh_flat_entry_shards(state)
        self.assertTrue(entries[0].shard.is_stop_grad())
        self.assertFalse(entries[1].shard.is_stop_grad())

        full[0].requires_grad_(True)
        self.assertTrue(entries[0].requires_grad)
        fsdp._refresh_flat_entry_shards(state)
        self.assertFalse(entries[0].shard.is_stop_grad())
        self.assertFalse(state.true_fsdp_flat_shard.is_stop_grad())
        jt.sync_all(True)

    def test_fsdp_freeze_registry_keeps_only_weak_holders(self):
        fsdp, state, entries, full = self._fake_flat_fsdp_state(
            ([1.0, 2.0], [3.0, 4.0]))
        state.true_fsdp_unsharded = True
        for entry, param in zip(entries, full):
            entry.full_param = param
            fsdp._mark_fsdp_param_var(param, state, entry, "full")
            setattr(entry.owner, entry.attr, param)
            param.requires_grad_(True)

        registry = get_tensor_state(jt).leaf_params
        self.assertIn(id(state.true_fsdp_flat_shard), registry)
        for entry, param in zip(entries, full):
            self.assertIn(id(param), registry)
            self.assertNotIn(id(entry.shard), registry)
            self.assertTrue(registry.is_weak(id(param)))
        self.assertTrue(registry.is_weak(id(state.true_fsdp_flat_shard)))

        full[0].requires_grad_(False)
        self.assertFalse(entries[0].requires_grad)
        self.assertFalse(full[0].requires_grad)
        self.assertFalse(entries[0].shard.requires_grad)
        self.assertTrue(state.true_fsdp_flat_shard.requires_grad)

        full[1].requires_grad_(False)
        self.assertFalse(entries[1].requires_grad)
        self.assertFalse(full[1].requires_grad)
        self.assertFalse(entries[1].shard.requires_grad)
        self.assertFalse(state.true_fsdp_flat_shard.requires_grad)

        full[1].requires_grad_(True)
        self.assertTrue(entries[1].requires_grad)
        self.assertTrue(full[1].requires_grad)
        self.assertTrue(entries[1].shard.requires_grad)
        self.assertTrue(state.true_fsdp_flat_shard.requires_grad)
        full[1].requires_grad_(False)

        full_ids = [id(param) for param in full]
        full_refs = [weakref.ref(param) for param in full]
        for entry in entries:
            entry.full_param = None
            setattr(entry.owner, entry.attr, entry.shard)
        del param, full
        jt.gc()
        self.assertTrue(all(reference() is None for reference in full_refs))
        self.assertTrue(all(holder_id not in registry for holder_id in full_ids))
        jt.sync_all(True)

    def test_shared_flat_fsdp_refreshes_every_optimizer_parameter(self):
        fsdp, state, entries, full = self._fake_flat_fsdp_state(([1.0, 2.0],))
        first = torch.optim.AdamW([entries[0].shard], lr=0.01)
        second = torch.optim.AdamW([entries[0].shard], lr=0.01)

        def local_sync(current_state, grads, **kwargs):
            return [grad.reshape(entry.shard.shape).stop_grad()
                    for entry, grad in zip(current_state.true_fsdp_params, grads)]

        with mock.patch.object(
                fsdp_grad_sync, "_sync_sharded_grads_from_full_grads",
                side_effect=local_sync):
            fsdp.fill_fsdp_optimizer_grads_from_grad_map(
                [first, second], {id(full[0]): jt.ones_like(full[0])})
            second.step()
        second_momentum = second.param_groups[0]["m"][0]

        entries[0].full_param = full[0]
        with mock.patch.object(
                fsdp_grad_sync, "_sync_sharded_grads_from_full_grads",
                side_effect=local_sync):
            fsdp.fill_fsdp_optimizer_grads_from_grad_map(
                [first, second], {id(full[0]): jt.ones_like(full[0])})
            first.step()

        self.assertIs(first.param_groups[0]["params"][0], entries[0].shard)
        self.assertIs(second.param_groups[0]["params"][0], entries[0].shard)
        self.assertIs(second.state[entries[0].shard]["exp_avg"], second_momentum)
        retained = second.param_groups[0]["grads"][0]
        second.zero_grad(set_to_none=False)
        self.assertIs(second.param_groups[0]["grads"][0], retained)
        np.testing.assert_array_equal(
            retained.numpy(), np.zeros_like(retained.numpy()))
        jt.sync_all(True)

    def test_mixed_fsdp_and_plain_adamw_advances_once(self):
        fsdp, _, entries, full = self._fake_fsdp_state(([1.0, 2.0],))
        plain = torch.tensor(
            np.array([3.0, 4.0], dtype="float32"), requires_grad=True)
        optimizer = torch.optim.AdamW(
            [entries[0].shard, plain], lr=0.01, weight_decay=0.1)

        def local_sync(state, grads, **kwargs):
            return [grad.stop_grad() for grad in grads]

        with mock.patch.object(
                fsdp_grad_sync, "_sync_sharded_grads_from_full_grads",
                side_effect=local_sync):
            ((full[0] * full[0]).sum() + (plain * plain).sum()).backward()
            optimizer.step()

        self.assertEqual(optimizer.n_step, 1)
        self.assertEqual(optimizer.param_groups[0]["_torch_steps"], [1, 1])
        self.assertIsNotNone(entries[0].shard.grad)
        self.assertIsNotNone(plain.grad)
        optimizer.zero_grad(set_to_none=True)
        fsdp_before = entries[0].shard.numpy().copy()
        plain_before = plain.numpy().copy()
        optimizer.step()
        np.testing.assert_array_equal(entries[0].shard.numpy(), fsdp_before)
        np.testing.assert_array_equal(plain.numpy(), plain_before)
        self.assertEqual(optimizer.n_step, 1)

        fsdp, _, entries, full = self._fake_fsdp_state(([1.0, 2.0],))
        plain = torch.tensor(
            np.array([3.0, 4.0], dtype="float32"), requires_grad=True)
        native_optimizer = torch.optim.AdamW(
            [entries[0].shard, plain], lr=0.01, weight_decay=0.1)
        native_loss = (full[0] * full[0]).sum() + (plain * plain).sum()
        with mock.patch.object(
                fsdp_grad_sync, "_sync_sharded_grads_from_full_grads",
                side_effect=local_sync):
            returned = native_optimizer.step(native_loss)
        self.assertIs(returned, native_loss)
        self.assertEqual(native_optimizer.n_step, 1)
        self.assertEqual(
            native_optimizer.param_groups[0]["_torch_steps"], [1, 1])
        self.assertIsNone(entries[0].shard.grad)
        self.assertIsNone(plain.grad)
        jt.sync_all(True)

    def test_single_rank_fully_shard_preserves_math_and_state(self):
        from torch.distributed.fsdp import (
            CPUOffloadPolicy,
            DataParallelMeshDims,
            FSDPModule,
            FlatParameter,
            FullOptimStateDictConfig,
            FullStateDictConfig,
            MixedPrecisionPolicy,
            OptimStateKeyType,
            StateDictType,
            StateDictSettings,
            FullyShardedDataParallel,
            ShardedGradScaler,
            fully_shard,
            share_comm_ctx,
        )
        from torch.distributed._composable.fsdp import fully_shard as composable_fully_shard
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint,
            checkpoint_wrapper,
        )

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        self.assertTrue(torch.distributed.is_available())
        self.assertIs(fully_shard, composable_fully_shard)
        self.assertEqual(mesh.size("dp"), 1)
        self.assertEqual(mesh.size(mesh_dim=0), 1)
        with mesh:
            pass
        self.assertEqual(DataParallelMeshDims(shard="dp").shard_names, ("dp",))

        module = torch.nn.Linear(3, 2)
        x = jt.array(np.random.RandomState(11).randn(4, 3).astype("float32"))
        ref = module(x).numpy()
        param_ids = [id(p) for p in module.parameters()]

        returned = fully_shard(
            module,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32),
            offload_policy=CPUOffloadPolicy(),
            dp_mesh_dims=DataParallelMeshDims(shard="dp"),
        )

        self.assertIs(returned, module)
        self.assertIsInstance(module, FSDPModule)
        self.assertNotIsInstance(module, FullyShardedDataParallel)
        self.assertEqual([id(p) for p in module.parameters()], param_ids)
        np.testing.assert_allclose(module(x).numpy(), ref, atol=1e-6)
        self.assertEqual(sorted(module.state_dict().keys()), ["bias", "weight"])

        grads = jt.grad((module(x) ** 2).sum(), list(module.parameters()))
        self.assertTrue(all(float(jt.abs(g).sum().item()) > 0 for g in grads))

        module.unshard()
        module.reshard()
        module.set_requires_gradient_sync(False)
        module.set_requires_all_reduce(False)
        module.set_all_reduce_hook(lambda *args, **kwargs: None, stream=None)
        module.set_allocate_memory_from_process_group_for_comm(True)
        module.set_custom_all_gather(lambda *args, **kwargs: None)
        module.set_custom_reduce_scatter(lambda *args, **kwargs: None)
        module.set_reduce_scatter_unused_params(True)
        module.set_reduce_scatter_max_input_buffers(2)
        module.set_separate_reduce_scatter_group(None)
        module.set_reshard_after_backward(False)
        module.set_unshard_in_backward(False)
        module.set_force_sum_reduction_for_comms(True)
        module.set_symm_mem_for_comm("NCCL")
        module.set_post_optim_event(None)
        module.reset_iter_state()
        with share_comm_ctx([module]):
            pass
        state = module._get_fsdp_state()
        self.assertTrue(callable(state.all_reduce_hook))
        self.assertTrue(state.allocate_memory_from_process_group_for_comm)
        self.assertTrue(callable(state.custom_all_gather))
        self.assertTrue(callable(state.custom_reduce_scatter))
        self.assertTrue(state.reduce_scatter_unused_params)
        self.assertEqual(state.reduce_scatter_max_input_buffers, 2)
        self.assertIsNone(state.separate_reduce_scatter_group)
        self.assertTrue(state.iter_state_reset)
        self.assertTrue(state.share_comm_ctx)
        self.assertEqual(state.symm_mem_for_comm, "NCCL")
        self.assertIs(module._get_fsdp_state(), getattr(module, "_fsdp_state", None))

        wrapped = FullyShardedDataParallel(torch.nn.Linear(3, 2))
        self.assertEqual(tuple(wrapped(x).shape), (4, 2))
        self.assertEqual(list(wrapped.buffers()), [])
        self.assertEqual(list(wrapped.named_buffers()), [])
        self.assertEqual(len(list(StateDictType)), 3)
        settings = StateDictSettings(
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(),
            FullOptimStateDictConfig(),
        )
        self.assertIs(settings.state_dict_type, StateDictType.FULL_STATE_DICT)
        self.assertIs(OptimStateKeyType.PARAM_NAME, OptimStateKeyType.PARAM_NAME)
        flat_parameter = FlatParameter(torch.ones(1))
        self.assertIsInstance(flat_parameter, torch.Tensor)
        self.assertTrue(flat_parameter.requires_grad)
        FullyShardedDataParallel.set_state_dict_type(
            wrapped,
            StateDictType.LOCAL_STATE_DICT,
            state_dict_config=FullStateDictConfig(),
        )
        self.assertIs(wrapped._fsdp_state_dict_type[0], StateDictType.LOCAL_STATE_DICT)
        with FullyShardedDataParallel.state_dict_type(wrapped, StateDictType.SHARDED_STATE_DICT):
            self.assertIs(wrapped._fsdp_state_dict_type[0], StateDictType.SHARDED_STATE_DICT)
        self.assertIs(wrapped._fsdp_state_dict_type[0], StateDictType.LOCAL_STATE_DICT)
        scaler = ShardedGradScaler(enabled=False)
        self.assertEqual(scaler.state_dict(), {"enabled": False})
        scaler.load_state_dict({"enabled": True})
        self.assertEqual(scaler.state_dict(), {"enabled": True})
        self.assertIs(checkpoint_wrapper(module), module)
        self.assertEqual(tuple(checkpoint(module, jt.ones((1, 3))).shape), (1, 2))

    def test_fsdp_module_metaclass_composes_with_abc(self):
        from torch.distributed.fsdp import FSDPModule

        class AbstractFSDPModule(abc.ABC, FSDPModule):
            @abc.abstractmethod
            def execute(self):
                pass

        self.assertTrue(issubclass(AbstractFSDPModule, FSDPModule))

    def test_module_to_empty_preserves_materialized_parameters(self):
        module = torch.nn.Linear(3, 2)
        before = [parameter.numpy().copy() for parameter in module.parameters()]
        returned = module.to_empty(device="cpu")
        self.assertIs(returned, module)
        for parameter, expected in zip(module.parameters(), before):
            np.testing.assert_array_equal(parameter.numpy(), expected)

    def test_private_clip_grad_helpers(self):
        from torch.nn.utils.clip_grad import (
            _clip_grads_with_norm_,
            _get_total_norm,
        )

        grads = [
            jt.array(np.array([3.0, 4.0], dtype="float32")),
            jt.array(np.array([0.0, -3.0], dtype="float32")),
        ]
        parameters = [types.SimpleNamespace(grad=grad) for grad in grads]
        total = _get_total_norm(grads, norm_type=2.0)
        self.assertAlmostEqual(float(total.item()), np.sqrt(34.0), places=5)
        _clip_grads_with_norm_(parameters, 1.0, total)
        clipped = np.concatenate([grad.numpy() for grad in grads])
        self.assertLessEqual(float(np.linalg.norm(clipped)), 1.00001)

    def test_distributed_store_types_are_importable(self):
        from torch.distributed import (
            Backend,
            FileStore,
            P2POp,
            PrefixStore,
            Store,
            TCPStore,
            batch_isend_irecv,
            is_backend_available,
            is_gloo_available,
            is_mpi_available,
            is_nccl_available,
            rendezvous as distributed_rendezvous,
        )
        import torch.distributed._symmetric_memory as symmetric_memory
        from torch.distributed.distributed_c10d import (
            _get_default_group,
            _get_default_timeout,
            _unregister_process_group,
        )
        from torch.distributed.rendezvous import rendezvous

        store = TCPStore()
        prefixed = PrefixStore("model/", store)
        prefixed.set("step", b"1")
        self.assertEqual(prefixed.get("step"), b"1")
        self.assertTrue(issubclass(TCPStore, Store))
        self.assertTrue(issubclass(FileStore, Store))
        self.assertIs(torch._C._distributed_c10d.Store, Store)
        self.assertIs(torch._C._distributed_c10d.TCPStore, TCPStore)
        self.assertIs(torch._C._distributed_c10d.FileStore, FileStore)
        self.assertIs(torch._C._distributed_c10d.PrefixStore, PrefixStore)
        self.assertEqual(Backend.NCCL, "nccl")
        # NCCL rides on CUDA, so its availability follows the build -- the CPU
        # session deliberately runs without a device, and asserting it
        # unconditionally only says which machine the suite last ran on.
        self.assertEqual(
            is_backend_available("nccl"), _test_capability.check_accelerator("cuda", backend=jt).enabled)
        self.assertFalse(is_backend_available("gloo"))
        self.assertFalse(is_gloo_available())
        self.assertEqual(is_nccl_available(), is_backend_available("nccl"))
        self.assertEqual(is_mpi_available(), is_backend_available("mpi"))
        self.assertEqual(
            is_backend_available("mpi"),
            _test_capability.library_enabled("mpi", backend=jt),
        )
        self.assertFalse(is_backend_available("unknown"))
        self.assertEqual(batch_isend_irecv([]), [])
        self.assertEqual(P2POp(lambda: None, jt.ones(1), 0).peer, 0)
        self.assertFalse(symmetric_memory.is_symm_mem_enabled_for_group("world"))
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            symmetric_memory.enable_symm_mem_for_group("world")
        self.assertIsNotNone(_get_default_group())
        self.assertGreater(_get_default_timeout().total_seconds(), 0)
        self.assertIsNone(_unregister_process_group("unused"))
        rendezvous_store, rank, world_size = next(rendezvous("env://"))
        self.assertIsInstance(rendezvous_store, TCPStore)
        self.assertEqual((rank, world_size), (0, 1))
        self.assertTrue(callable(distributed_rendezvous))

    def test_distributed_composite_backend_matching(self):
        self.assertTrue(_backend_matches_active("nccl", "nccl"))
        self.assertTrue(
            _backend_matches_active("cpu:gloo,cuda:nccl", "nccl")
        )
        self.assertTrue(_backend_matches_active("cpu:mpi,cuda:nccl", "mpi"))
        self.assertFalse(
            _backend_matches_active("cpu:gloo,cuda:gloo", "nccl")
        )
        self.assertFalse(_backend_matches_active("gloo,nccl", "nccl"))

    def test_dtensor_and_private_import_paths(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import (
            DTensor,
            Placement,
            Replicate,
            Shard,
            distribute_tensor,
            ones,
            zeros,
            full,
            randn,
        )
        from torch.distributed.tensor._dtensor_spec import DTensorSpec
        from torch.distributed.tensor._api import empty as api_empty
        from torch.distributed._tensor import distribute_tensor as legacy_distribute_tensor
        from torch.distributed._tensor import linspace as legacy_linspace
        from torch.distributed._tensor.device_mesh import init_device_mesh as legacy_init_device_mesh
        from torch.distributed.tensor.placement_types import Partial
        from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module
        from torch.distributed.tensor.parallel.api import parallelize_module as api_parallelize_module
        from torch.distributed.tensor.parallel.loss import loss_parallel
        from torch.distributed.tensor.parallel.style import RowwiseParallel
        from torch.distributed.fsdp.wrap import CustomPolicy, ModuleWrapPolicy, _or_policy
        from torch.distributed.fsdp._fully_shard import fully_shard as package_fully_shard
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.fsdp._runtime_utils import _lazy_init
        from torch.distributed.fsdp._common_utils import _get_module_fsdp_state
        from torch.distributed.fsdp._fully_shard._fsdp_state import (
            _get_module_fsdp_state_if_fully_sharded_module,
        )
        from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
            all_gather,
            reduce_scatter,
        )
        from torch.distributed.fsdp._fully_shard._fsdp_param import (
            FlatParameter as PrivateFlatParameter,
        )
        from torch.distributed.fsdp._fully_shard._fsdp_common import (
            FSDPMeshInfo,
            ShardPlacementResult,
        )
        from torch.distributed.fsdp._fully_shard._fsdp_init import (
            _get_mesh_info,
            _get_post_forward_mesh_info,
        )

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        self.assertEqual(legacy_init_device_mesh("cpu", (1,)).size(mesh_dim=0), 1)
        self.assertIs(fully_shard, package_fully_shard)

        dt = distribute_tensor(jt.array([1, 2, 3]), mesh, [Replicate()])
        dt2 = legacy_distribute_tensor(jt.array([1, 2, 3]), mesh, [Shard(0)])
        for tensor in (
            ones((2, 3), device_mesh=mesh, placements=[Replicate()]),
            zeros(2, 3, device_mesh=mesh, placements=[Replicate()]),
            full((2, 3), 7, device_mesh=mesh, placements=[Replicate()]),
            randn(2, 3, device_mesh=mesh, placements=[Replicate()]),
            api_empty(2, 3, device_mesh=mesh, placements=[Replicate()]),
            legacy_linspace(0, 1, 3, device_mesh=mesh, placements=[Replicate()]),
        ):
            self.assertIsInstance(tensor, DTensor)
            self.assertTrue(hasattr(tensor, "to_local"))
        self.assertIsInstance(dt, DTensor)
        self.assertIsInstance(dt2, DTensor)
        self.assertIs(dt.to_local(), dt)
        self.assertTrue(isinstance(Shard(0), Placement))
        self.assertTrue(Partial().is_partial())
        self.assertEqual(np.asarray(DTensor(jt.array([1, 2, 3]))).tolist(), [1, 2, 3])
        self.assertIsInstance(DTensorSpec(mesh=mesh, placements=[Replicate()]), DTensorSpec)

        self.assertIsInstance(FSDPMeshInfo(mesh=mesh), FSDPMeshInfo)
        self.assertIsInstance(ShardPlacementResult(shard_dim=0), ShardPlacementResult)
        self.assertIs(_get_mesh_info(mesh).mesh, mesh)
        self.assertIs(_get_post_forward_mesh_info(mesh).mesh, mesh)

        module = torch.nn.Linear(2, 2)
        self.assertIs(parallelize_module(module), module)
        self.assertIs(api_parallelize_module(module), module)
        self.assertIsInstance(ParallelStyle(), ParallelStyle)
        self.assertIsInstance(RowwiseParallel(), ParallelStyle)
        self.assertTrue(ModuleWrapPolicy([torch.nn.Linear])(module, False, 0))
        self.assertTrue(CustomPolicy(lambda *args: True)(module, False, 0))
        self.assertTrue(_or_policy(module, False, 0, policies=[lambda **kwargs: True]))
        with loss_parallel():
            pass
        fully_shard(module)
        self.assertIs(_lazy_init(module._get_fsdp_state()), module._get_fsdp_state())
        self.assertIs(_get_module_fsdp_state(module), module._get_fsdp_state())
        self.assertIs(
            _get_module_fsdp_state_if_fully_sharded_module(module),
            module._get_fsdp_state(),
        )
        self.assertIs(all_gather(dt), dt)
        self.assertIs(reduce_scatter(dt), dt)
        flat_parameter = PrivateFlatParameter(torch.ones(1))
        self.assertIsInstance(flat_parameter, torch.Tensor)
        self.assertTrue(flat_parameter.requires_grad)


class TestTheTwoForwardHooksActAsOne(unittest.TestCase):
    """One forward unshards once.

    Two hooks reach ``_execute_with_true_fsdp`` for the same forward, and both
    are needed, because they cover different dispatch paths:

    * ``Module.__call__`` (installed once, in ``compat/torch/installers/nn.py``)
      is the only one that sees a torch-style ``forward`` override, a
      per-instance ``self.forward``, or the fused RMSNorm shortcut -- none of
      which reach ``execute``;
    * the per-instance ``execute`` wrapper that ``fully_shard`` installs is the
      only one that sees a direct ``module.execute(...)`` bypassing
      ``__call__``.

    On the ordinary jittor-style path both fire, nested, and the whole
    unshard/reshard sequence therefore ran twice per forward. It was never
    *wrong* -- each half is individually idempotent -- which is exactly why it
    survived: the second pass was invisible except as work.

    These tests drive the two hooks directly. ``fully_shard`` cannot reach the
    true-FSDP path on one rank (``common._in_true_distributed()`` is false
    without an MPI/NCCL world), so a real module would exercise nothing.
    """

    def _module(self, reshard_after_forward=True):
        state = types.SimpleNamespace(
            true_fsdp_initialized=True,
            true_fsdp_flat=False,
            true_fsdp_params=[],
            reshard_after_forward=reshard_after_forward,
        )
        return types.SimpleNamespace(_fsdp_state=state), state

    def _nested_call(self, module, forward):
        """Exactly the nesting the two hooks produce for one forward."""
        def execute_hook(*args, **kwargs):        # fully_shard's execute wrapper
            return fsdp_shard._execute_with_true_fsdp(
                module, forward, *args, **kwargs)

        return fsdp_shard._execute_with_true_fsdp(   # Module.__call__'s hook
            module, execute_hook)

    def test_one_forward_unshards_and_reshards_exactly_once(self):
        module, _state = self._module()
        calls = {"unshard": 0, "reshard": 0}
        real_unshard = fsdp_shard._unshard_module_params
        real_reshard = fsdp_shard._reshard_module_params

        def counting_unshard(m):
            calls["unshard"] += 1
            return real_unshard(m)

        def counting_reshard(m):
            calls["reshard"] += 1
            return real_reshard(m)

        with mock.patch.object(fsdp_shard, "_unshard_module_params",
                               counting_unshard), \
             mock.patch.object(fsdp_shard, "_reshard_module_params",
                               counting_reshard):
            out = self._nested_call(module, lambda: "forward-result")

        self.assertEqual(out, "forward-result")
        # Without the depth guard this is {"unshard": 2, "reshard": 2}: the
        # inner hook reshards on the way out and the outer one then re-enters
        # the entire sequence, on every forward.
        self.assertEqual(calls, {"unshard": 1, "reshard": 1})

    def test_the_parameters_are_whole_at_the_innermost_point(self):
        module, state = self._module()
        seen = []

        def forward():
            seen.append(bool(getattr(state, "true_fsdp_unsharded", False)))
            return None

        self._nested_call(module, forward)
        self.assertEqual(seen, [True])
        self.assertFalse(getattr(state, "true_fsdp_unsharded", False))

    def test_the_unsharded_window_closes_in_the_outer_hook_not_the_inner_one(self):
        # What the idempotence was covering up. The inner hook used to put the
        # shards back before the outer hook's `finally` ran, so the window in
        # which the parameters were whole ended in the middle of the outer
        # hook. Anything the outer hook did after its inner call -- and the
        # `nn.py` side wraps the result in the execution-pipelining hook there
        # -- saw resharded parameters.
        module, state = self._module()
        real_reshard = fsdp_shard._reshard_module_params
        found_unsharded = []

        def recording_reshard(m):
            found_unsharded.append(bool(getattr(state, "true_fsdp_unsharded", False)))
            return real_reshard(m)

        with mock.patch.object(fsdp_shard, "_reshard_module_params",
                               recording_reshard):
            self._nested_call(module, lambda: None)

        # One reshard, and it is the one that actually closes the window --
        # not a second visit finding the work already done.
        self.assertEqual(found_unsharded, [True])

    def test_the_depth_counter_is_left_clean_even_when_the_forward_raises(self):
        module, state = self._module()

        def boom():
            raise ValueError("forward failed")

        with self.assertRaises(ValueError):
            self._nested_call(module, boom)

        self.assertEqual(getattr(state, "true_fsdp_execute_depth", 0), 0)
        self.assertFalse(getattr(state, "true_fsdp_unsharded", False))

    def test_reshard_after_forward_false_keeps_the_parameters_whole(self):
        module, state = self._module(reshard_after_forward=False)
        self._nested_call(module, lambda: None)
        self.assertTrue(getattr(state, "true_fsdp_unsharded", False))
        self.assertEqual(getattr(state, "true_fsdp_execute_depth", 0), 0)

    def test_a_module_without_fsdp_state_is_passed_straight_through(self):
        plain = types.SimpleNamespace()
        self.assertEqual(
            fsdp_shard._execute_with_true_fsdp(plain, lambda: "plain"), "plain")


if __name__ == "__main__":
    unittest.main()
