"""FSDP2 compatibility tests for ``import jittor as torch``.

Run:
    python -m pytest tests/compat/torch/test_torch_compat_fsdp2.py
"""
import abc
import unittest
import types
from unittest import mock
import numpy as np
import jittor as torch
import jittor as jt
from jittor.compat import fsdp2 as canonical_fsdp
from jittor.compat.fsdp2 import grad_sync as fsdp_grad_sync
from jittor.compat.fsdp2 import shard as fsdp_shard
from jittor.compat.torch.installers.distributed import _backend_matches_active


class TestFSDP2Compat(unittest.TestCase):
    def _fake_fsdp_state(self, values):
        fsdp = canonical_fsdp

        owner = types.SimpleNamespace()
        entries = []
        full_params = []
        state = types.SimpleNamespace(
            true_fsdp_initialized=True,
            true_fsdp_flat=False,
            true_fsdp_rank=0,
            true_fsdp_world_size=1,
            true_fsdp_unsharded=False,
            true_fsdp_module=None,
        )
        for i, value in enumerate(values):
            full = jt.array(np.asarray(value, dtype="float32"))
            shard = jt.array(np.asarray(value, dtype="float32"))
            attr = f"param_{i}"
            entry = types.SimpleNamespace(
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
        flat = jt.array(np.concatenate([value.reshape(-1) for value in arrays]))
        state = types.SimpleNamespace(
            true_fsdp_initialized=True,
            true_fsdp_flat=True,
            true_fsdp_rank=0,
            true_fsdp_world_size=1,
            true_fsdp_unsharded=False,
            true_fsdp_module=None,
            true_fsdp_flat_total_numel=int(flat.numel()),
            true_fsdp_flat_padded_numel=int(flat.numel()),
            true_fsdp_flat_shard_numel=int(flat.numel()),
            true_fsdp_flat_shard=flat,
        )
        entries = []
        full_params = []
        offset = 0
        for i, value in enumerate(arrays):
            full = jt.array(value)
            attr = f"param_{i}"
            entry = types.SimpleNamespace(
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
                    side_effect=lambda value: value):
                sharded = fsdp_grad_sync._sync_sharded_grads_from_full_grads(
                    state, full_grads)
            self.assertEqual(len(sharded), len(entries))
            for entry, grad in zip(entries, sharded):
                self.assertEqual(tuple(grad.shape), tuple(entry.shard.shape))
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
        plain = jt.array(np.array([3.0, 4.0], dtype="float32"))
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
        plain = jt.array(np.array([3.0, 4.0], dtype="float32"))
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
        self.assertIsNotNone(FlatParameter(torch.ones(1)))
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
        self.assertIs(jt._C._distributed_c10d.Store, Store)
        self.assertIs(jt._C._distributed_c10d.TCPStore, TCPStore)
        self.assertIs(jt._C._distributed_c10d.FileStore, FileStore)
        self.assertIs(jt._C._distributed_c10d.PrefixStore, PrefixStore)
        self.assertEqual(Backend.NCCL, "nccl")
        # NCCL rides on CUDA, so its availability follows the build -- the CPU
        # session deliberately runs without a device, and asserting it
        # unconditionally only says which machine the suite last ran on.
        self.assertEqual(
            is_backend_available("nccl"), bool(getattr(jt, "has_cuda", False)))
        self.assertFalse(is_backend_available("gloo"))
        self.assertFalse(is_gloo_available())
        self.assertEqual(is_nccl_available(), is_backend_available("nccl"))
        self.assertEqual(is_mpi_available(), is_backend_available("mpi"))
        self.assertEqual(
            is_backend_available("mpi"),
            bool(getattr(jt.compile_extern, "has_mpi", False)),
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
        self.assertIsNotNone(PrivateFlatParameter(torch.ones(1)))


if __name__ == "__main__":
    unittest.main()
