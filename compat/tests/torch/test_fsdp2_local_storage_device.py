"""Public CUDA world1 FSDP local storage and optimizer lifetime contracts.

Run Native and compat in separate processes. A passing singleton is not
multi-rank, other-device, mixed-precision, migration or PPO acceptance.
"""
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import pytest

pytestmark = pytest.mark.cuda


class TestFSDP2LocalStorageDevice(unittest.TestCase):
    def environment(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('real CUDA required')
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision('highest')
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
        jt = None
        if hasattr(torch, '_torch_compat_install_context'):
            import jittor as jt
            self.assertEqual(jt.core.backend_fallback_count(), 0)
            self.addCleanup(lambda: self.assertEqual(jt.core.backend_fallback_count(), 0))
        return torch, jt

    def test_cuda_canonical_initialization_independent_storage(self):
        import numpy as np
        torch, jt = self.environment()
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        from torch.distributed.device_mesh import init_device_mesh
        if not torch.distributed.is_initialized():
            directory = tempfile.TemporaryDirectory(prefix='fsdp-canonical-cuda-')
            self.addCleanup(directory.cleanup)
            torch.distributed.init_process_group('nccl', init_method=Path(directory.name, 'rdzv').as_uri(),
                rank=0, world_size=1, timeout=timedelta(seconds=120))
            self.addCleanup(torch.distributed.destroy_process_group)
        self.assertEqual(torch.distributed.get_world_size(), 1)
        self.assertEqual(torch.distributed.get_rank(), 0)
        self.assertEqual(torch.distributed.get_backend(), 'nccl')
        shapes = {'root_scale': (3,), 'frozen': (3,), 'root_linear.weight': (3, 3),
            'root_linear.bias': (3,), 'child.weight': (3, 3), 'child.bias': (3,)}
        expected = {}
        for seed, (name, shape) in enumerate(shapes.items(), 1):
            j = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
            expected[name] = (((j*7+seed*3) % 31-15)/23).astype(np.float32)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.root_scale = torch.nn.Parameter(torch.tensor(expected['root_scale'], device='cuda:0'))
                self.frozen = torch.nn.Parameter(torch.tensor(expected['frozen'], device='cuda:0'), requires_grad=False)
                self.root_linear = torch.nn.Linear(3, 3, device='cuda:0', dtype=torch.float32)
                self.child = torch.nn.Linear(3, 3, device='cuda:0', dtype=torch.float32)
                with torch.no_grad():
                    for name, p in self.named_parameters(): p.copy_(torch.tensor(expected[name], device='cuda:0'))
                self.scale_alias = self.root_scale
                self.frozen_alias = self.frozen
                self.child.weight_alias = self.child.weight
        def flags(t):
            return (t.requires_grad, t.is_leaf, t.retains_grad, t.grad_fn is None, t.grad is None)
        def descriptor(t):
            # No sync/materialization here: post-copy reads must not restore placement.
            row = dict(device=(t.device.type, t.device.index), dtype=t.dtype,
                shape=tuple(t.shape), stride=tuple(t.stride()), offset=t.storage_offset(),
                contiguous=t.is_contiguous(), flags=flags(t))
            if jt is not None:
                native = getattr(t, '_var', t)
                row['native'] = dict(location=native.location(),
                    dispatch=jt.core.dispatch_context([native]),
                    native_device_id=native.device_id, placement_backend=native.placement_backend)
            return row
        def snapshot(t):
            if jt is not None: getattr(t, '_var', t).sync()
            original_pointer = t.data_ptr()
            before = descriptor(t)
            self.assertEqual(before['device'], ('cuda', 0))
            self.assertEqual(before['dtype'], torch.float32)
            if jt is not None:
                self.assertEqual(before['native'], dict(location='device', dispatch=('cuda', 0),
                    native_device_id=0, placement_backend=1))
            with torch.no_grad():
                transport = t.clone()
                self.assertIsNot(transport, t)
                self.assertFalse(transport.requires_grad)
                if jt is not None:
                    transport_native = getattr(transport, '_var', transport)
                    self.assertIsNot(transport_native, getattr(t, '_var', t))
                    transport_native.sync()
                self.assertNotEqual(transport.data_ptr(), original_pointer)
                self.assertEqual(transport.device, torch.device('cuda:0'))
                host = transport.cpu()
                self.assertIsNot(host, t)
                self.assertEqual(host.device, torch.device('cpu'))
                result = np.array(host.numpy(), copy=True)
            # First original-object observation after transport; deliberately no sync.
            self.assertEqual(descriptor(t), before)
            self.assertTrue(np.isfinite(result).all())
            return result
        def exact(t, want):
            got = snapshot(t)
            self.assertEqual(got.shape, want.shape)
            self.assertEqual(got.dtype, want.dtype)
            np.testing.assert_array_equal(got.view(np.uint8), want.view(np.uint8))
        def aliases(model, parameters):
            self.assertIs(model.scale_alias, parameters['root_scale'])
            self.assertIs(model.frozen_alias, parameters['frozen'])
            self.assertIs(model.child.weight_alias, parameters['child.weight'])
        scope = nullcontext() if jt is None else jt.runtime.scope(use_cuda=1, device_id=0,
            backend_fallback='error', cuda_kernel_math='strict')
        with scope:
            model = Model().train()
            originals = dict(model.named_parameters())
            self.assertEqual(set(originals), set(shapes)); aliases(model, originals)
            old_flags = {name: flags(p) for name, p in originals.items()}
            for name, p in originals.items(): exact(p, expected[name])
            mesh = init_device_mesh('cuda', (1,))
            policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
            fully_shard(model.child, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
            fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
            canonical = dict(model.named_parameters())
            self.assertEqual(set(canonical), set(shapes)); aliases(model, canonical)
            local = {}
            for name, p in canonical.items():
                with torch.no_grad(): local[name] = p.to_local()
                with self.subTest(parameter=name):
                    self.assertIsNot(p, originals[name])
                    self.assertIsInstance(p, torch.nn.Parameter)
                    self.assertIsInstance(p, torch.distributed.tensor.DTensor)
                    self.assertEqual(tuple(p.shape), shapes[name])
                    self.assertEqual(p.dtype, torch.float32)
                    self.assertEqual(p.device, torch.device('cuda:0'))
                    self.assertEqual(flags(p), (name != 'frozen', True, False, True, True))
                    self.assertNotEqual(local[name].data_ptr(), originals[name].data_ptr())
                    if jt is not None:
                        self.assertIsNot(getattr(p, '_var', p), getattr(originals[name], '_var', originals[name]))
                    exact(local[name], expected[name]); exact(originals[name], expected[name])
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.125)
            slots = tuple(p for group in optimizer.param_groups for p in group['params'])
            self.assertEqual(len(slots), len(shapes))
            self.assertEqual({id(p) for p in slots}, {id(p) for p in canonical.values()})
            self.assertTrue(all(p is not old for p in slots for old in originals.values()))
            # Disposable initialization fixture: mutate both allocations independently.
            # No forward/backward/optimizer step or full-leaf lifecycle is exercised.
            for name, p in canonical.items():
                with self.subTest(independent_storage=name):
                    changed = expected[name] + np.float32(0.5)
                    with torch.no_grad(): local[name].copy_(torch.tensor(changed, device='cuda:0'))
                    exact(local[name], changed); exact(originals[name], expected[name])
                    old_changed = expected[name] - np.float32(0.25)
                    with torch.no_grad(): originals[name].copy_(torch.tensor(old_changed, device='cuda:0'))
                    exact(originals[name], old_changed); exact(local[name], changed)
                    with torch.no_grad(): current_local = p.to_local()
                    exact(current_local, changed)
                    self.assertEqual(current_local.data_ptr(), local[name].data_ptr())
                    self.assertEqual(flags(originals[name]), old_flags[name])
                    self.assertEqual(flags(p), (name != 'frozen', True, False, True, True))
                    aliases(model, canonical)
            self.assertIsNone(canonical['frozen'].grad)
            self.assertIsNone(originals['frozen'].grad)
            self.assertTrue(all(a is b for a, b in zip(slots,
                (p for group in optimizer.param_groups for p in group['params']))))


    def test_cuda_user_hook_parameter_windows(self):
        """Only C/U publication timing; not persistent-U, leaf or storage release."""
        import numpy as np
        torch, jt = self.environment()
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        from torch.distributed.device_mesh import init_device_mesh
        if not torch.distributed.is_initialized():
            directory = tempfile.TemporaryDirectory(prefix='fsdp-hook-window-cuda-')
            self.addCleanup(directory.cleanup)
            torch.distributed.init_process_group('nccl', init_method=Path(directory.name, 'rdzv').as_uri(),
                rank=0, world_size=1, timeout=timedelta(seconds=120))
            self.addCleanup(torch.distributed.destroy_process_group)
        self.assertEqual(torch.distributed.get_world_size(), 1)
        self.assertEqual(torch.distributed.get_rank(), 0)
        self.assertEqual(torch.distributed.get_backend(), 'nccl')
        shapes = {'root_scale': (3,), 'frozen': (3,), 'root_linear.weight': (3, 3),
            'root_linear.bias': (3,), 'child.weight': (3, 3), 'child.bias': (3,)}
        expected = {}
        for seed, (name, shape) in enumerate(shapes.items(), 1):
            j = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
            expected[name] = (((j*7+seed*3) % 31-15)/23).astype(np.float32)
        x_np = np.array([[3, -2, 7], [-5, 11, 1]], dtype=np.float32)/23
        dy_np = np.array([[-7, 5, 2], [13, -3, 4]], dtype=np.float32)/19
        state64 = {name: value.astype(np.float64) for name, value in expected.items()}
        h = x_np.astype(np.float64) @ state64['child.weight'].T + state64['child.bias']
        dh = dy_np.astype(np.float64) @ state64['root_linear.weight']
        expected_y = h @ state64['root_linear.weight'].T + state64['root_linear.bias'] + x_np*(state64['root_scale']+state64['frozen'])
        expected_dx = dh @ state64['child.weight'] + dy_np*(state64['root_scale']+state64['frozen'])
        gradients = {'root_scale': (dy_np.astype(np.float64)*x_np).sum(axis=0),
            'root_linear.weight': dy_np.astype(np.float64).T @ h,
            'root_linear.bias': dy_np.astype(np.float64).sum(axis=0),
            'child.weight': dh.T @ x_np.astype(np.float64), 'child.bias': dh.sum(axis=0)}
        def descriptor(t):
            row = (t.device.type, t.device.index, t.dtype, tuple(t.shape), tuple(t.stride()),
                t.storage_offset(), t.requires_grad, t.is_leaf, t.retains_grad, t.grad_fn is None)
            if jt is not None:
                native = getattr(t, '_var', t)
                row += (native.location(), jt.core.dispatch_context([native]),
                    native.device_id, native.placement_backend)
            return row
        def compare(t, want):
            with torch.no_grad():
                t = t.to_local() if isinstance(t, torch.distributed.tensor.DTensor) else t
            if jt is not None:
                getattr(t, '_var', t).sync()
            before = descriptor(t)
            self.assertEqual(t.device, torch.device('cuda:0'))
            self.assertEqual(t.dtype, torch.float32)
            if jt is not None:
                self.assertEqual(before[-4:], ('device', ('cuda', 0), 0, 1))
            with torch.no_grad():
                transport = t.clone()
                if jt is not None: getattr(transport, '_var', transport).sync()
                self.assertIsNot(transport, t)
                self.assertNotEqual(transport.data_ptr(), t.data_ptr())
                got = np.array(transport.cpu().numpy(), copy=True)
            self.assertEqual(descriptor(t), before)  # no original sync after transport
            want = np.asarray(want, dtype=np.float32)
            self.assertEqual(got.shape, want.shape)
            self.assertEqual(got.dtype, want.dtype)
            self.assertTrue(np.isfinite(got).all())
            np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)
        scope = nullcontext() if jt is None else jt.runtime.scope(use_cuda=1, device_id=0,
            backend_fallback='error', cuda_kernel_math='strict')
        with scope:
            for reshard in (True, False):
                live, events = {}, []
                class Child(torch.nn.Linear):
                    def forward(self, x):
                        live.update({'child.'+name: p for name, p in self.named_parameters()})
                        return super().forward(x)
                class Model(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.root_scale = torch.nn.Parameter(torch.tensor(expected['root_scale'], device='cuda:0'))
                        self.frozen = torch.nn.Parameter(torch.tensor(expected['frozen'], device='cuda:0'), requires_grad=False)
                        self.root_linear = torch.nn.Linear(3, 3, device='cuda:0', dtype=torch.float32)
                        self.child = Child(3, 3, device='cuda:0', dtype=torch.float32)
                        with torch.no_grad():
                            for name, p in self.named_parameters():
                                p.copy_(torch.tensor(expected[name], device='cuda:0'))
                    def forward(self, x):
                        live.update({name: p for name, p in self.named_parameters() if not name.startswith('child.')})
                        return self.root_linear(self.child(x)) + x*self.root_scale + x*self.frozen
                model = Model().train()
                mesh = init_device_mesh('cuda', (1,))
                policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
                fully_shard(model.child, mesh=mesh, reshard_after_forward=reshard, mp_policy=policy)
                fully_shard(model, mesh=mesh, reshard_after_forward=reshard, mp_policy=policy)
                canonical = dict(model.named_parameters())
                self.assertEqual(set(canonical), set(shapes))
                optimizer = torch.optim.SGD(model.parameters(), lr=0.125)
                optimizer.zero_grad(set_to_none=True)
                hooks = []
                for label, module in (('root', model), ('child', model.child)):
                    for prepend, suffix in ((True, 'prepend'), (False, 'default')):
                        def pre(module, args, label=label, suffix=suffix):
                            events.append((label+'/pre_'+suffix, dict(model.named_parameters())))
                        def post(module, args, out, label=label, suffix=suffix):
                            events.append((label+'/post_'+suffix, dict(model.named_parameters())))
                        hooks.append(module.register_forward_pre_hook(pre, prepend=prepend))
                        hooks.append(module.register_forward_hook(post, prepend=prepend))
                try:
                    x = torch.tensor(x_np, device='cuda:0', requires_grad=True)
                    y = model(x)
                    after = dict(model.named_parameters())
                    y.backward(torch.tensor(dy_np, device='cuda:0'))
                finally:
                    for hook in hooks: hook.remove()
                compare(y, expected_y); compare(x.grad, expected_dx)
                self.assertIsNone(canonical['frozen'].grad)
                for name, want in gradients.items():
                    self.assertIsNotNone(canonical[name].grad)
                    compare(canonical[name].grad, want)
                self.assertEqual(set(live), set(shapes))
                labels = ['root/pre_prepend', 'root/pre_default', 'child/pre_prepend',
                    'child/pre_default', 'child/post_prepend', 'child/post_default',
                    'root/post_prepend', 'root/post_default']
                self.assertEqual([label for label, _ in events], labels)
                publications = [('C','C'), ('U','C'), ('U','C'), ('U','U'), ('U','U'),
                    ('U','C' if reshard else 'U'), ('U','C' if reshard else 'U'),
                    ('C','C') if reshard else ('U','U')]
                for (label, observed), pair in zip(events, publications):
                    with self.subTest(reshard_after_forward=reshard, window=label):
                        self.assertEqual(set(observed), set(shapes))
                        for name in shapes:
                            published = pair[1 if name.startswith('child.') else 0]
                            self.assertIs(observed[name], canonical[name] if published == 'C' else live[name])
                            if published == 'U':
                                self.assertIsNot(observed[name], canonical[name])
                with self.subTest(reshard_after_forward=reshard, window='after_call'):
                    for name in shapes:
                        self.assertIs(after[name], canonical[name] if reshard else live[name])

    def test_cuda_stored_local_survives_SGD(self):
        import numpy as np
        torch, jt = self.environment()
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        from torch.distributed.device_mesh import init_device_mesh
        if not torch.distributed.is_initialized():
            directory = tempfile.TemporaryDirectory(prefix='fsdp-local-cuda-')
            self.addCleanup(directory.cleanup)
            torch.distributed.init_process_group('nccl', init_method=Path(directory.name, 'rdzv').as_uri(),
                rank=0, world_size=1, timeout=timedelta(seconds=120))
            self.addCleanup(torch.distributed.destroy_process_group)
        self.assertEqual(torch.distributed.get_world_size(), 1)
        self.assertEqual(torch.distributed.get_rank(), 0)
        self.assertEqual(torch.distributed.get_backend(), 'nccl')
        shapes = {'root_scale': (3,), 'frozen': (3,), 'root_linear.weight': (3, 3),
            'root_linear.bias': (3,), 'child.weight': (3, 3), 'child.bias': (3,)}
        expected = {}
        for seed, (name, shape) in enumerate(shapes.items(), 1):
            j = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
            expected[name] = (((j*7+seed*3) % 31-15)/23).astype(np.float32)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.root_scale = torch.nn.Parameter(torch.tensor(expected['root_scale'], device='cuda:0'))
                self.frozen = torch.nn.Parameter(torch.tensor(expected['frozen'], device='cuda:0'), requires_grad=False)
                self.root_linear = torch.nn.Linear(3, 3, device='cuda:0', dtype=torch.float32)
                self.child = torch.nn.Linear(3, 3, device='cuda:0', dtype=torch.float32)
                with torch.no_grad():
                    for name, p in self.named_parameters(): p.copy_(torch.tensor(expected[name], device='cuda:0'))
            def forward(self, x):
                return self.root_linear(self.child(x)) + x*self.root_scale + x*self.frozen
        def descriptor(t):
            return (tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device), int(t.storage_offset()),
                t.is_contiguous(), t.requires_grad, t.is_leaf, t.retains_grad, t.grad_fn is None, t.grad is None)
        def local(t):
            with torch.no_grad(): return t.to_local() if isinstance(t, torch.distributed.tensor.DTensor) else t
        def snapshot(t):
            t = local(t); before = descriptor(t)
            self.assertEqual(t.device, torch.device('cuda:0')); self.assertEqual(t.dtype, torch.float32)
            if jt is not None:
                native = getattr(t, '_var', t); native.sync()
                self.assertEqual(jt.core.dispatch_context([native]), ('cuda', 0))
                self.assertEqual(native.location(), 'device')
            with torch.no_grad(): result = np.array(t.detach().cpu().numpy(), copy=True)
            self.assertEqual(descriptor(t), before); self.assertTrue(np.isfinite(result).all())
            return result
        def compare(t, want):
            got = snapshot(t); want = np.asarray(want, dtype=np.float32)
            self.assertEqual(got.shape, want.shape)
            np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)
        def forward(x, state):
            h = x @ state['child.weight'].T + state['child.bias']
            return h @ state['root_linear.weight'].T + state['root_linear.bias'] + x*state['root_scale'] + x*state['frozen']
        scope = nullcontext() if jt is None else jt.runtime.scope(use_cuda=1, device_id=0,
            backend_fallback='error', cuda_kernel_math='strict')
        with scope:
            model = Model().train(); mesh = init_device_mesh('cuda', (1,))
            policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
            fully_shard(model.child, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
            fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
            parameters = dict(model.named_parameters()); self.assertEqual(set(parameters), set(shapes))
            cached = {name: local(p) for name, p in parameters.items()}
            cached_flags = {name: descriptor(t) for name, t in cached.items()}
            for name, p in parameters.items():
                stored = cached[name]
                self.assertIs(local(p), stored); self.assertIsNot(stored, p)
                self.assertNotIsInstance(stored, torch.distributed.tensor.DTensor)
                self.assertFalse(stored.requires_grad); self.assertTrue(stored.is_leaf)
                self.assertIsNone(stored.grad_fn); self.assertIsNone(stored.grad)
                with torch.enable_grad(): one, two = p.to_local(), p.to_local()
                self.assertIsNot(one, two); self.assertIsNot(one, stored)
                self.assertEqual(one.requires_grad, name != 'frozen')
                self.assertEqual(one.data_ptr(), stored.data_ptr()); self.assertEqual(two.data_ptr(), stored.data_ptr())
                if name != 'frozen':
                    cot = torch.tensor(np.full(shapes[name], 7/23, dtype=np.float32), device='cuda:0')
                    gradient, = torch.autograd.grad(one, (p,), grad_outputs=cot, allow_unused=False)
                    compare(gradient, np.full(shapes[name], 7/23, dtype=np.float32))
                    self.assertIsNone(p.grad)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.125)
            original_slots = tuple(p for g in optimizer.param_groups for p in g['params'])
            x_np = np.array([[3, -2, 7], [-5, 11, 1]], dtype=np.float32)/23
            dy_np = np.array([[-7, 5, 2], [13, -3, 4]], dtype=np.float32)/19
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                state = {name: snapshot(p).astype(np.float64) for name, p in model.named_parameters()}
                x = torch.tensor(x_np, device='cuda:0', requires_grad=True)
                dy = torch.tensor(dy_np, device='cuda:0'); y = model(x); y.backward(dy)
                compare(y, forward(x_np.astype(np.float64), state))
                h = x_np.astype(np.float64) @ state['child.weight'].T + state['child.bias']
                dh = dy_np.astype(np.float64) @ state['root_linear.weight']
                gradients = {'root_scale': (dy_np.astype(np.float64)*x_np).sum(axis=0),
                    'root_linear.weight': dy_np.astype(np.float64).T @ h,
                    'root_linear.bias': dy_np.astype(np.float64).sum(axis=0),
                    'child.weight': dh.T @ x_np.astype(np.float64), 'child.bias': dh.sum(axis=0)}
                self.assertIsNotNone(x.grad)
                compare(x.grad, dh @ state['child.weight'] + dy_np*(state['root_scale']+state['frozen']))
                parameters = dict(model.named_parameters())
                self.assertEqual({name: p.grad is None for name, p in parameters.items()},
                    {name: name == 'frozen' for name in shapes})
                for name, grad in gradients.items(): compare(parameters[name].grad, grad)
                optimizer.step()
                after = dict(model.named_parameters())
                slots = tuple(p for g in optimizer.param_groups for p in g['params'])
                self.assertEqual(len(slots), len(original_slots))
                self.assertTrue(all(a is b for a, b in zip(slots, original_slots)))
                for name in shapes:
                    want = state[name]-0.125*gradients[name] if name in gradients else state[name]
                    compare(after[name], want); compare(cached[name], want)
                    self.assertIs(local(after[name]), cached[name])
                    self.assertEqual(descriptor(cached[name]), cached_flags[name])
                    with torch.enable_grad(): current = after[name].to_local()
                    self.assertEqual(current.data_ptr(), cached[name].data_ptr())
                next_state = {name: snapshot(p).astype(np.float64) for name, p in after.items()}
                compare(model(torch.tensor(x_np, device='cuda:0')), forward(x_np.astype(np.float64), next_state))
                np.testing.assert_array_equal(snapshot(x), x_np); np.testing.assert_array_equal(snapshot(dy), dy_np)


    def test_cuda_grad_fn_view_history_exact_runtime_error(self):
        """A detected stale no-grad view raises the exact Torch public type."""
        torch, jt = self.environment()
        scope = nullcontext() if jt is None else jt.runtime.scope(use_cuda=1, device_id=0,
            backend_fallback='error', cuda_kernel_math='strict')
        with scope:
            for writer_kind in ('data', 'detach'):
                with self.subTest(writer_kind=writer_kind):
                    base = torch.tensor([[1., 2.], [3., 4.]], device='cuda:0',
                        dtype=torch.float32, requires_grad=True)
                    with torch.no_grad():
                        held = base.transpose(0, 1)
                    writer = base.data if writer_kind == 'data' else base.detach()
                    with torch.no_grad():
                        self.assertIs(writer.copy_(writer), writer)
                    self.assertEqual(base.device, torch.device('cuda:0'))
                    self.assertEqual(writer.data_ptr(), base.data_ptr())
                    if writer_kind == 'data':
                        self.assertIsNone(held.grad_fn)
                        self.assertTrue(held.is_leaf)
                    else:
                        with self.assertRaises(RuntimeError) as caught:
                            _ = held.grad_fn
                        self.assertIs(type(caught.exception), RuntimeError)
                        self.assertIn('view was created in no_grad mode', str(caught.exception))
                        if jt is not None:
                            self.assertIs(type(caught.exception.__cause__), jt.JittorError)
                            self.assertEqual(caught.exception.args, caught.exception.__cause__.args)
            # Successful metadata/interning is still the original graph owner.
            value = torch.tensor([1., -2., 3.], device='cuda:0', dtype=torch.float32,
                requires_grad=True)
            output = (value*value).sum()
            torch.cuda.synchronize()
            self.assertEqual(output.device, torch.device('cuda:0'))
            producer = output.grad_fn
            self.assertIsNotNone(producer)
            self.assertIs(output.grad_fn, producer)

    def test_cuda_no_grad_view_history_guard_remains(self):
        torch, jt = self.environment()
        scope = nullcontext() if jt is None else jt.runtime.scope(use_cuda=1, device_id=0,
            backend_fallback='error', cuda_kernel_math='strict')
        with scope:
            base = torch.tensor([1., -2., 3.], device='cuda:0', requires_grad=True)
            with torch.no_grad(): view = base.view(3)
            with torch.no_grad(): base.add_(0.125)
            with self.assertRaisesRegex(RuntimeError, 'view was created in no_grad mode'):
                _ = view.is_leaf
