"""Canonical installation-owned NN types and real CUDA FSDP mixin behavior."""
import importlib
import os
import pickle
import sys
import tempfile
import unittest

import pytest

pytestmark = pytest.mark.cuda


class TestCanonicalNNTypesCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.jt = sys.modules.get('jittor')
        if cls.jt is None:
            assert hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context')
        else:
            assert hasattr(torch, '_torch_compat_install_context')
            assert cls.jt.compiler.has_cuda and cls.jt.compiler.is_cuda

    def setUp(self):
        if not self.torch.cuda.is_available():
            self.skipTest('CUDA hardware required; a skip is not acceptance')
        self.torch.cuda.set_device(0)
        anchor = self.torch.ones((1,), device='cuda:0')
        self.torch.cuda.synchronize()
        self.assertEqual(anchor.device.type, 'cuda')
        if self.jt is not None:
            self.assertEqual(self.jt.flags.backend_fallback, 'error')
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def tearDown(self):
        self.torch.cuda.synchronize()
        if self.jt is not None:
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def test_same_object_public_imports_and_pickle_identity(self):
        expected = dict(Module='module', Linear='linear', GELU='activation',
                        LayerNorm='normalization', Conv1d='conv', Conv3d='conv',
                        Embedding='sparse', Sequential='container', ModuleList='container')
        original_native = ({} if self.jt is None else
                           {name: getattr(self.jt.nn, name).__module__ for name in expected})
        for name, family in expected.items():
            with self.subTest(class_name=name):
                cls = getattr(self.torch.nn, name)
                self.assertEqual(cls.__module__, 'torch.nn.modules.' + family)
                self.assertEqual(cls.__qualname__, name)
                owner = importlib.import_module(cls.__module__)
                self.assertIs(getattr(owner, name), cls)
                self.assertIs(getattr(self.torch.nn.modules, name), cls)
                self.assertIs(pickle.loads(pickle.dumps(cls)), cls)
                if self.jt is not None:
                    self.assertIsNot(cls, getattr(self.jt.nn, name))
                    self.assertEqual(getattr(self.jt.nn, name).__module__, original_native[name])

    def test_modulelist_is_not_sequential_and_preserves_registered_children(self):
        torch = self.torch
        self.assertIsNot(torch.nn.ModuleList, torch.nn.Sequential)
        with torch.device('cuda:0'):
            a, b = torch.nn.Linear(4, 4), torch.nn.Linear(4, 2)
            layers = torch.nn.ModuleList([a, b])
        self.assertEqual(list(layers), [a, b])
        self.assertEqual(list(layers.named_children()), [('0', a), ('1', b)])
        sliced = layers[:1]
        self.assertIs(type(sliced), torch.nn.ModuleList)
        self.assertIsNot(sliced, layers)
        self.assertIs(sliced[0], a)
        self.assertTrue(all(p.device.type == 'cuda' for p in layers.parameters()))
        with self.assertRaises(NotImplementedError):
            layers(torch.ones((2, 4), device='cuda:0'))

    def test_fsdp_public_mixin_is_physically_in_mro(self):
        torch = self.torch
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard, FSDPModule
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
        self.assertFalse(dist.is_initialized(), 'standalone singleton process required')
        run = tempfile.mkdtemp(prefix='canonical-nn-fsdp-', dir=os.environ.get('TMPDIR'))
        dist.init_process_group('nccl', init_method='file://' + run + '/store',
                                rank=0, world_size=1, device_id=torch.device('cuda:0'))
        try:
            mesh = init_device_mesh('cuda', (1,))
            with torch.device('cuda:0'):
                module = torch.nn.Linear(4, 2)
            full_state = {name: value.detach().clone()
                          for name, value in module.state_dict().items()}
            expected_state = {name: value.detach().clone()
                              for name, value in full_state.items()}
            original_class = type(module)
            original_parameter_names = tuple(name for name, _ in module.named_parameters())
            result = fully_shard(module, mesh=mesh)
            self.assertIs(result, module)
            self.assertIn(FSDPModule, type(module).__mro__)
            self.assertIn(original_class, type(module).__mro__)
            self.assertTrue(type(module).__name__.startswith('FSDP'))
            self.assertEqual(type(module).__module__, 'torch.distributed.fsdp._fully_shard._fully_shard')
            # A virtual isinstance marker alone cannot satisfy either MRO check.
            self.assertIsInstance(module, FSDPModule)
            self.assertEqual(tuple(name for name, _ in module.named_parameters()), original_parameter_names)
            module.to(device=0, non_blocking=True)
            with torch.no_grad():
                for parameter in module.parameters():
                    parameter.fill_(0)
            loaded = set_model_state_dict(module, full_state, options=StateDictOptions(
                full_state_dict=True, cpu_offload=False, broadcast_from_rank0=True))
            self.assertEqual(loaded.missing_keys, [])
            self.assertEqual(loaded.unexpected_keys, [])
            for name, parameter in module.named_parameters():
                self.assertTrue(torch.equal(parameter.to_local(), expected_state[name]))
            value = torch.ones((2, 4), device='cuda:0', requires_grad=True)
            output = module(value)
            self.assertEqual(output.device.type, 'cuda')
            self.assertEqual(tuple(output.shape), (2, 2))
            output.sum().backward()
            self.assertIsNotNone(value.grad)
            self.assertEqual(value.grad.device.type, 'cuda')
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(parameter.grad.device.type, 'cuda')
            torch.cuda.synchronize()
        finally:
            dist.destroy_process_group()

    def test_registered_parameter_and_buffer_lookup_cuda(self):
        torch = self.torch
        with torch.device('cuda:0'):
            net = torch.nn.Module()
            net.left = torch.nn.Linear(2, 2, bias=False)
            net.right = torch.nn.Linear(2, 2, bias=False)
            net.right.weight = net.left.weight
            net.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
            net.register_parameter('absent_parameter', None)
            net.register_buffer('first', torch.tensor([1., 2.]))
            net.register_buffer('second', net.first)
            net.register_buffer('absent_buffer', None)
            net.ordinary = torch.ones(2)
        self.assertIs(net.get_parameter('left.weight'), net.left.weight)
        self.assertIs(net.get_parameter('right.weight'), net.left.weight)
        self.assertIs(net.get_parameter('frozen'), net.frozen)
        self.assertEqual([name for name, _ in net.named_parameters()],
                         ['frozen', 'left.weight'])
        self.assertEqual([name for name, _ in net.named_parameters(remove_duplicate=False)],
                         ['frozen', 'left.weight', 'right.weight'])
        self.assertIs(net.get_buffer('first'), net.first)
        self.assertIs(net.get_buffer('second'), net.first)
        self.assertIsNone(net.get_buffer('absent_buffer'))
        self.assertEqual([name for name, _ in net.named_buffers()], ['first'])
        self.assertEqual([name for name, _ in net.named_buffers(remove_duplicate=False)],
                         ['first', 'second'])
        for name in ('first', 'ordinary', 'absent_parameter', 'missing'):
            with self.assertRaises(AttributeError):
                net.get_parameter(name)
        for name in ('frozen', 'ordinary', 'missing'):
            with self.assertRaises(AttributeError):
                net.get_buffer(name)
        with torch.no_grad():
            net.get_parameter('right.weight').fill_(2.)
        net.get_buffer('second').fill_(3.)
        self.assertTrue(torch.equal(net.first, torch.full((2,), 3., device='cuda:0')))
        x = torch.tensor([[1., 3.]], device='cuda:0', requires_grad=True)
        y = net.left(x) + net.right(x)
        self.assertTrue(torch.equal(y, torch.tensor([[16., 16.]], device='cuda:0')))
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([[8., 8.]], device='cuda:0')))
        self.assertTrue(torch.equal(net.left.weight.grad,
                                   torch.tensor([[2., 6.], [2., 6.]], device='cuda:0')))
        self.assertIsNone(net.frozen.grad)
        for value in (x, y, x.grad, net.left.weight, net.left.weight.grad,
                      net.frozen, net.first, net.second, net.ordinary):
            self.assertEqual(value.device, torch.device('cuda:0'))
            if self.jt is not None:
                value.sync(True, False)
                self.assertEqual((value.location(), value.device_id,
                                  value.placement_backend), ('device', 0, 1))
                self.assertEqual(self.jt.core.dispatch_context([value]), ('cuda', 0))

    def test_public_boolean_reductions_and_allclose_cuda(self):
        torch = self.torch
        value = torch.tensor([[True, False], [True, True]],
                             device='cuda:0', dtype=torch.bool)
        cases = ((value.all(), False), (value.any(), True),
                 (value.all(dim=1, keepdim=True), [[False], [True]]),
                 (value.any(dim=0), [True, True]),
                 (torch.empty((0,), device='cuda:0', dtype=torch.bool).all(), True),
                 (torch.empty((0,), device='cuda:0', dtype=torch.bool).any(), False))
        for result, expected in cases:
            self.assertIs(type(result), torch.Tensor)
            self.assertIs(result.dtype, torch.bool)
            self.assertEqual(result.device, torch.device('cuda:0'))
            self.assertTrue(torch.equal(result,
                torch.tensor(expected, device='cuda:0', dtype=torch.bool)))
            if self.jt is not None:
                result.sync(True, False)
                self.assertEqual(self.jt.core.dispatch_context([result]), ('cuda', 0))
        source = torch.tensor([1., -2., 3.], device='cuda:0')
        close = torch.tensor([1.000001, -2., 3.], device='cuda:0')
        far = torch.tensor([1.001, -2., 3.], device='cuda:0')
        for other, expected in ((source, True), (close, True), (far, False)):
            result = torch.allclose(source, other, atol=1e-5, rtol=1e-5)
            self.assertIs(type(result), bool)
            self.assertIs(result, expected)
        self.assertTrue(torch.equal(source,
            torch.tensor([1., -2., 3.], device='cuda:0')))

    def test_public_six_richcomparisons_cuda(self):
        import operator
        torch = self.torch
        a_values, b_values = (-1., 0., 2.), (0., 0., 1.)
        a = torch.tensor(a_values, device='cuda:0')
        b = torch.tensor(b_values, device='cuda:0')
        for name in ('lt', 'le', 'gt', 'ge', 'eq', 'ne'):
            compare = getattr(operator, name)
            for side, left, right, lh, rh in (
                    ('pair', a, b, a_values, b_values),
                    ('scalar_right', a, 0., a_values, (0.,) * 3),
                    ('scalar_left', 0., a, (0.,) * 3, a_values)):
                with self.subTest(comparison=name, operand=side):
                    result = compare(left, right)
                    self.assertIs(type(result), torch.Tensor)
                    self.assertIs(result.dtype, torch.bool)
                    self.assertEqual(tuple(result.shape), (3,))
                    self.assertEqual(result.device, torch.device('cuda:0'))
                    expected = torch.tensor([compare(x, y) for x, y in zip(lh, rh)],
                                            dtype=torch.bool, device='cuda:0')
                    self.assertTrue(torch.equal(result, expected))
                    if self.jt is not None:
                        result.sync(True, False)
                        self.assertEqual(self.jt.core.dispatch_context([result]), ('cuda', 0))
                        self.assertEqual((result.location(), result.device_id,
                                          result.placement_backend), ('device', 0, 1))
        self.assertTrue(torch.equal(a, torch.tensor(a_values, device='cuda:0')))
        self.assertTrue(torch.equal(b, torch.tensor(b_values, device='cuda:0')))

    def test_public_equal_real_tensor_values_and_errors(self):
        torch = self.torch
        value = torch.tensor([1., 2., 3.], device='cuda:0')
        clone = value.clone()
        unequal = torch.tensor([1., 2., 4.], device='cuda:0')
        self.assertTrue(torch.equal(value, clone))
        self.assertFalse(torch.equal(value, unequal))
        self.assertFalse(torch.equal(value, value.reshape(1, 3)))
        self.assertTrue(torch.equal(torch.empty((0,), device='cuda:0'),
                                    torch.empty((0,), device='cuda:0')))
        nan = torch.tensor([float('nan')], device='cuda:0')
        # Real Torch CUDA accepts aliases, but an independent NaN copy differs.
        self.assertTrue(torch.equal(nan, nan))
        self.assertTrue(torch.equal(nan, nan.detach()))
        self.assertTrue(torch.equal(nan, nan.view(1)))
        self.assertFalse(torch.equal(nan, nan.clone()))
        with self.assertRaises(TypeError):
            torch.equal(value, [1., 2., 3.])
        for tensor in (value, clone, unequal, nan):
            self.assertEqual(tensor.device.type, 'cuda')
            self.assertEqual(tensor.device.index, 0)
            if self.jt is not None:
                tensor.sync()
                self.assertEqual(self.jt.core.dispatch_context([tensor]), ('cuda', 0))
                self.assertEqual(tensor.location(), 'device')


if __name__ == '__main__':
    unittest.main()
