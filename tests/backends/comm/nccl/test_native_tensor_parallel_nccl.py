"""Real WORLD CUDA TP linear storage, adjoints, training and native restore."""

import ctypes
import datetime
import json
import os
from pathlib import Path
import sys

import jittor as jt
from jittor import distributed as dist
from jittor.distributed import checkpoint
from jittor.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from jittor._runtime.fallback import forbid_backend_fallbacks
import numpy as np
import pytest
from _helpers import native_tp as case


pytestmark = pytest.mark.skipif(not jt.compiler.has_cuda or dist.get_world_size() != 2,
                              reason="requires two real CUDA/NCCL WORLD ranks")


@pytest.fixture(autouse=True)
def _world():
    assert jt.compile_extern.get_library('nccl', load=True) is not None
    if dist.get_default_store() is None:
        dist.init_process_group('nccl', init_method='env://',
                                timeout=datetime.timedelta(seconds=60))
    assert dist.get_world_size() == 2 and dist.get_backend() == 'nccl'
    assert 'torch' not in sys.modules


def _device(value):
    value.sync()
    assert str(value.dtype) == "float32"
    assert jt.core.dispatch_context([value]) == ('cuda', 0)
    assert value.location() == 'device'
    ordinal = ctypes.c_int(-1)
    assert ctypes.CDLL('libcuda.so.1').cuPointerGetAttribute(
        ctypes.byref(ordinal), 9, ctypes.c_void_p(value.device_raw_ptr)) == 0
    assert ordinal.value == 0


@pytest.mark.parametrize('gather', [False, True])
@pytest.mark.parametrize('shape', [(4, 3), (2, 2, 3)])
def test_native_tp_two_rank_training_and_fresh_resume(gather, shape, monkeypatch):
    before = jt.core.backend_fallback_count()
    rank = dist.get_rank()
    expected = case.reference(shape=shape)
    root = Path(os.environ['JITTOR_TP_TEST_DIR']) / ('{}-{}'.format(gather, len(shape)))
    # Count the actual legacy owner calls: a TP shard must never take the
    # data-parallel reduction or periodic root-parameter broadcast path.
    from jittor.distributed import process_group
    original = process_group._optimizer_legacy_collectives
    calls = {'reduce': 0, 'broadcast': 0}
    def wrapped():
        reduce, broadcast = original()
        def counted_reduce(value):
            calls['reduce'] += 1
            return reduce(value)
        def counted_broadcast(value):
            calls['broadcast'] += 1
            return broadcast(value)
        return counted_reduce, counted_broadcast
    monkeypatch.setattr(process_group, '_optimizer_legacy_collectives', wrapped)
    with forbid_backend_fallbacks():
        model = case.Model(gather)
        opt = case.optimizer(model)
        assert model.column.weight.shape == [3, 3]
        assert model.column.bias.shape == [3]
        assert model.row.weight.shape == [4, 3]
        assert model.row.bias.shape == [4]
        assert sum(int(p.numel()) for p in model.parameters()) == 28  # 52 unsharded
        # There is no full weight Var stored on either module.
        for layer in (model.column, model.row):
            assert len(layer.parameters()) == 2
            for parameter in layer.parameters():
                _device(parameter)
        for n in range(3):
            case.assert_trace(case.step(model, opt, n, shape, _device), expected[n], model)
        checkpoint.save_training_state(model, opt, root, extra={'step': 3})
        continued = [case.step(model, opt, n, shape, _device) for n in (3, 4)]
        fresh = case.Model(gather, initialize=False)
        resumed_opt = case.optimizer(fresh)
        assert checkpoint.load_training_state(fresh, resumed_opt, root) == {'step': 3}
        manifest = checkpoint.read_manifest(root)
        assert manifest['world_size'] == 2
        for n, exact in zip((3, 4), continued):
            trace = case.step(fresh, resumed_opt, n, shape, _device)
            case.assert_trace(trace, exact, fresh, exact=True)
            case.assert_trace(trace, expected[n], fresh)
    assert calls == {'reduce': 0, 'broadcast': 0}
    assert jt.core.backend_fallback_count() == before == 0
    print('NATIVE_TP_RESULT', json.dumps({'rank': rank, 'gather': gather, 'shape': shape,
          'parameters_per_rank': 28, 'unsharded_parameters': 52,
          'exact_resume': True, 'fallbacks': 0, 'legacy_calls': calls}), flush=True)


@pytest.mark.parametrize('kind', ['column', 'row'])
def test_native_tp_individual_linear_adjoint_and_divisibility(kind):
    before = jt.core.backend_fallback_count()
    rank = dist.get_rank()
    with forbid_backend_fallbacks():
        layer = (ColumnParallelLinear(4, 6, gather_output=True) if kind == 'column'
                 else RowParallelLinear(4, 6, input_is_parallel=False))
        values = np.arange(24, dtype=np.float32).reshape(6, 4) / 25 - .3
        layer.weight.update(jt.array(case.local(values, layer.weight), dtype='float32'))
        layer.bias.update(jt.array(case.local(np.arange(6, dtype=np.float32)/30, layer.bias), dtype='float32'))
        x = jt.array(np.arange(16, dtype=np.float32).reshape(4, 4)/10, dtype='float32').start_grad()
        output = layer(x)
        _device(output)
        input_numpy = np.arange(16, dtype=np.float32).reshape(4, 4)/10
        expected = input_numpy @ values.T + np.arange(6, dtype=np.float32)/30
        np.testing.assert_allclose(output.numpy(), expected, atol=2e-6, rtol=2e-5)
        coefficients = np.arange(24, dtype=np.float32).reshape(4, 6)/20 - .2
        dx, dw, db = jt.grad((output * jt.array(coefficients)).sum(), [x, layer.weight, layer.bias])
        jt.sync([dx, dw, db], True, False)
        for value in (dx, dw, db):
            _device(value)
        np.testing.assert_allclose(dx.numpy(), coefficients @ values, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(dw.numpy(), case.local(coefficients.T @ input_numpy, layer.weight), atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(db.numpy(), case.local(coefficients.sum(0), layer.bias), atol=2e-6, rtol=2e-5)
        constructor = ColumnParallelLinear if kind == 'column' else RowParallelLinear
        with pytest.raises(ValueError, match='divide WORLD'):
            constructor(3, 3)
    assert jt.core.backend_fallback_count() == before == 0


def test_native_tp_checkpoint_rejects_replicated_bias_drift():
    before = jt.core.backend_fallback_count()
    root = Path(os.environ['JITTOR_TP_TEST_DIR']) / 'replica-drift'
    with forbid_backend_fallbacks():
        model = case.Model()
        opt = case.optimizer(model)
        model.row.bias.update(model.row.bias + dist.get_rank() / 10)
        with pytest.raises(checkpoint.CheckpointError, match=r'replicas differ at /model/row.bias'):
            checkpoint.save_training_state(model, opt, root)
        assert not (root / 'manifest.json').exists()
    assert jt.core.backend_fallback_count() == before == 0


def test_native_tp_checkpoint_rejects_different_global_layouts():
    before = jt.core.backend_fallback_count()
    root = Path(os.environ['JITTOR_TP_TEST_DIR']) / 'layout-drift'
    class Model(jt.Module):
        def __init__(self):
            self.layer = ColumnParallelLinear(4, 6 + 2 * dist.get_rank(), bias=False)
    with forbid_backend_fallbacks():
        model = Model()
        opt = case.optimizer(model)
        with pytest.raises(checkpoint.CheckpointError, match='replicated metadata differs'):
            checkpoint.save_training_state(model, opt, root)
        assert not (root / 'manifest.json').exists()
    assert jt.core.backend_fallback_count() == before == 0
