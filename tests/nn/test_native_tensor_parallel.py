"""Native TP singleton semantics; real partition/communication lives in NCCL tests."""

import numpy as np
import pytest
import jittor as jt
from jittor.distributed import checkpoint
from jittor.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from _helpers import native_tp as case
from _helpers.child_process import run_python_child


@pytest.mark.parametrize("gather", [False, True])
@pytest.mark.parametrize("shape", [(4, 3), (2, 2, 3)])
def test_native_tp_cpu_training_and_fresh_resume(tmp_path, gather, shape):
    model = case.Model(gather)
    opt = case.optimizer(model)
    expected = case.reference(shape=shape)
    for n in range(3):
        case.assert_trace(case.step(model, opt, n, shape), expected[n], model)
    checkpoint.save_training_state(model, opt, tmp_path / 'train', extra={"step": 3})
    continued = [case.step(model, opt, n, shape) for n in (3, 4)]
    fresh = case.Model(gather, initialize=False)
    resumed_opt = case.optimizer(fresh)
    assert checkpoint.load_training_state(fresh, resumed_opt, tmp_path / 'train') == {"step": 3}
    for n, exact in zip((3, 4), continued):
        trace = case.step(fresh, resumed_opt, n, shape)
        case.assert_trace(trace, exact, fresh, exact=True)
        case.assert_trace(trace, expected[n], fresh)


@pytest.mark.parametrize("layer", [ColumnParallelLinear, RowParallelLinear])
def test_native_tp_feature_and_binding_validation(layer):
    with pytest.raises(ValueError):
        layer(0, 2)
    module = layer(3, 6, bias=False)
    with pytest.raises(ValueError):
        module(jt.ones((2, 4)))
    copied = module.partition()
    copied['weight']['shape'] = (999,)
    assert module.partition()['weight']['shape'] == (6, 3)
    module.weight.update(jt.ones((5, 3)))
    with pytest.raises(ValueError, match='layout'):
        module(jt.ones((2, 3)))


def test_native_tp_checkpoint_rejects_same_shape_wrong_orientation(tmp_path):
    class Wrapped(jt.Module):
        def __init__(self, layer):
            self.layer = layer(4, 4)
    source = Wrapped(ColumnParallelLinear)
    source_opt = case.optimizer(source)
    checkpoint.save_training_state(source, source_opt, tmp_path / 'layout')
    target = Wrapped(RowParallelLinear)
    before = {k: v.numpy().copy() for k, v in target.state_dict().items()}
    with pytest.raises(ValueError, match='tensor-parallel ownership'):
        checkpoint.load_training_state(target, case.optimizer(target), tmp_path / 'layout')
    for key, value in target.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), before[key])


def test_native_tp_public_owner_and_reject_extra_optimizer_input():
    from jittor.nn import parallel
    assert parallel.ColumnParallelLinear is ColumnParallelLinear
    assert parallel.RowParallelLinear is RowParallelLinear
    model = case.Model()
    opt = case.optimizer(model)
    value = jt.ones((2, 3)).start_grad()
    opt.set_input_into_param_group(value)
    with pytest.raises(NotImplementedError, match='explicit jt.grad'):
        opt.backward(model(value).sum())


def test_native_tp_public_exports_do_not_load_owner_during_core_import():
    # A fresh process matters: this test module itself imports the TP owner.
    run_python_child(["-c", """
import sys
import jittor
from jittor.nn import parallel
assert 'jittor.distributed.tensor_parallel' not in sys.modules
from jittor.nn.parallel import ColumnParallelLinear, RowParallelLinear
from jittor.distributed import tensor_parallel
assert ColumnParallelLinear is tensor_parallel.ColumnParallelLinear
assert RowParallelLinear is tensor_parallel.RowParallelLinear
try:
    parallel.unknown_wrapper
except AttributeError:
    pass
else:
    raise AssertionError('unknown parallel export must raise AttributeError')
"""], check=True)


def test_native_tp_cpu_bf16_bias_input_is_quantized():
    from jittor._runtime.fallback import forbid_backend_fallbacks
    with forbid_backend_fallbacks():
        case.assert_row_bf16_bias()


@pytest.mark.parametrize("gather", [False, True])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_native_tp_cpu_adamw_full_torch_oracle_and_resume(tmp_path, gather, precision, monkeypatch):
    from _helpers import native_tp_mixed
    native_tp_mixed.exercise(tmp_path, gather, precision, monkeypatch)
