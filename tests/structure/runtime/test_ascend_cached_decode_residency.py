"""Decode setup must finish device work before residency checks and timing."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('lazy_runtime', [False, True])
def test_cached_decode_materializes_setup_before_residency_and_timing(lazy_runtime):
    path = ROOT / 'benchmarks/profiling/benchmark_qwen3_ascend.py'
    source = ast.parse(path.read_text())
    functions = {node.name: node for node in ast.walk(source)
                 if isinstance(node, ast.FunctionDef)}
    scope = {}
    module = ast.Module(body=[functions['_assert_npu_values'],
                              functions['run_cached_decode']], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    values, syncs, admissions, casts = [], [], [], []
    device = SimpleNamespace(type='npu', index=0)

    class Value:
        dtype = 'int64'

        def __init__(self, ready=False):
            self.ready = ready
            self.device = device
            values.append(self)

        def __getitem__(self, key):
            return self

        def argmax(self, **kwargs):
            return Value()

        def to(self, **kwargs):
            assert kwargs == {'copy': True, 'dtype': 'float32', 'device': device}
            casts.append(kwargs)
            return Value()

        def location(self):
            return 'device' if self.ready else 'unavailable'

    class SetupAudited(Exception):
        pass

    def synchronize():
        syncs.append(True)
        for value in values:
            value.ready = True

    actual_assert = scope['_assert_npu_values']
    jt = (SimpleNamespace(core=SimpleNamespace(dispatch_context=lambda tensors: ('acl', 0)))
          if lazy_runtime else None)
    torch = SimpleNamespace(float32='float32', npu=SimpleNamespace(current_device=lambda: 0),
                            no_grad=nullcontext, ones=lambda *args, **kwargs: Value())
    # Native backend metadata alone must never waive the strict location gate.
    if lazy_runtime:
        with pytest.raises(RuntimeError, match='lacks native ACL residency'):
            actual_assert(torch, [('unmaterialized', Value())], jt)

    def assert_setup(framework, items, backend):
        assert actual_assert(framework, items, backend) == 3
        assert syncs == [True]
        assert len(casts) == 1
        assert [label for label, value in items] == [
            'decode_initial_token', 'decode_mask/0', 'decode_mask/1']
        assert admissions == []
        raise SetupAudited

    def forbid_timer():
        raise AssertionError('setup entered the measured decode region')

    prefill = SimpleNamespace(logits=Value(ready=True), past_key_values=object())
    scope.update(input_slots=[(Value(ready=True), Value(ready=True))],
                 cached_decode_processor=lambda input_ids, scores: scores,
                 torch=torch, jt=jt, run_prefill=lambda slot: (prefill, 0., 0.),
                 args=SimpleNamespace(batch_size=1, new_tokens=3), prompt_tokens=8,
                 attention_mask=Value(ready=True), device=device,
                 control=SimpleNamespace(phase='unused', before=lambda *args: admissions.append(args)),
                 synchronize=synchronize, _assert_npu_values=assert_setup,
                 time=SimpleNamespace(perf_counter=forbid_timer))
    with pytest.raises(SetupAudited):
        scope['run_cached_decode'](0, 'first_decode')
