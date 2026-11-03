"""Pure guards for the opt-in generation indexing diagnostic."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[2]/'agent/skills/torch-compat-repo-runbook/scripts/trl_generation_index_probe.py'
    spec = importlib.util.spec_from_file_location('generation_index_probe_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_must_be_external_and_fresh(probe, tmp_path):
    root = tmp_path/'repo'
    root.mkdir()
    for path in (root, root/'out'):
        with pytest.raises(ValueError, match='outside'):
            probe.external_output(root, path)
    output = probe.external_output(root, tmp_path/'external')
    with pytest.raises(ValueError, match='exists'):
        probe.external_output(root, output)


@pytest.mark.parametrize('dtype', ['int32', 'int64'])
def test_range_check_accepts_public_grids_and_negative_indices(probe, dtype):
    import numpy as np
    probe.check_indices((1, 4), [np.zeros((1, 1, 1, 1), dtype=dtype),
                               np.array([[[[-4, -1, 0, 3]]]], dtype=dtype)])


@pytest.mark.parametrize('values', [[4], [-5], [1061036018]])
def test_range_check_stops_bad_indices_before_native_trap(probe, values):
    import numpy as np
    with pytest.raises(ValueError, match='out of bounds'):
        probe.check_indices((1, 4), [np.array([0], dtype='int32'), np.array(values, dtype='int32')])


@pytest.mark.parametrize('dtype', ['float32', 'bool', 'uint8'])
def test_range_check_rejects_wrong_dtype(probe, dtype):
    import numpy as np
    with pytest.raises(ValueError, match='int32/int64'):
        probe.check_indices((1, 4), [np.array([0], dtype='int32'), np.array([0], dtype=dtype)])


def test_range_check_is_bounded(probe):
    import numpy as np
    with pytest.raises(ValueError, match='bounded'):
        probe.check_indices((1, 100), [np.zeros(1, dtype='int64'), np.arange(65, dtype='int64')])


@pytest.fixture
def fake_torch():
    import numpy as np

    class Tensor:
        reads = 0

        def __init__(self, values):
            self.values = np.asarray(values)
            self.shape = self.values.shape
            self.dtype = 'torch.'+str(self.values.dtype)
            self.device = 'cuda:0'

        def numel(self):
            return self.values.size

        def stride(self):
            return [s//self.values.itemsize for s in self.values.strides]

        def __getitem__(self, index):
            return Tensor(self.values[tuple(value.values for value in index)])

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            Tensor.reads += 1
            return self.values

    return SimpleNamespace(Tensor=Tensor, is_tensor=lambda value: isinstance(value, Tensor))


def mask_inputs(torch):
    import numpy as np
    return torch.Tensor(np.array([[True, False, True, True]])), (
        torch.Tensor(np.zeros((1, 1, 1, 1), dtype='int32')),
        torch.Tensor(np.arange(4, dtype='int32').reshape(1, 1, 1, 4)))


def events(path):
    return [json.loads(line) for line in (path/'index-events.jsonl').read_text().splitlines()]


def test_metadata_mode_never_reads_values_and_restores_owner(probe, fake_torch, tmp_path):
    original = fake_torch.Tensor.__getitem__
    target, index = mask_inputs(fake_torch)
    with probe.IndexObserver(fake_torch, tmp_path, 'metadata', 'torch') as observer:
        actual = target[index]
        assert observer.selected == 1 and actual.shape == (1, 1, 1, 4)
    assert fake_torch.Tensor.__getitem__ is original and fake_torch.Tensor.reads == 0
    assert [event['kind'] for event in events(tmp_path)] == ['installed', 'before_getitem', 'after_getitem']


def test_values_mode_checks_actual_operand_gather(probe, fake_torch, tmp_path):
    target, index = mask_inputs(fake_torch)
    with probe.IndexObserver(fake_torch, tmp_path, 'values', 'torch'):
        target[index]
    assert fake_torch.Tensor.reads == 4
    assert events(tmp_path)[-1]['kind'] == 'checked_getitem'


def test_bad_values_are_recorded_and_error_is_not_swallowed(probe, fake_torch, tmp_path):
    import numpy as np
    original = fake_torch.Tensor.__getitem__
    target, index = mask_inputs(fake_torch)
    index = (index[0], fake_torch.Tensor(np.array([1061036018], dtype='int32')))
    with pytest.raises(ValueError, match='out of bounds'):
        with probe.IndexObserver(fake_torch, tmp_path, 'values', 'torch'):
            target[index]
    assert fake_torch.Tensor.__getitem__ is original
    assert events(tmp_path)[-1]['kind'] == 'first_failure'
    assert events(tmp_path)[-1]['phase'] == 'index_range_check'


def test_constructor_failure_restores_owner_and_records_phase(probe, fake_torch, tmp_path, monkeypatch):
    def failing(target, index):
        raise RuntimeError('injected constructor failure')

    monkeypatch.setattr(fake_torch.Tensor, '__getitem__', failing)
    target, index = mask_inputs(fake_torch)
    with pytest.raises(RuntimeError, match='injected'):
        with probe.IndexObserver(fake_torch, tmp_path, 'metadata', 'torch'):
            target[index]
    assert fake_torch.Tensor.__getitem__ is failing
    assert events(tmp_path)[-1]['phase'] == 'gather_construction'


def test_observer_scope_excludes_other_tensors(probe, fake_torch, tmp_path):
    import numpy as np
    observer = probe.IndexObserver(fake_torch, tmp_path, 'metadata', 'torch')
    target, index = mask_inputs(fake_torch)
    assert observer.selected_indices(target, index) is not None
    assert observer.selected_indices(fake_torch.Tensor(np.ones((1, 4), dtype='float32')), index) is None
    assert observer.selected_indices(target, (index[0], slice(None))) is None
    assert observer.selected_indices(target, [*index]) is None


def test_cli_requires_explicit_observation_for_ppo(probe):
    with pytest.raises(ValueError, match='explicit observation'):
        probe.main(['ppo', '--root', '/', '--runtime', 'jittor', '--device', 'cuda', '--output', '/unused'])


@pytest.mark.parametrize('overrides', [dict(mode='ppo'), dict(runtime='torch'),
                                     dict(device='cpu'), dict(observe='values'),
                                     dict(observe='metadata'), dict(boundary_stress=False)])
def test_required_async_scope_is_fail_closed(probe, overrides):
    options = dict(mode='mask', runtime='jittor', device='cuda', observe='off',
                   boundary_stress=True, require_async=True)
    options.update(overrides)
    with pytest.raises(ValueError, match='requires|requires unobserved'):
        probe.check_stress_options(SimpleNamespace(**options))


def test_async_guard_requires_an_actual_new_worker_batch(probe):
    with pytest.raises(ValueError, match='actual async worker'):
        probe.check_async_coverage(7, 7, True)
    probe.check_async_coverage(7, 8, True)
    probe.check_async_coverage(7, 7, False)


def test_default_stress_options_preserve_the_original_modes(probe):
    for mode in ('mask', 'ppo'):
        probe.check_stress_options(SimpleNamespace(mode=mode, runtime='jittor', device='cuda',
                                  observe='off', boundary_stress=False, require_async=False))


def test_genuine_oracle_rejects_shim_and_python_placeholder(probe, tmp_path):
    with pytest.raises(ValueError, match='genuine'):
        probe.genuine_oracle(SimpleNamespace(_torch_compat_install_context=True))
    path = tmp_path/'fake.py'
    path.write_text('not a compiled oracle')
    with pytest.raises(ValueError, match='compiled'):
        probe.genuine_oracle(SimpleNamespace(_C=SimpleNamespace(__file__=str(path))))


def test_oracle_source_binding_does_not_depend_on_unused_jittor_code(probe):
    root = Path(__file__).resolve().parents[2]
    oracle = probe.source_binding(SimpleNamespace(root=root, runtime='torch', mode='mask'))
    candidate = probe.source_binding(SimpleNamespace(root=root, runtime='jittor', mode='mask'))
    assert set(oracle['owners']) == {'compat/tests/torch/_ecosystem_runner.py'}
    assert 'python/jittor/nn/functional/loss.py' in candidate['owners']


@pytest.fixture
def reference(probe, tmp_path):
    import numpy as np
    arrays = {str(n)+'-'+str(p)+'/'+kind: np.ones((1, 1, n, n), dtype=('bool' if kind == 'bool' else 'float32'))
              for n in (3, 4, 5, 6) for p in (0, 1) for kind in ('bool', 'eager')}
    np.savez(tmp_path/'arrays.npz', **arrays)
    metadata = dict(runtime='torch', genuine_oracle=True, status='complete', versions=dict(transformers='4.56.2'),
                    arrays_sha256=probe.sha(tmp_path/'arrays.npz'))
    (tmp_path/'result.json').write_text(json.dumps(metadata))
    return tmp_path


def test_reads_complete_reference_inventory(probe, reference):
    assert len(probe.read_mask_reference(reference, '4.56.2')) == 16


@pytest.mark.parametrize('field,value', [('runtime', 'jittor'), ('genuine_oracle', False), ('status', 'failed')])
def test_rejects_nonoracle_reference(probe, reference, field, value):
    path = reference/'result.json'
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='genuine oracle'):
        probe.read_mask_reference(reference, '4.56.2')


def test_reference_dependency_mismatch(probe, reference):
    with pytest.raises(ValueError, match='dependency'):
        probe.read_mask_reference(reference, '0.0.0')


@pytest.mark.parametrize('damage', ['empty', 'truncated', 'dtype', 'shape', 'corrupt'])
def test_reference_missing_or_wrong_arrays_fail(probe, reference, damage):
    import numpy as np
    with np.load(reference/'arrays.npz') as archive:
        arrays = {name: archive[name] for name in archive.files}
    if damage == 'empty':
        arrays = {}
    elif damage == 'truncated':
        arrays.pop(next(iter(arrays)))
    elif damage == 'dtype':
        arrays['3-0/bool'] = arrays['3-0/bool'].astype('int32')
    elif damage == 'shape':
        arrays['3-0/bool'] = arrays['3-0/bool'].reshape(-1)
    np.savez(reference/'arrays.npz', **arrays)
    metadata = json.loads((reference/'result.json').read_text())
    if damage != 'corrupt':
        metadata['arrays_sha256'] = probe.sha(reference/'arrays.npz')
        (reference/'result.json').write_text(json.dumps(metadata))
    else:
        with (reference/'arrays.npz').open('ab') as stream:
            stream.write(b'corruption')
    with pytest.raises(ValueError, match='inventory|shape/dtype|hash'):
        probe.read_mask_reference(reference, '4.56.2')
