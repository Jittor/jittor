"""Controls for practical Trainer checkpoint and independent-runtime comparisons."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('trl_practical_offline', scripts / 'trl_practical_probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_run(probe, path, runtime, resume=False, delta=0.):
    from trl_practical_compare import expected_adapters
    path.mkdir()
    adapters = expected_adapters()
    protocol = dict(steps=10, checkpoint_step=5, optimizer='AdamW', learning_rate=1e-4,
                    betas=[.9, .999], epsilon=1e-8, weight_decay=.01, scheduler='linear', warmup_steps=0,
                    batch_size=1, max_length=32, order='SequentialSampler', attention='eager',
                    lora=dict(r=8, alpha=16, dropout=0., targets=['q_proj', 'v_proj']),
                    texts=list(map(str, range(10))), eval_prompts=['a', 'b', 'c'], precision='fp32')
    steps = list(range(6 if resume else 1, 11))
    manifest = dict(status='complete', fallback_count=0, protocol=protocol, versions={},
                    trl_source={}, peft_source={}, model_artifacts={}, initial=dict.fromkeys(adapters, 'initial'), device='cpu',
                    tool_sources={}, runtime=runtime, arrays={}, rng_before_loss={}, scheduler={}, scaler={},
                    resumed_from_step=5 if resume else None, trainable_parameters=adapters, trainable_parameter_count=1146880,
                    initial_adapters_sha256='initial-adapters', forward_dtypes=['torch.float32'],
                    optimizer_steps=steps, gradient_norms=[1.]*len(steps), updates=[1]*len(steps))
    # Reuse identical tiny fixture payloads across names/steps. No model-sized trajectory.
    cache = {}
    def record(key, shape, value=1., dtype='float32', exact=False):
        cache_key = (tuple(shape), value, dtype)
        if cache_key not in cache:
            array = np.full(shape, value, dtype=dtype)
            filename = 'payload-%d.npy' % len(cache)
            np.save(path / filename, array)
            cache[cache_key] = dict(path=filename, sha256=probe.array_hash(array), shape=shape, dtype=dtype)
        manifest['arrays'][key] = dict(cache[cache_key], exact=exact)
    for step in steps:
        prefix = 'step/%d/' % step
        record(prefix+'loss', [], 1.+delta)
        for name in ('input_ids', 'attention_mask', 'labels'):
            record(prefix+'input/'+name, [1, 2], dtype='int64', exact=True)
        for name, shape in adapters.items():
            for category in ('gradient/', 'parameter/', 'adam/'):
                for suffix in (('/step', '/exp_avg', '/exp_avg_sq') if category == 'adam/' else ('',)):
                    record(prefix+category+name+suffix, [] if suffix == '/step' else shape,
                           step if suffix == '/step' else 1.+delta, exact=suffix == '/step')
        manifest['rng_before_loss'][str(step)] = dict(python='rng', numpy='rng', cpu='rng')
        manifest['scheduler'][str(step)] = dict(last_epoch=step, base_lrs=[1e-4], _step_count=step+1,
            _is_initial=False, _get_lr_called_within_step=False, _last_lr=[1e-4*(10-step)/10], lr_lambdas=[{}])
        manifest['scaler'][str(step)] = None
    for index in range(3):
        prefix = 'final/%d/' % index
        record(prefix+'input_ids', [1, 2], dtype='int64', exact=True)
        record(prefix+'logits', [1, 2, 151936], 1.+delta)
        record(prefix+'next_logits', [151936], 1.+delta)
    checkpoint = path/'trainer/checkpoint-5'
    checkpoint.mkdir(parents=True)
    hashes = {}
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt', 'rng_state.pth',
                 'scheduler.pt', 'trainer_state.json', 'training_args.bin'):
        (checkpoint/name).write_text('fixture-'+name)
        hashes[name] = probe.digest(checkpoint/name)
    manifest['checkpoint_files'] = {'checkpoint-5/'+key: value for key, value in hashes.items()}
    if resume:
        manifest['resume_checkpoint_hashes'] = hashes
    (path / 'manifest.json').write_text(json.dumps(manifest))
    return path


def test_cross_runtime_accepts_equal_trajectory(probe, tmp_path):
    a = make_run(probe, tmp_path/'a', 'torch')
    b = make_run(probe, tmp_path/'b', 'jittor')
    assert probe.compare_runs(a, b)['passed']


def test_resume_requires_exact_values_not_only_tolerance(probe, tmp_path):
    a = make_run(probe, tmp_path/'a', 'torch')
    b = make_run(probe, tmp_path/'b', 'torch', resume=True, delta=1e-6)
    result = probe.compare_runs(a, b, resume=True)
    assert not result['passed']


def test_resume_requires_same_rng_and_scheduler(probe, tmp_path):
    a = make_run(probe, tmp_path/'a', 'torch')
    b = make_run(probe, tmp_path/'b', 'torch', resume=True)
    m = json.loads((b/'manifest.json').read_text())
    m['scheduler']['6']['last_epoch'] = 0
    (b/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(AssertionError, match='scheduler'):
        probe.compare_runs(a, b, resume=True)


def test_missing_resumed_step_rejected(probe, tmp_path):
    a = make_run(probe, tmp_path/'a', 'torch')
    b = make_run(probe, tmp_path/'b', 'torch', resume=True)
    m = json.loads((b/'manifest.json').read_text())
    del m['arrays'][next(key for key in m['arrays'] if key.startswith('step/6/gradient/'))]
    (b/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(AssertionError, match='inventory'):
        probe.compare_runs(a, b, resume=True)


def test_shim_cannot_be_its_own_independent_oracle(probe, tmp_path):
    a = make_run(probe, tmp_path/'a', 'jittor')
    b = make_run(probe, tmp_path/'b', 'jittor')
    with pytest.raises(AssertionError, match='independent'):
        probe.compare_runs(a, b)


def test_altered_checkpoint_observation_rejected(probe, tmp_path):
    a = make_run(probe, tmp_path/'a', 'torch')
    b = make_run(probe, tmp_path/'b', 'torch', resume=True)
    m = json.loads((b/'manifest.json').read_text())
    info = m['arrays']['step/6/loss']
    np.save(b/info['path'], np.array(9., dtype=np.float32))
    with pytest.raises(AssertionError, match='hash mismatch'):
        probe.compare_runs(a, b, resume=True)


def test_counter_values_and_representation_are_reported_separately(probe, tmp_path):
    from trl_practical_compare import compare
    a = make_run(probe, tmp_path/'a', 'torch')
    b = make_run(probe, tmp_path/'b', 'jittor')
    for path, dtype in ((a, np.float32), (b, np.float64)):
        m = json.loads((path/'manifest.json').read_text())
        for key, info in m['arrays'].items():
            if '/adam/' not in key or not key.endswith('/step'):
                continue
            data = np.array(int(key.split('/')[1]), dtype=dtype)
            filename = 'counter-'+key.split('/')[1]+'.npy'
            np.save(path/filename, data)
            info.update(path=filename, dtype=str(data.dtype), sha256=probe.array_hash(data), shape=[])
        (path/'manifest.json').write_text(json.dumps(m))

    result = compare(a, b)
    assert result['numerical_passed']
    assert not result['strict_passed']
    assert len(result['counter_representation_mismatches']) == 1120


def test_repeat_control_is_explicit_and_same_runtime_only(probe, tmp_path):
    from trl_practical_compare import compare
    a = make_run(probe, tmp_path/'a', 'jittor')
    b = make_run(probe, tmp_path/'b', 'jittor')
    assert compare(a, b, repeat=True)['byte_equal']
    c = make_run(probe, tmp_path/'c', 'torch')
    with pytest.raises(AssertionError, match='same runtime'):
        compare(a, c, repeat=True)


def amp_run(probe, path, runtime, low_rsqrt=False):
    from trl_amp_precision_probe import NODES
    path.mkdir()
    arrays = {}
    for precision in ('fp16', 'bf16'):
        for name in NODES:
            value = np.array([1.], dtype=np.float32)
            filename = precision+'-'+name+'.npy'
            np.save(path/filename, value)
            arrays[precision+'/'+name] = dict(path=filename, shape=[1], dtype='float32',
                tensor_dtype='float16' if low_rsqrt and name == 'rsqrt' else 'float32',
                sha256=probe.array_hash(value))
    manifest = dict(status='complete', fallback_count=0, runtime=runtime, device='cuda',
                    protocol={}, input_hash='same', weight_hash='same', arrays=arrays)
    (path/'manifest.json').write_text(json.dumps(manifest))
    return path


def test_amp_dtype_narrowing_detected_even_when_values_match(probe, tmp_path):
    from trl_amp_precision_probe import compare
    a = amp_run(probe, tmp_path/'a', 'torch')
    b = amp_run(probe, tmp_path/'b', 'jittor', low_rsqrt=True)
    result = compare(a, b)
    assert result['dtype_mismatches'] == 2
    assert result['numerical_failed_fields'] == 0


def test_amp_incomplete_nodes_rejected(probe, tmp_path):
    from trl_amp_precision_probe import compare
    a = amp_run(probe, tmp_path/'a', 'torch')
    b = amp_run(probe, tmp_path/'b', 'jittor')
    m = json.loads((b/'manifest.json').read_text())
    del m['arrays']['bf16/rsqrt']
    (b/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(AssertionError, match='Incomplete'):
        compare(a, b)


@pytest.mark.parametrize('damage', ['all_gradients', 'all_adam', 'all_logits', 'inputs', 'loss',
                                    'optimizer_steps', 'rng_before_loss', 'scheduler', 'scaler',
                                    'adapters', 'precision', 'zero_updates', 'scheduler_contents'])
def test_equally_incomplete_traces_cannot_pass(probe, tmp_path, damage):
    from trl_practical_compare import compare
    a = make_run(probe, tmp_path/'a', 'torch')
    b = make_run(probe, tmp_path/'b', 'jittor')
    for path in (a, b):
        m = json.loads((path/'manifest.json').read_text())
        if damage in ('all_gradients', 'all_adam', 'all_logits', 'inputs', 'loss'):
            marker = {'all_gradients': '/gradient/', 'all_adam': '/adam/', 'all_logits': 'logits',
                      'inputs': '/input/', 'loss': '/loss'}[damage]
            m['arrays'] = {key: info for key, info in m['arrays'].items() if marker not in key}
        elif damage in ('rng_before_loss', 'scheduler', 'scaler'):
            m[damage] = {}
        elif damage == 'optimizer_steps':
            m['optimizer_steps'] = list(range(2, 11))
        elif damage == 'scheduler_contents':
            m['scheduler'] = {key: {'last_epoch': value['last_epoch']} for key, value in m['scheduler'].items()}
        elif damage == 'adapters':
            m['trainable_parameters'].pop(next(iter(m['trainable_parameters'])))
        elif damage == 'precision':
            m['forward_dtypes'] = []
        else:
            m['updates'][0] = 0
        (path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(AssertionError):
        compare(a, b)


def test_identically_wrong_adam_step_rejected(probe, tmp_path):
    from trl_practical_compare import compare
    a = make_run(probe, tmp_path/'a', 'torch')
    b = make_run(probe, tmp_path/'b', 'jittor')
    for path in (a, b):
        m = json.loads((path/'manifest.json').read_text())
        key = next(key for key in m['arrays'] if '/adam/' in key and key.endswith('/step'))
        wrong = np.array(99., dtype=np.float32)
        np.save(path/'wrong.npy', wrong)
        m['arrays'][key].update(path='wrong.npy', sha256=probe.array_hash(wrong))
        (path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(AssertionError, match='Adam counter'):
        compare(a, b)


@pytest.mark.parametrize('damage', ['empty_rng', 'checkpoint_hash', 'checkpoint_file'])
def test_resume_requires_full_state_and_checkpoint_binding(probe, tmp_path, damage):
    a = make_run(probe, tmp_path/'a', 'torch')
    b = make_run(probe, tmp_path/'b', 'torch', resume=True)
    m = json.loads((b/'manifest.json').read_text())
    if damage == 'empty_rng':
        m['rng_before_loss'] = {}
    elif damage == 'checkpoint_hash':
        m['resume_checkpoint_hashes']['optimizer.pt'] = 'wrong'
    else:
        (a/'trainer/checkpoint-5/optimizer.pt').write_text('corrupted')
    (b/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(AssertionError):
        probe.compare_runs(a, b, resume=True)


def test_same_directory_repeat_is_not_evidence(probe, tmp_path):
    from trl_practical_compare import compare
    a = make_run(probe, tmp_path/'a', 'torch')
    with pytest.raises(AssertionError, match='distinct'):
        compare(a, a, repeat=True)
