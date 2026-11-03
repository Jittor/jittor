"""Pure guards for the offline TRL gate; importing this test never imports Torch."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def gate(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root/'agent/skills/torch-compat-repo-runbook/scripts'))
    spec = importlib.util.spec_from_file_location('maintained_trl_gate_test', root/'tools/run_trl_gate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment_info(gate, tmp_path, runtime='torch'):
    prefix = tmp_path/runtime
    binary = prefix/'lib/torch/_C.so'
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b'independent binary fixture')
    return dict(runtime=runtime, versions=dict(gate.PINS), python=[3, 11], prefix=str(prefix),
                base_prefix='/usr', executable=str(prefix/'bin/python'), shim=False,
                torch_version='2.8.0+cpu', torch_origin=str(binary.parent/'__init__.py'), binary=str(binary))


def test_private_genuine_oracle(gate, tmp_path):
    info = environment_info(gate, tmp_path)
    gate.validate_environment(info, 'torch')
    assert info['binary_sha256'] == gate.digest(info['binary'])


@pytest.mark.parametrize('field,value,message', [
    ('shim', True, 'shim'), ('versions', {}, 'pin mismatch'), ('python', [3, 12], 'Python 3.11'),
    ('torch_version', '2.7.1', '2.8.0'), ('binary', None, 'binary'), ('prefix', '/usr', 'private'),
])
def test_environment_rejection(gate, tmp_path, field, value, message):
    info = environment_info(gate, tmp_path)
    info[field] = value
    with pytest.raises(ValueError, match=message):
        gate.validate_environment(info, 'torch')


def test_binary_must_belong_to_private_oracle(gate, tmp_path):
    info = environment_info(gate, tmp_path)
    info['binary'] = str(tmp_path/'outside.so')
    Path(info['binary']).write_bytes(b'wrong environment')
    with pytest.raises(ValueError, match='escaped'):
        gate.validate_environment(info, 'torch')


@pytest.fixture
def pair(gate, tmp_path):
    import numpy as np
    from trl_rl_probe import array_hash, compare_arrays
    paths = [tmp_path/'torch', tmp_path/'jittor']
    spec = {}
    for name in ('embed_tokens.weight', 'layer.weight'):
        value = np.array([.25, -.5], dtype=np.float32)
        spec['parameter/'+name] = dict(shape=[2], dtype='float32', sha256=array_hash(value))
    for runtime, path in zip(('torch', 'jittor'), paths):
        (path/'initial').mkdir(parents=True)
        (path/'arrays').mkdir()
        payload = {name: np.array([.25, -.5], dtype=np.float32) for name in spec}
        np.savez(path/'initial/policy.npz', **payload)
        manifest = dict(status='complete', runtime=runtime, device='cpu', fallback_count=0,
                        oracle_binary=str(tmp_path/'oracle.so'), torch_version='2.8.0+cpu', protocol=dict(algorithm='sft', steps=3),
                        case=dict(expected_optimizer_steps=3), optimizer_steps=3,
                        gradient_norms=[1., 1., 1.], parameter_updates=[dict(l2=.1, changed_elements=2)]*3,
                        trainable_parameters={name[len('parameter/'):]: [2] for name in spec},
                        initial=dict(policy=dict(artifact_sha256=gate.digest(path/'initial/policy.npz'), tensors=spec)),
                        tool_sources={name: gate.digest(gate.ROOT/'compat/tests/torch/_ecosystem_runner.py'
                                                       if name == '_ecosystem_runner.py' else gate.SCRIPTS/name)
                                      for name in gate.expected_tools('sft')},
                        trl_source=dict(source_verified=True, version=gate.PINS['trl'], upstream=dict(revision=gate.REVISION)),
                        versions=gate.PINS, generation_calls=[],
                        arrays={}, comparison={}, numerical_passed=True)
        def record(key, exact=False):
            value = np.array([1, 2], dtype=np.int64) if exact else np.array([.25, -.5], dtype=np.float32)
            if key.endswith('/loss/0'):
                value = np.array(.25, dtype=np.float32)
            relative = 'arrays/'+array_hash(value)+'.npy'
            np.save(path/relative, value)
            manifest['arrays'][key] = dict(path=relative, shape=list(value.shape), dtype=str(value.dtype),
                                            sha256=array_hash(value), exact=exact)
            if runtime == 'jittor':
                manifest['comparison'][key] = compare_arrays(value, value, exact=exact)
        for step in range(3):
            prefix = 'step/%d/' % step
            for name in manifest['trainable_parameters']:
                record(prefix+'gradient/'+name+'/0')
                record(prefix+'updated_parameter/'+name+'/0')
            for name in ('embedding_gradient/policy/0/0', 'forward/output/0/0', 'loss/0'):
                record(prefix+name)
            record(prefix+'forward/input/input_ids/0', exact=True)
        for index in range(10):
            for name in ('input_ids', 'logits', 'next_logits'):
                record('step/3/final_eval/%d/%s/0' % (index, name), exact=name == 'input_ids')
        gate.write(path/'manifest.json', manifest)
    (tmp_path/'oracle.so').write_bytes(b'oracle proof')
    return paths


def alter_pair(gate, pair, mutate, both=True):
    for path in pair if both else pair[1:]:
        manifest = json.loads((path/'manifest.json').read_text())
        mutate(manifest)
        gate.write(path/'manifest.json', manifest)


def audit(gate, pair):
    return gate.audit_pair(*pair, method='sft', device='cpu', steps=3)


def test_audit_recomputes_every_array(gate, pair):
    row = audit(gate, pair)
    assert row['numerical_passed'] and row['steps'] == 3
    assert row['gradients']['fields'] == row['parameters']['fields'] == 6
    assert row['nonzero_parameter_update_steps'] == {'torch': 3, 'jittor': 3}


def test_vector_cannot_masquerade_as_scalar_loss_in_both_runtimes(gate, pair):
    def mutate(manifest):
        key = 'step/1/loss/0'
        source = 'step/1/forward/output/0/0'
        manifest['arrays'][key] = dict(manifest['arrays'][source])
        if source in manifest['comparison']:
            manifest['comparison'][key] = dict(manifest['comparison'][source])
    alter_pair(gate, pair, mutate)
    with pytest.raises(ValueError, match='scalar loss'):
        audit(gate, pair)


@pytest.mark.parametrize('kind', ['gradient', 'updated_parameter', 'embedding_gradient', 'forward/input', 'forward/output'])
def test_both_missing_observations_fail(gate, pair, kind):
    def mutate(manifest):
        for key in list(manifest['arrays']):
            if key.startswith('step/1/'+kind+'/'):
                manifest['arrays'].pop(key)
                manifest['comparison'].pop(key, None)
    alter_pair(gate, pair, mutate)
    with pytest.raises(ValueError, match='Missing'):
        audit(gate, pair)


@pytest.mark.parametrize('method', ['ppo', 'rloo'])
def test_online_losses_cannot_be_missing_in_both_runtimes(gate, pair, method):
    import shutil
    for path in pair:
        manifest = json.loads((path/'manifest.json').read_text())
        manifest['protocol']['algorithm'] = method
        manifest['tool_sources'] = {
            name: gate.digest(gate.ROOT/'compat/tests/torch/_ecosystem_runner.py'
                              if name == '_ecosystem_runner.py' else gate.SCRIPTS/name)
            for name in gate.expected_tools(method)}
        if method == 'ppo':
            shutil.copyfile(path/'initial/policy.npz', path/'initial/value.npz')
            manifest['initial']['value'] = manifest['initial']['policy']
            original_names = list(manifest['trainable_parameters'])
            manifest['trainable_parameters'] = {
                role+'.'+name: [2] for role in ('policy', 'value_model') for name in original_names}
            for field in ('arrays', 'comparison'):
                values = manifest[field]
                for step in range(3):
                    prefix = 'step/%d/' % step
                    for category in ('gradient', 'updated_parameter'):
                        for name in original_names:
                            key = prefix+category+'/'+name+'/0'
                            if key in values:
                                info = values.pop(key)
                                for role in ('policy', 'value_model'):
                                    values[prefix+category+'/'+role+'.'+name+'/0'] = info
                    key = prefix+'embedding_gradient/policy/0/0'
                    if key in values:
                        values[prefix+'embedding_gradient/value/0/0'] = values[key]
        for field in ('arrays', 'comparison'):
            manifest[field].pop('step/1/loss/0', None)
        gate.write(path/'manifest.json', manifest)
    with pytest.raises(ValueError, match='Missing scalar loss'):
        gate.audit_pair(*pair, method=method, device='cpu', steps=3)


@pytest.mark.parametrize('field,value,message', [
    ('optimizer_steps', 2, 'optimizer calls'), ('optimizer_steps', 3., 'optimizer calls'),
    ('fallback_count', 1, 'fallback'), ('device', 'cuda', 'device'),
    ('gradient_norms', [0., 0., 0.], 'training signal'),
])
def test_missing_actual_training_fail(gate, pair, field, value, message):
    alter_pair(gate, pair, lambda manifest: manifest.update({field: value}))
    with pytest.raises(ValueError, match=message):
        audit(gate, pair)


def test_both_truncated_parameter_inventory_fail(gate, pair):
    alter_pair(gate, pair, lambda manifest: manifest['trainable_parameters'].pop('layer.weight'))
    with pytest.raises(ValueError, match='trainable inventory'):
        audit(gate, pair)


def test_corrupt_retained_arrays_fail(gate, pair):
    import numpy as np
    path = pair[1]
    manifest = json.loads((path/'manifest.json').read_text())
    info = manifest['arrays']['step/0/loss/0']
    np.save(path/info['path'], np.array(.5, dtype=np.float32))
    with pytest.raises(AssertionError, match='hash mismatch'):
        audit(gate, pair)


def test_both_missing_final_prompt_fail(gate, pair):
    def mutate(manifest):
        key = 'step/3/final_eval/9/input_ids/0'
        manifest['arrays'].pop(key)
        manifest['comparison'].pop(key, None)
    alter_pair(gate, pair, mutate)
    with pytest.raises(ValueError, match='final evaluation'):
        audit(gate, pair)


def test_final_inputs_cannot_use_floating_budget(gate, pair):
    alter_pair(gate, pair, lambda manifest: manifest['arrays']['step/3/final_eval/0/input_ids/0'].update(exact=False))
    with pytest.raises(ValueError, match='must be exact'):
        audit(gate, pair)


def test_unused_oracle_jittor_context_is_not_compared(gate, pair):
    alter_pair(gate, pair, lambda manifest: manifest.update(jittor_commit='irrelevant', source_dirty=True), both=False)
    assert audit(gate, pair)['numerical_passed']


def test_manifest_pass_cannot_hide_comparison_corruption(gate, pair):
    alter_pair(gate, pair, lambda manifest: manifest['comparison']['step/0/loss/0'].update(max_abs=99), both=False)
    with pytest.raises(AssertionError, match='Recorded comparison'):
        audit(gate, pair)


def test_command_failure_is_not_retried(gate, tmp_path, monkeypatch):
    (tmp_path/'logs').mkdir()
    calls = []
    def failed(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=3)
    monkeypatch.setattr(gate.subprocess, 'run', failed)
    with pytest.raises(ValueError, match='Command failed'):
        gate.command(['fixture'], tmp_path, 'training', {})
    assert len(calls) == 1
    assert json.loads((tmp_path/'training.command.json').read_text())['exit_code'] == 3


def test_only_explicit_premeasurement_warmup_can_restart(gate, tmp_path, monkeypatch):
    (tmp_path/'logs').mkdir()
    def rebuilt(*args, **kwargs):
        kwargs['stdout'].write('jit_utils was rebuilt')
        return SimpleNamespace(returncode=3)
    monkeypatch.setattr(gate.subprocess, 'run', rebuilt)
    assert gate.command(['fixture'], tmp_path, 'warmup', {}, allow_restart=True) is False


def test_cache_gate_lease_is_exclusive(gate, tmp_path):
    with gate.lease(tmp_path):
        with pytest.raises(RuntimeError, match='leased'):
            with gate.lease(tmp_path):
                pytest.fail('concurrent cache lease accepted')


def test_source_checkout_cannot_hold_artifacts(gate):
    with pytest.raises(ValueError, match='outside'):
        gate.external(gate.ROOT/'evidence')


def test_environment_is_explicit_cpu_and_offline(gate, tmp_path, monkeypatch):
    monkeypatch.setenv('REAL_TORCH_SITE', 'polluted')
    monkeypatch.setenv('PYTHONPATH', 'polluted')
    env = gate.environment(tmp_path, 'torch', 'cpu', tmp_path/'cache')
    assert env['CUDA_VISIBLE_DEVICES'] == env['nvcc_path'] == env['PYTHONPATH'] == ''
    assert env['JITTOR_TORCH_SHIM'] == '0' and env['HF_HUB_OFFLINE'] == '1'
    assert 'REAL_TORCH_SITE' not in env


def test_cuda_requires_explicit_single_device(gate, tmp_path):
    with pytest.raises(ValueError, match='explicit'):
        gate.environment(tmp_path, 'jittor', 'cuda', tmp_path/'cache', '0,1')


def test_both_unverified_trl_sources_fail(gate, pair):
    alter_pair(gate, pair, lambda manifest: manifest['trl_source'].update(source_verified=False))
    with pytest.raises(ValueError, match='official TRL source'):
        audit(gate, pair)


@pytest.fixture
def fake_run(gate, tmp_path, monkeypatch):
    """Exercise the orchestration, not a replacement numerical implementation."""
    import trl_inventory
    inv = dict(denominator=dict(count=18), evaluation_subset=dict(count=16))
    monkeypatch.setattr(trl_inventory, 'inventory', lambda *args, **kwargs: inv)
    monkeypatch.setattr(gate, 'common_sources', lambda: {'unchanged producer': 'sha'})
    state = dict(context='before oracle batch', calls=[], fingerprints=[])
    def fingerprint():
        state['fingerprints'].append(state['context'])
        return dict(runtime=state['context'])
    monkeypatch.setattr(gate, 'fingerprint', fingerprint)
    def command(argv, output, label, env, **kwargs):
        state['calls'].append(label)
        if label.endswith('-preflight'):
            runtime = label[:-len('-preflight')]
            gate.write(argv[-1], environment_info(gate, tmp_path, runtime))
        elif label.endswith('-torch'):
            # Oracle measurement never imported these runtime files.
            state['context'] = label
        elif label.endswith('-audit'):
            request = json.loads(Path(argv[-1]).read_text())
            gate.write(Path(argv[-1]).with_name(request['method']+'.json'), dict(numerical_passed=True))
        return True
    monkeypatch.setattr(gate, 'command', command)
    interpreters = {}
    for runtime in ('oracle', 'candidate'):
        path = tmp_path/runtime/'bin/python'
        path.parent.mkdir(parents=True)
        path.write_text('fixture interpreter')
        interpreters[runtime] = path
    args = SimpleNamespace(output=tmp_path/'output', upstream=tmp_path/'upstream',
                           oracle_python=interpreters['oracle'], jittor_python=interpreters['candidate'],
                           device='cpu', steps=3, jittor_home=tmp_path/'cache', cuda_device=None)
    return args, state


def test_all_fresh_oracles_precede_frozen_candidate_batch(gate, fake_run):
    args, state = fake_run
    assert gate.run(args) == 0
    calls = state['calls']
    assert [label for label in calls if label.endswith('-torch')] == [method+'-torch' for method in gate.METHODS]
    assert calls.index('xpo-torch') < calls.index('sft-jittor')
    assert len(state['fingerprints']) == 33
    assert set(state['fingerprints']) == {'xpo-torch'}
    summary = json.loads((args.output/'summary.json').read_text())
    assert summary['status'] == 'passed' and len(summary['rows']) == 16


def test_run_propagates_candidate_failure_and_preserves_receipt(gate, fake_run, monkeypatch):
    args, state = fake_run
    original = gate.command
    def failing(argv, output, label, env, **kwargs):
        if label == 'sft-jittor':
            raise RuntimeError('actual process failed')
        return original(argv, output, label, env, **kwargs)
    monkeypatch.setattr(gate, 'command', failing)
    with pytest.raises(RuntimeError, match='actual process failed'):
        gate.run(args)
    summary = json.loads((args.output/'summary.json').read_text())
    assert summary['status'] == 'failed' and not summary['rows']
    assert not any(label.endswith('-audit') for label in state['calls'])


def test_existing_output_is_never_retried(gate, fake_run):
    args, state = fake_run
    args.output.mkdir()
    with pytest.raises(ValueError, match='fresh external'):
        gate.run(args)
    assert not state['calls']


def test_workflow_provisions_two_pinned_private_environments(gate):
    import yaml
    workflow = yaml.load((gate.ROOT/'.github/workflows/trl-compat-cpu.yml').read_text(), Loader=yaml.BaseLoader)
    assert {'schedule', 'workflow_dispatch', 'pull_request'} <= workflow['on'].keys()
    steps = workflow['jobs']['trl-text']['steps']
    scripts = '\n'.join(step.get('run', '') for step in steps)
    assert 'for runtime in oracle candidate' in scripts
    assert 'torch==2.8.0' in scripts and 'https://download.pytorch.org/whl/cpu' in scripts
    assert '--config-settings editable_mode=compat' in scripts and '-e compat' in scripts
    assert 'setuptools==80.9.0 wheel==0.48.0' in scripts
    assert 'tools/run_trl_gate.py' in scripts and '--steps 3' in scripts
    assert gate.REVISION in scripts
    assert '--device cpu' in scripts and '--device cuda' not in scripts
    requirements = (gate.ROOT/'requirements/trl-compat.txt').read_text()
    for package, version in gate.PINS.items():
        assert package+'=='+version in requirements
    assert 'safetensors==0.6.2' in requirements and 'tqdm==4.70.1' in requirements


@pytest.mark.parametrize('mutation', ['empty', 'truncated', 'unrelated'])
def test_both_incomplete_producer_source_inventories_fail(gate, pair, mutation):
    def mutate(manifest):
        sources = manifest['tool_sources']
        if mutation == 'empty':
            sources.clear()
        elif mutation == 'truncated':
            sources.pop('trl_sft_probe.py')
        else:
            sources.pop('trl_sft_probe.py')
            sources['trl_inventory.py'] = gate.digest(gate.SCRIPTS/'trl_inventory.py')
    alter_pair(gate, pair, mutate)
    with pytest.raises(ValueError, match='producer source inventory'):
        audit(gate, pair)


@pytest.mark.parametrize('method', ['sft', 'dpo', 'kto', 'orpo', 'cpo', 'reward', 'ppo', 'rloo',
                                    'gkd', 'online_dpo', 'grpo', 'prm', 'bco', 'iterative_sft', 'nash_md', 'xpo'])
def test_exact_source_owner_map(gate, method):
    names = gate.expected_tools(method)
    assert len(names) == 5
    assert {'_ecosystem_runner.py', 'trl_rl_probe.py', 'trl_sft_probe.py'} <= names
    if method in gate.TAIL:
        assert {'trl_tail_probe.py', 'trl_tail_cases.py'} <= names
    else:
        case = 'trl_offline_cases.py' if method in ('sft', 'dpo', 'kto', 'orpo', 'cpo', 'reward') else 'trl_mainstream_online.py'
        assert {'trl_mainstream_probe.py', case} <= names
