"""Offline complete-nightly controls; numerical claims require actual workers."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'


@pytest.fixture
def collector(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import collect_accelerate_scorecards
    return collect_accelerate_scorecards


@pytest.fixture
def complete_cpu(tmp_path, collector):
    from _accelerate_fresh_scorecard import protocol_spec
    artifacts = []
    for pid, (runtime, scenario, phase) in enumerate(
            ((runtime, scenario, phase) for runtime in ('reference', 'candidate')
             for scenario in ('base', 'perturb') for phase in ('produce', 'consume')), 100):
        role = '%s-%s-rank0-%s-metadata' % (runtime, scenario, phase)
        path = tmp_path / (role + '.json')
        path.write_text(json.dumps({'interpreter': {'pid': pid}}))
        artifacts.append(collector.artifact(role, path))
    row = {'repo': 'accelerate', 'device': 'cpu', 'protocol': protocol_spec(1, 'no'),
           'status': 'passed', 'level': 3,
           'numerics': {'passed': True, 'finite': True, 'checked_tensors': 438, 'missing_tensors': [],
                        'parameter_gradients': {'requested': 28, 'checked': 28},
                        'input_gradients': {'requested': 14, 'checked': 14}, 'atol': 2e-6, 'rtol': 2e-5},
           'training': {'steps': 5, 'loss_trajectory_passed': True, 'parameter_trajectory_passed': True},
           'own_continuation': {'model_optimizer_exact': True, 'rng_exact': True, 'trajectory_exact': True},
           'fresh_process': {'distinct_producer_consumer': True, 'producer_terminated': True,
                             'checkpoint_files_immutable': True, 'full_rng_state_exact': True,
                             'scheduler_exact': True}, 'evidence': {'artifacts': artifacts}}
    return [row], {'accepted': [dict(row, verified_level=3)], 'blocked': [], 'rejected': []}


def test_complete_original_protocol_accepts(collector, complete_cpu):
    collector.require_complete_cpu(*complete_cpu)


@pytest.mark.parametrize('change', ['arrays', 'param_grad', 'input_grad', 'steps', 'budget',
                                   'world', 'device', 'level', 'rng', 'scheduler', 'phase', 'blocked'])
def test_complete_nightly_refuses_shortened_or_unadmitted_evidence(collector, complete_cpu, change):
    rows, admission = copy.deepcopy(complete_cpu)
    row = rows[0]
    if change == 'arrays':
        row['numerics']['checked_tensors'] = 437
    elif change in ('param_grad', 'input_grad'):
        row['numerics']['parameter_gradients' if change == 'param_grad' else 'input_gradients']['checked'] -= 1
    elif change == 'steps':
        row['training']['steps'] = 3
    elif change == 'budget':
        row['numerics']['atol'] = .002
    elif change == 'world':
        row['protocol']['world_size'] = 2
    elif change == 'device':
        row['device'] = 'cuda'
    elif change == 'level':
        admission['accepted'][0]['verified_level'] = 2
    elif change in ('rng', 'scheduler'):
        row['fresh_process']['full_rng_state_exact' if change == 'rng' else 'scheduler_exact'] = False
    elif change == 'phase':
        row['evidence']['artifacts'].pop()
    else:
        admission['blocked'].append({'reason': 'missing CPU counterpart'})
    with pytest.raises(ValueError, match='Complete CPU|Missing complete CPU'):
        collector.require_complete_cpu(rows, admission)


@pytest.mark.parametrize('changed_hash', (False, True))
def test_actual_phase_bytes_and_distinct_processes_required(collector, complete_cpu, changed_hash):
    rows, admission = copy.deepcopy(complete_cpu)
    artifact = rows[0]['evidence']['artifacts'][1]
    path = Path(artifact['path'])
    path.write_text(json.dumps({'interpreter': {'pid': 100}}))
    if changed_hash:
        artifact['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='artifact changed|distinct actual worker'):
        collector.require_complete_cpu(rows, admission)


def invoke(monkeypatch, tmp_path, complete=True, device='cpu', arguments=()):
    function = next(node for node in ast.parse((ROOT / 'noxfile.py').read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'ecosystem_accelerate')
    function.decorator_list = []
    namespace = {'Path': Path, 'os': os, 'REPO_ROOT': ROOT, 'NOX_STATE_ROOT': tmp_path / 'nox',
                 'subprocess': SimpleNamespace(check_output=lambda *_args, **_kwargs: 'a' * 40),
                 '_session_env': lambda *_: (tmp_path / 'session', {}),
                 '_hardware_python': lambda: 'candidate-python',
                 '_set_hardware_python_config': lambda *_: None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 'noxfile.py', 'exec'), namespace)
    monkeypatch.setenv('REAL_TORCH_PYTHON', 'oracle-python')
    monkeypatch.setenv('JITTOR_ACCELERATE_CHECKOUT', str(tmp_path / 'upstream'))
    monkeypatch.setenv('JITTOR_ACCELERATE_WHEEL', str(tmp_path / 'official.whl'))
    monkeypatch.setenv('JITTOR_TEST_DEVICES', device)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    monkeypatch.setenv('JITTOR_ACCELERATE_COMPLETE_CPU', '1' if complete else '0')
    monkeypatch.setenv('JITTOR_ACCELERATE_EVIDENCE_DIR', str(tmp_path / 'retained'))
    calls = []
    def error(message):
        raise ValueError(message)
    session = SimpleNamespace(posargs=list(arguments), error=error,
                              run=lambda *args, **kwargs: calls.append((args, kwargs)))
    namespace['ecosystem_accelerate'](session)
    return calls


def test_nightly_runs_full_gate_then_strict_collector_with_separate_benchmark_cache(monkeypatch, tmp_path):
    calls = invoke(monkeypatch, tmp_path)
    assert len(calls) == 2
    gate, collect = (row[0] for row in calls)
    assert '--fresh-process' in gate and gate[gate.index('--device') + 1] == 'cpu'
    assert gate[gate.index('--output') + 1] == str(tmp_path / 'retained')
    assert gate[gate.index('--cache-root') + 1] == str(tmp_path / 'session/benchmark-cache')
    assert '--require-complete-cpu' in collect
    assert collect[collect.index('--run') + 1] == str(tmp_path / 'retained')
    assert all(row[1]['env']['JITTOR_TORCH_SHIM'] == '0' for row in calls)


@pytest.mark.parametrize('device,arguments', [('cuda', ()), ('cpu', ('--seed', '7')),
                                            ('cpu', ('--output', 'elsewhere'))])
def test_nightly_cannot_shrink_or_substitute_protocol(monkeypatch, tmp_path, device, arguments):
    with pytest.raises(ValueError, match='mandatory'):
        invoke(monkeypatch, tmp_path, device=device, arguments=arguments)


@pytest.mark.parametrize('device', ('cpu', 'cuda'))
def test_existing_cpu_cuda_entrypoints_keep_optional_arguments(monkeypatch, tmp_path, device):
    calls = invoke(monkeypatch, tmp_path, complete=False, device=device, arguments=('--cache-root', 'own-cache'))
    assert len(calls) == 1
    args = calls[0][0]
    assert args[-2:] == ('--cache-root', 'own-cache')
    if device == 'cuda':
        assert args[args.index('--devices') + 1] == '0,1'


def test_profile_uses_official_wheel_and_preserves_real_pip_record(monkeypatch):
    spec = importlib.util.spec_from_file_location('accelerate_nightly_provision', ROOT / 'tools/provision_ecosystem_nightly.py')
    provision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provision)
    assert provision.session_name('accelerate') == 'ecosystem_accelerate'
    assert provision.runtime_roots('accelerate') == ['numpy==1.26.4', 'huggingface-hub==0.35.3',
                                                  'astunparse==1.6.3', 'pillow==11.0.0', 'tqdm==4.67.1']


def test_committed_control_closure_is_baseline_specific_and_missing_files_refuse(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import _accelerate_scorecard as validator
    root = tmp_path / 'source'
    root.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE).decode().strip()
    git('init', '-q')
    git('config', 'user.email', 'fixture@example.invalid')
    git('config', 'user.name', 'Offline fixture')
    for name in validator.MEASUREMENT_SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# source fixture\n')
    git('add', '.')
    git('commit', '-qm', 'original fixture')
    old = git('rev-parse', 'HEAD')
    monkeypatch.setattr(validator, 'ROOT', root)
    original = validator.measurement_sources(old)
    protocol = root / 'compat/tests/torch/_accelerate_protocol.py'
    protocol.write_text("CONTROL_SOURCE_FILES = ('tools/control.py',)\n")
    git('add', '.')
    git('commit', '-qm', 'missing control fixture')
    with pytest.raises(ValueError, match='Unknown measured'):
        validator.measurement_sources(git('rev-parse', 'HEAD'))
    control = root / 'tools/control.py'
    control.write_text('CONTROL = 1\n')
    git('add', '.')
    git('commit', '-qm', 'bound control fixture')
    current = git('rev-parse', 'HEAD')
    captured = validator.measurement_sources(current)
    assert validator.measurement_sources(old) == original
    assert captured['tools/control.py'] == hashlib.sha256(control.read_bytes()).hexdigest()
    control.write_text('CONTROL = 2\n')
    assert hashlib.sha256(control.read_bytes()).hexdigest() != validator.measurement_sources(current)['tools/control.py']
    control.unlink()
    assert validator.measurement_sources(current) == captured  # Never silently omit tracked files.
