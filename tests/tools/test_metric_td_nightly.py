"""Offline full-profile/nox contracts; numerical acceptance needs real runtimes."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'
BASE = 'd63976115fb6a31d94c4474871b07f2aba69d7e7'


@pytest.fixture
def provision():
    spec = importlib.util.spec_from_file_location('metric_td_nightly', ROOT / 'tools/provision_ecosystem_nightly.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_six_profiles_keep_original_roots_sessions_and_nox_implementations(provision):
    namespace = {'__name__': 'original_nightly', '__file__': str(ROOT / 'tools/provision_ecosystem_nightly.py')}
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE + ':tools/provision_ecosystem_nightly.py'], text=True)
    exec(compile(source, 'original-nightly.py', 'exec'), namespace)
    for library in ('peft', 'swift', 'verl', 'transformers', 'openmmlab', 'diffusers'):
        assert provision.runtime_roots(library) == namespace['runtime_roots'](library)
        assert provision.session_name(library) == namespace['session_name'](library)
    original = ast.parse(subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', BASE + ':noxfile.py'], text=True))
    current = ast.parse((ROOT / 'noxfile.py').read_text())
    before = {node.name: ast.dump(node) for node in original.body if isinstance(node, ast.FunctionDef)}
    after = {node.name: ast.dump(node) for node in current.body if isinstance(node, ast.FunctionDef)}
    assert all(after[name] == value for name, value in before.items())
    assert set(after) - set(before) == {'ecosystem_metric_tensordict_source'}


def test_full_profile_keeps_original_numeric_dependencies_and_actual_pip_receipts(provision):
    assert provision.session_name('metric_tensordict') == 'ecosystem_metric_tensordict_source'
    assert set(provision.runtime_roots('metric_tensordict')) == {
        'numpy==1.26.4', 'astunparse==1.6.3', 'pillow==11.0.0', 'tqdm==4.67.1'}
    tree = ast.parse((ROOT / 'tools/provision_ecosystem_nightly.py').read_text())
    conditions = [node.test for node in ast.walk(tree) if isinstance(node, ast.If)
                  and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                          and child.func.id == 'unpack_pinned_wheel' for child in ast.walk(node))]
    skips = [condition for condition in conditions if isinstance(condition, ast.Compare)
             and isinstance(condition.left, ast.Name) and condition.left.id == 'library'
             and isinstance(condition.ops[0], ast.NotIn)]
    assert len(skips) == 1
    assert set(ast.literal_eval(skips[0].comparators[0])) == {
        'openmmlab', 'diffusers', 'metric_tensordict'}


@pytest.mark.parametrize('valid', (True, False))
def test_real_release_tag_is_fetched_and_peeled_commit_is_required(provision, tmp_path, monkeypatch, valid):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import _metric_td_installation as component

    checkout_calls, fetch_calls = [], []
    monkeypatch.setattr(provision, 'checkout', lambda *args: checkout_calls.append(args))
    monkeypatch.setattr(provision, 'official_wheel', lambda repo, *_: tmp_path / (repo + '.whl'))
    monkeypatch.setattr(component, 'source_descriptor', lambda repo, *_: {'repo': repo})
    sha = component.PINS['tensordict'][1]
    monkeypatch.setattr(provision.subprocess, 'check_output', lambda argv, **_: (sha if valid else '0' * 40) + '\n')
    commands = SimpleNamespace(run=lambda *args: fetch_calls.append(args))
    if not valid:
        with pytest.raises(ValueError, match='release tag differs'):
            provision.metric_td_checkouts(commands, tmp_path, tmp_path / 'readonly-cache', tmp_path / 'wheels')
        return
    checkouts, wheels, sources = provision.metric_td_checkouts(
        commands, tmp_path, tmp_path / 'readonly-cache', tmp_path / 'wheels')
    assert set(checkouts) == set(wheels) == set(sources) == {'torchmetrics', 'tensordict'}
    assert len(checkout_calls) == 2
    assert fetch_calls == [('git', '-C', checkouts['tensordict'], 'fetch', '--depth', '1',
                            tmp_path / 'readonly-cache/tensordict',
                            'refs/tags/v0.10.0:refs/tags/v0.10.0')]


def invoke(tmp_path, monkeypatch, arguments=(), device='cpu', oracle=True):
    node = next(node for node in ast.parse((ROOT / 'noxfile.py').read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'ecosystem_metric_tensordict_source')
    node.decorator_list = []
    namespace = {'Path': Path, 'os': os, 'REPO_ROOT': ROOT,
                 '_session_env': lambda *args: (tmp_path / 'session', {}),
                 '_hardware_python': lambda: 'candidate-python',
                 '_set_hardware_python_config': lambda *args: None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 'noxfile.py', 'exec'), namespace)
    reference = tmp_path / 'oracle-python'
    reference.touch()
    monkeypatch.setenv('REAL_TORCH_PYTHON', str(reference) if oracle else '')
    monkeypatch.setenv('JITTOR_TEST_DEVICES', device)
    for repo in ('TORCHMETRICS', 'TENSORDICT'):
        wheel = tmp_path / (repo + '.whl')
        wheel.touch()
        monkeypatch.setenv('JITTOR_' + repo + '_CHECKOUT', str(tmp_path))
        monkeypatch.setenv('JITTOR_' + repo + '_WHEEL', str(wheel))
    monkeypatch.setenv('JITTOR_METRIC_TD_EVIDENCE_DIR', str(tmp_path / 'retained'))
    calls = []

    def error(message):
        raise ValueError(message)

    session = SimpleNamespace(posargs=list(arguments), error=error,
                              run=lambda *args, **kwargs: calls.append((args, kwargs)))
    namespace['ecosystem_metric_tensordict_source'](session)
    return calls


def test_nox_calls_both_complete_components_and_only_workers_activate_shim(tmp_path, monkeypatch):
    calls = invoke(tmp_path, monkeypatch)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:2] == ('candidate-python', str(ROOT / 'tools/run_metric_td_gate.py'))
    assert args[args.index('--device') + 1] == 'cpu'
    assert args[args.index('--output') + 1] == str(tmp_path / 'retained')
    assert args[args.index('--cache-root') + 1] == str(tmp_path / 'session/benchmark-cache')
    for repo in ('torchmetrics', 'tensordict'):
        assert '--' + repo + '-checkout' in args and '--' + repo + '-wheel' in args
        assert kwargs['env']['JITTOR_REQUIRE_' + repo.upper()] == '1'
    assert kwargs['env']['JITTOR_TORCH_SHIM'] == '0' and kwargs['env']['PYTHONPATH'] == ''
    assert kwargs['env']['JITTOR_REQUIRE_REAL_TORCH'] == kwargs['env']['JITTOR_TEST_REQUIRE_EXECUTION'] == '1'


@pytest.mark.parametrize('arguments', [('--preflight',), ('--repo', 'torchmetrics'),
                                      ('--steps', '1'), ('--seed', '7'), ('--atol', '1'), ('--output',)])
def test_nox_cannot_shrink_or_substitute_complete_protocol(tmp_path, monkeypatch, arguments):
    with pytest.raises(ValueError, match='mandatory'):
        invoke(tmp_path, monkeypatch, arguments)


@pytest.mark.parametrize('device,oracle', [('cpu,cuda', True), ('cpu', False)])
def test_nox_requires_one_device_and_independent_binary_oracle(tmp_path, monkeypatch, device, oracle):
    with pytest.raises(ValueError, match='binary REAL_TORCH_PYTHON'):
        invoke(tmp_path, monkeypatch, device=device, oracle=oracle)
