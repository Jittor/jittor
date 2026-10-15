"""Offline full-profile/nox contracts; numerical acceptance needs real runtimes."""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'


@pytest.fixture
def provision():
    spec = importlib.util.spec_from_file_location('metric_td_nightly', ROOT / 'tools/provision_ecosystem_nightly.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_six_profiles_keep_original_public_roots_and_sessions(provision):
    # Public pins from d639; this control also runs in depth-one CI checkouts.
    common = ['numpy==1.26.4', 'accelerate==1.10.1', 'huggingface-hub==0.35.3',
              'tokenizers==0.22.2', 'safetensors==0.8.0', 'astunparse==1.6.3',
              'pillow==11.0.0', 'tqdm==4.67.1']
    expected = {
        'peft': (common, 'ecosystem_peft'),
        'swift': (common + ['datasets==3.6.0', 'pyarrow==19.0.1', 'modelscope==1.33.0',
                            'addict==2.4.0', 'dacite==1.9.2', 'einops==0.8.1', 'pandas==2.2.3',
                            'simplejson==4.2.0', 'sentencepiece==0.2.1', 'rouge==1.0.1',
                            'trl==0.21.0'], 'ecosystem_swift'),
        'verl': (['numpy==2.2.6', 'transformers==5.5.3', 'ray[default]==2.54.1',
                  'tensordict==0.10.0', 'huggingface-hub==1.5.0', 'tokenizers==0.22.2',
                  'safetensors==0.8.0', 'peft==0.17.1', 'accelerate==1.10.1', 'datasets==3.6.0',
                  'pandas==2.2.3', 'pyarrow==19.0.1', 'astunparse==1.6.3', 'pillow==11.0.0',
                  'tqdm==4.67.1'], 'ecosystem_verl_training'),
        'transformers': (common, 'ecosystem_transformers_source'),
        'openmmlab': (['numpy==1.26.4', 'opencv-python==4.11.0.86', 'astunparse==1.6.3',
                       'pillow==11.0.0', 'tqdm==4.67.1'], 'ecosystem_openmmlab_installed'),
        'diffusers': (['numpy==1.26.4', 'huggingface-hub==0.35.3', 'astunparse==1.6.3',
                      'pillow==11.0.0', 'tqdm==4.67.1'], 'ecosystem_diffusers_source'),
    }
    for library, (roots, session) in expected.items():
        assert provision.runtime_roots(library) == roots
        assert provision.session_name(library) == session


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
