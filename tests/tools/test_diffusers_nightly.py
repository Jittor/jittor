"""Exercise maintained Diffusers launch policy without importing a tensor runtime."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def invoke(tmp_path, monkeypatch, arguments=(), device='cpu', oracle=True):
    nodes = ast.parse((ROOT / 'noxfile.py').read_text()).body
    node = next(node for node in nodes if isinstance(node, ast.FunctionDef)
                and node.name == 'ecosystem_diffusers_source')
    node.decorator_list = []
    namespace = {'Path': Path, 'os': os, 'REPO_ROOT': ROOT,
                 '_session_env': lambda *args: (tmp_path / 'session', {}),
                 '_hardware_python': lambda: 'candidate-python',
                 '_set_hardware_python_config': lambda *args: None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 'noxfile.py', 'exec'), namespace)
    reference, wheel = tmp_path / 'oracle-python', tmp_path / 'official.whl'
    reference.touch()
    wheel.touch()
    monkeypatch.setenv('REAL_TORCH_PYTHON', str(reference) if oracle else '')
    monkeypatch.setenv('JITTOR_TEST_DEVICES', device)
    monkeypatch.setenv('JITTOR_DIFFUSERS_CHECKOUT', str(tmp_path))
    monkeypatch.setenv('JITTOR_DIFFUSERS_WHEEL', str(wheel))
    monkeypatch.setenv('JITTOR_DIFFUSERS_EVIDENCE_DIR', str(tmp_path / 'retained-output'))
    calls = []

    def error(message):
        raise ValueError(message)

    session = SimpleNamespace(posargs=list(arguments), error=error,
                              run=lambda *args, **kwargs: calls.append((args, kwargs)))
    namespace['ecosystem_diffusers_source'](session)
    return calls


def test_actual_nox_session_calls_full_gate_and_retains_output(tmp_path, monkeypatch):
    calls = invoke(tmp_path, monkeypatch)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:2] == ('candidate-python', str(ROOT / 'tools/run_diffusers_gate.py'))
    assert args[args.index('--device') + 1] == 'cpu'
    assert args[args.index('--output') + 1] == str(tmp_path / 'retained-output')
    assert args[args.index('--cache-root') + 1] == str(tmp_path / 'session/benchmark-cache')
    assert kwargs['env']['JITTOR_REQUIRE_DIFFUSERS'] == '1'
    assert kwargs['env']['JITTOR_REQUIRE_REAL_TORCH'] == '1'
    assert kwargs['env']['JITTOR_TEST_REQUIRE_EXECUTION'] == '1'


@pytest.mark.parametrize('arguments', [('--preflight',), ('--case', 'tiny'),
                                      ('--steps', '1'), ('--seed', '123'),
                                      ('--atol', '1'), ('--output',)])
def test_nox_cannot_reduce_original_protocol(tmp_path, monkeypatch, arguments):
    with pytest.raises(ValueError, match='mandatory'):
        invoke(tmp_path, monkeypatch, arguments)


@pytest.mark.parametrize('device,oracle', [('cpu,cuda', True), ('cpu', False)])
def test_nox_requires_one_device_and_real_oracle(tmp_path, monkeypatch, device, oracle):
    with pytest.raises(ValueError, match='binary REAL_TORCH_PYTHON'):
        invoke(tmp_path, monkeypatch, device=device, oracle=oracle)
