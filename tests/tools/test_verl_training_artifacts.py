"""Host-only ownership tests; synthetic outputs grant no numerical coverage."""

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pytest

from _helpers.child_process import PYTHON

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'compat/tests/torch'))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module('_ecosystem_verl_training_harness')


def run_host_protocol(harness, monkeypatch, *, fail_phase=None):
    contract = importlib.import_module('_verl_component_contract')
    registry = importlib.import_module('registry_inventory')
    observed, phases = [], []
    source = {'upstream': {'commit': 'e5a627cf55c94d85406556854c2c1a22db4ff5e7'}}
    monkeypatch.setattr(registry, 'inventory', lambda *args: source)
    monkeypatch.setattr(harness.subprocess, 'check_output', lambda *args, **kw: 'a' * 40)

    def worker(python, case, command, **kwargs):
        output = Path(command[command.index('--output') + 1] if '--output' in command
                      else command[command.index('--') + 3])
        seed = int(command[command.index('--seed') + 1])
        np.savez(output, __output__=np.array([seed + .2], dtype='float32'))
        output.with_suffix('.json').write_text(json.dumps({'synthetic_host_fixture': True, 'seed': seed}))
        weights = output.with_name(output.stem + '.weights.npz')
        np.savez(weights, weight=np.ones(1, dtype='float32'))
        observed.append(output)

    def compare(*args, **kwargs):
        phases.append((args[5], kwargs.get('training', False)))
        if len(phases) == fail_phase:
            raise ValueError('final bound metadata refused')
        return {'rows': [{'name': 'synthetic_host_fixture'}],
                'reference_result': {'weights_sha256': 'verified-by-mock-only'}}

    monkeypatch.setattr(harness, '_run_command', worker)
    monkeypatch.setattr(contract, 'compare_pair', compare)
    harness.compare_training(SimpleNamespace(device='cpu', assertEqual=unittest.TestCase().assertEqual,
                            assertFalse=unittest.TestCase().assertFalse),
                             'verl_distillation_k3', ROOT, ROOT / 'wheel.whl')
    return observed, phases


def test_explicit_output_retains_all_phases_and_hashes_only_after_final_validation(harness, tmp_path, monkeypatch):
    monkeypatch.setenv('JITTOR_VERL_EVIDENCE_DIR', str(tmp_path / 'evidence'))
    outputs, phases = run_host_protocol(harness, monkeypatch)
    assert phases == [(0, False), (7, False), (19, False), (0, True)]
    assert len(outputs) == 8 and all(path.is_file() for path in outputs)
    directory = outputs[0].parent
    completed = json.loads((directory / 'gate-complete.json').read_text())
    assert completed['status'] == 'passed' and len(completed['comparisons']) == 4
    assert completed['files'] == {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in directory.iterdir() if path.name != 'gate-complete.json'}
    assert {path.name for path in outputs} == {'seed%d-%s.npz' % (seed, runtime)
                                             for seed in (0, 7, 19) for runtime in ('torch', 'jittor')} | {
                                                 'train-torch.npz', 'train-jittor.npz'}
    with pytest.raises(FileExistsError):
        run_host_protocol(harness, monkeypatch)


def test_failed_final_metadata_keeps_raw_without_success_record(harness, tmp_path, monkeypatch):
    directory = tmp_path / 'failed' / 'cpu/verl_distillation_k3'
    monkeypatch.setenv('JITTOR_VERL_EVIDENCE_DIR', str(tmp_path / 'failed'))
    with pytest.raises(ValueError, match='final bound metadata'):
        run_host_protocol(harness, monkeypatch, fail_phase=4)
    assert len(list(directory.glob('*.npz'))) == 16
    assert not (directory / 'gate-complete.json').exists()


def test_default_keeps_temporary_cleanup(harness, monkeypatch):
    monkeypatch.delenv('JITTOR_VERL_EVIDENCE_DIR', raising=False)
    outputs, _ = run_host_protocol(harness, monkeypatch)
    assert all(not path.exists() and not path.parent.exists() for path in outputs)


def test_nox_default_directory_is_unique_and_explicit_output_is_preserved(tmp_path, monkeypatch):
    node = next(node for node in ast.parse((ROOT / 'noxfile.py').read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'ecosystem_verl_training')
    node.decorator_list = []
    source, wheel = tmp_path / 'upstream', tmp_path / 'upstream.whl'
    source.mkdir()
    wheel.touch()
    monkeypatch.setenv('REAL_TORCH_PYTHON', PYTHON)
    monkeypatch.setenv('JITTOR_VERL_UPSTREAM', str(source))
    monkeypatch.setenv('JITTOR_VERL_WHEEL', str(wheel))
    monkeypatch.setenv('JITTOR_TEST_DEVICES', 'cpu')
    monkeypatch.delenv('JITTOR_VERL_EVIDENCE_DIR', raising=False)
    captured = []
    scratch = tmp_path / 'session-scratch'

    def fresh_session(*args):
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir()
        return scratch, {}

    namespace = {'Path': Path, 'os': os, 'NOX_STATE_ROOT': tmp_path / 'state', 'THREAD_POOL_ENV_NAMES': (),
                 '_session_env': fresh_session, '_hardware_python': lambda: PYTHON,
                 '_set_hardware_python_config': lambda *args: None, '_install_compat_source': lambda *args: None,
                 '_run_pytest_once': lambda session, tests, env, *args, **kw: captured.append(dict(env))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'noxfile.py', 'exec'), namespace)
    session = SimpleNamespace(run=mock.Mock(), error=lambda message: pytest.fail(message))
    for _ in range(2):
        namespace['ecosystem_verl_training'](session)
    left, right = (Path(env['JITTOR_VERL_EVIDENCE_DIR']) for env in captured)
    assert left != right and left.is_dir() and right.is_dir()
    assert scratch not in left.parents and scratch not in right.parents
    monkeypatch.setenv('JITTOR_VERL_EVIDENCE_DIR', str(tmp_path / 'explicit'))
    namespace['ecosystem_verl_training'](session)
    assert captured[-1]['JITTOR_VERL_EVIDENCE_DIR'] == str(tmp_path / 'explicit')
