"""The real vLLM component session must fail closed and isolate its oracle."""
import ast
import runpy
from pathlib import Path

import pytest

from _helpers.nox_probe import _FakeSession, _load_noxfile


@pytest.mark.parametrize('device', ('cpu', 'cuda'))
def test_vllm_session_binds_gate_device_versions_and_oracle(monkeypatch, tmp_path, device):
    oracle = tmp_path / 'oracle-python'
    oracle.touch()
    monkeypatch.setenv('REAL_TORCH_PYTHON', str(oracle))
    monkeypatch.setenv('JITTOR_CI_PYTHON', '/opt/candidate/bin/python')
    monkeypatch.setenv('JITTOR_TEST_DEVICES', device)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'allocated')
    monkeypatch.setenv('JT_BUILD_NVCC_PATH', '/opt/cuda/bin/nvcc')
    monkeypatch.setenv('JITTOR_SOURCE_ROOT', '/foreign/source')
    monkeypatch.setenv('JITTOR_ECOSYSTEM_SPEED_RATIO', '1.01')
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, '/opt/candidate/bin/python3-config')
    session.python = False
    module['ecosystem_vllm'](session)
    calls = [(args, kw['env']) for args, kw in session.calls if 'pytest' in args]
    assert len(calls) == 1
    arguments, env = calls[0]
    assert 'compat/tests/torch/test_ecosystem_vllm.py' in arguments
    assert arguments[0] == '/opt/candidate/bin/python'
    for flag in ('JITTOR_REQUIRE_REAL_TORCH', 'JITTOR_REQUIRE_VLLM',
                 'JITTOR_REQUIRE_TRANSFORMERS', 'JITTOR_TEST_REQUIRE_EXECUTION',
                 'JITTOR_ECOSYSTEM_VLLM', 'JITTOR_TORCH_SHIM'):
        assert env[flag] == '1'
    assert env['backend_fallback'] == 'error'
    assert env['JITTOR_ECOSYSTEM_SPEED_RATIO'] == ''
    assert env['JITTOR_ECOSYSTEM_TF32'] == '0'
    assert env['JT_BACKEND'] == env['JITTOR_TEST_DEVICES'] == device
    assert env['JT_USE_CUDA'] == env['use_cuda'] == str(int(device == 'cuda'))
    assert {env[name] for name in module['THREAD_POOL_ENV_NAMES']} == {'1'}
    assert env['cache_name'] == 'nox_ecosystem_vllm_' + device
    assert env['JITTOR_HOME'].endswith('/ecosystem-vllm/' + device + '/jittor')
    assert env['CUDA_VISIBLE_DEVICES'] == ('allocated' if device == 'cuda' else '')
    assert env['nvcc_path'] == env['JT_BUILD_NVCC_PATH'] == (
        '/opt/cuda/bin/nvcc' if device == 'cuda' else '')
    oracle_calls = [(args, kw['env']) for args, kw in session.calls if args[0] == str(oracle)]
    assert len(oracle_calls) == 1
    command, oracle_env = oracle_calls[0]
    assert "'2.13.0'" in command[-1]
    assert "'0.29.0'" in command[-1] and "'5.13.0'" in command[-1]
    assert '_torch_compat_install_context' in command[-1]
    assert not oracle_env['PYTHONPATH'] and not oracle_env['JITTOR_SOURCE_ROOT']
    assert oracle_env['JITTOR_TORCH_SHIM'] == '0'
    assert ('assert torch.cuda.is_available()' if device == 'cuda' else
            'assert not torch.cuda.is_available()') in command[-1]


@pytest.mark.parametrize('missing', ('oracle', 'oracle-file', 'device', 'allocation', 'compiler'))
def test_vllm_session_rejects_incomplete_configuration(monkeypatch, tmp_path, missing):
    oracle = tmp_path / 'oracle-python'
    oracle.touch()
    monkeypatch.setenv('REAL_TORCH_PYTHON', str(oracle))
    monkeypatch.setenv('JITTOR_TEST_DEVICES', 'cuda')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'allocated')
    monkeypatch.setenv('JT_BUILD_NVCC_PATH', '/opt/cuda/bin/nvcc')
    if missing == 'oracle':
        monkeypatch.delenv('REAL_TORCH_PYTHON')
    elif missing == 'oracle-file':
        monkeypatch.setenv('REAL_TORCH_PYTHON', str(tmp_path / 'missing'))
    elif missing == 'device':
        monkeypatch.setenv('JITTOR_TEST_DEVICES', 'cpu,cuda')
    elif missing == 'allocation':
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    else:
        monkeypatch.delenv('JT_BUILD_NVCC_PATH')
        monkeypatch.delenv('nvcc_path', raising=False)
        monkeypatch.setenv('PATH', '')
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, '/opt/candidate/bin/python3-config')
    session.python = False
    with pytest.raises(AssertionError, match='vLLM'):
        module['ecosystem_vllm'](session)
    assert not session.calls


def test_vllm_case_and_full_input_gradient_contract_are_registered():
    root = Path(__file__).resolve().parents[2]
    source = root / 'compat/tests/torch/_ecosystem_cases.py'
    tree = ast.parse(source.read_text())
    mapping = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'CASES'
                           for target in node.targets))
    assert isinstance(mapping, ast.Dict)
    entry = {ast.literal_eval(key): value for key, value in zip(mapping.keys, mapping.values)}
    assert ast.literal_eval(entry['vllm_decoder_primitives'].elts[1]) == ('vllm', 'transformers')
    case = runpy.run_path(str(root / "compat/tests/torch/_ecosystem_vllm_cases.py"))
    assert set(case["REQUIRED_GRADIENTS"]) == {
        'ingrad::gates', 'ingrad::hidden_states', 'ingrad::query', 'ingrad::key',
    }
