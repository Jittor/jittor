"""A required FlashAttention session keeps a genuine, isolated CUDA oracle."""

from pathlib import Path
import os

import pytest

from _helpers.nox_probe import _FakeSession, _load_noxfile


def configure(monkeypatch, tmp_path):
    paths = {}
    for name in ("JITTOR_FLASH_ATTN_JITTOR_SRC", "JITTOR_FLASH_ATTN_ORACLE_CUDA_HOME"):
        path = tmp_path / name.lower()
        path.mkdir()
        paths[name] = str(path)
    extension = tmp_path / "flash_attn_2_cuda.so"
    extension.write_bytes(b"session configuration fixture; never imported")
    paths["JITTOR_FLASH_ATTN_ORACLE_EXTENSION"] = str(extension)
    proof = tmp_path / 'source-proof.json'
    proof.write_text('{}')
    paths['JITTOR_FLASH_ATTN_SOURCE_PROOF'] = str(proof)
    paths.update(REAL_TORCH_PYTHON="/opt/oracle/bin/python", JITTOR_CI_PYTHON="/opt/native/bin/python",
                 JITTOR_TEST_DEVICES="cuda", CUDA_VISIBLE_DEVICES="2", JT_BUILD_NVCC_PATH="/opt/cuda/bin/nvcc",
                 JITTOR_SOURCE_ROOT="/foreign/source", OMP_NUM_THREADS="3")
    for name, value in paths.items():
        monkeypatch.setenv(name, value)
    return paths


def test_flash_session_requires_execution_and_isolates_oracle(monkeypatch, tmp_path):
    paths = configure(monkeypatch, tmp_path)
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/opt/native/bin/python3-config")
    session.python = False
    run = session.run

    def check_oracle_state_before_workers(*args, **kwargs):
        if 'pytest' in args:
            assert Path(kwargs['env']['JITTOR_FLASH_ATTN_ORACLE_STATE']).is_dir()
        return run(*args, **kwargs)

    monkeypatch.setattr(session, 'run', check_oracle_state_before_workers)
    module["ecosystem_flash_attention"](session)
    oracle = [kw["env"] for args, kw in session.calls if args[0] == "/opt/oracle/bin/python"]
    assert len(oracle) == 1
    assert not oracle[0]["PYTHONPATH"] and not oracle[0]["LD_LIBRARY_PATH"]
    assert not oracle[0]["JITTOR_SOURCE_ROOT"] and oracle[0]["JITTOR_TORCH_SHIM"] == "0"
    assert oracle[0]["CUDA_HOME"] == paths["JITTOR_FLASH_ATTN_ORACLE_CUDA_HOME"]
    executions = [(args, kw["env"]) for args, kw in session.calls if "pytest" in args]
    assert len(executions) == 1
    args, env = executions[0]
    assert "compat/tests/torch/test_ecosystem_flash_attention.py" in args
    assert env["JITTOR_REQUIRE_FLASH_ATTENTION"] == env["JITTOR_REQUIRE_REAL_TORCH"] == "1"
    assert env["JITTOR_FLASH_ATTN_JITTOR_REQUIRED"] == env["JITTOR_TEST_REQUIRE_EXECUTION"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert {env[name] for name in module["THREAD_POOL_ENV_NAMES"]} == {"3"}
    assert env["JITTOR_HOME"].endswith("ecosystem-flash-attention/jittor-home")
    assert Path(env['TMPDIR']) not in Path(env['JITTOR_TORCH_CACHE_ROOT']).parents
    assert env['JITTOR_FLASH_ATTN_SOURCE_PROOF'] == paths['JITTOR_FLASH_ATTN_SOURCE_PROOF']
    collectors = [(args, kw) for args, kw in session.calls if any('collect_scorecard.py' in str(a) for a in args)]
    assert len(collectors) == 1
    assert '--source-proof' in collectors[0][0] and '--baseline' in collectors[0][0]
    assert Path(env['TMPDIR']) not in Path(env['JITTOR_FLASH_ATTN_ORACLE_STATE']).parents
    deployments = [args for args, _ in session.calls if 'jittor.compat.shim.deploy' in args]
    assert len(deployments) == 2 and '--target' in deployments[0] and '--check' in deployments[1]
    target = deployments[0][deployments[0].index('--target') + 1]
    assert target in env['PYTHONPATH'].split(os.pathsep)
    assert target not in oracle[0]['PYTHONPATH']


@pytest.mark.parametrize("missing", ("upstream", "extension", "cuda_home", "oracle", "cpu", "allocation", "nvcc", "source_proof"))
def test_flash_session_rejects_missing_configuration_before_interpreter(monkeypatch, tmp_path, missing):
    configure(monkeypatch, tmp_path)
    names = {"upstream": "JITTOR_FLASH_ATTN_JITTOR_SRC", "extension": "JITTOR_FLASH_ATTN_ORACLE_EXTENSION",
             "cuda_home": "JITTOR_FLASH_ATTN_ORACLE_CUDA_HOME", "oracle": "REAL_TORCH_PYTHON",
             "allocation": "CUDA_VISIBLE_DEVICES", 'source_proof': 'JITTOR_FLASH_ATTN_SOURCE_PROOF'}
    if missing in names:
        monkeypatch.delenv(names[missing])
    elif missing == "cpu":
        monkeypatch.setenv("JITTOR_TEST_DEVICES", "cpu")
    else:
        monkeypatch.delenv("JT_BUILD_NVCC_PATH")
        monkeypatch.delenv("nvcc_path", raising=False)
        monkeypatch.setenv("PATH", "")
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/opt/native/bin/python3-config")
    session.python = False
    with pytest.raises(AssertionError, match="FlashAttention"):
        module["ecosystem_flash_attention"](session)
    assert not session.calls
