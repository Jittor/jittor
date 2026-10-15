"""Offline checks for actual maintained nightly wiring and isolated provisioning."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def provision():
    spec = importlib.util.spec_from_file_location(
        'nightly_provision', ROOT / 'tools/provision_ecosystem_nightly.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nightly_keeps_existing_gate_and_runs_both_complete_maintained_sessions():
    workflow = yaml.load((ROOT / '.github/workflows/ecosystem.yml').read_text(), Loader=yaml.BaseLoader)
    assert workflow['on']['schedule'] and 'workflow_dispatch' in workflow['on']
    assert workflow['jobs']['ecosystem']['steps'][-1]['run'] == 'python -m nox -s ecosystem'
    job = workflow['jobs']['maintained-cpu']
    assert job['strategy']['matrix']['library'] == ['peft', 'swift']
    assert job['strategy']['fail-fast'] == 'false'
    steps = job['steps']
    run = next(step for step in steps if 'tools/provision_ecosystem_nightly.py' in step.get('run', ''))
    assert 'if' not in run and '--run' in run['run']
    assert '${RUNNER_TEMP}/maintained-${{ matrix.library }}' in run['run']
    assert '--preflight' not in run['run']
    assert steps[-1]['if'] == 'always()' and steps[-1]['with']['if-no-files-found'] == 'error'


def test_environment_discards_shim_overlay_and_gpu_selection(provision, tmp_path, monkeypatch):
    for key in ('PYTHONPATH', 'PYTHONHOME', 'REAL_TORCH_SITE', 'REAL_TORCH_PYTHON',
                'JITTOR_ECOSYSTEM_PACKAGE_SITE', 'JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE',
                'JITTOR_SOURCE_ROOT', 'JITTOR_TORCH_RUNTIME_ROOT', 'JT_USE_CUDA', 'CUDA_VISIBLE_DEVICES'):
        monkeypatch.setenv(key, 'hostile-inherited-value')
    env = provision.environment(tmp_path)
    assert env['PYTHONPATH'] == '' and 'PYTHONHOME' not in env
    assert env['JITTOR_TORCH_SHIM'] == '0' and env['CUDA_VISIBLE_DEVICES'] == ''
    assert env['JT_USE_CUDA'] == env['use_cuda'] == '0'
    assert env['JT_BUILD_NVCC_PATH'] == env['nvcc_path'] == ''
    for name in ('JITTOR_ECOSYSTEM_PACKAGE_SITE', 'JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE',
                 'JITTOR_SOURCE_ROOT', 'JITTOR_TORCH_RUNTIME_ROOT', 'REAL_TORCH_SITE'):
        assert name not in env
    for name in ('HOME', 'JITTOR_HOME', 'TMPDIR', 'HF_HOME', 'TORCH_HOME', 'PIP_CACHE_DIR'):
        assert Path(env[name]).is_dir() and tmp_path in Path(env[name]).parents
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '1'


def test_state_refuses_checkout_and_symlink_into_checkout(provision, tmp_path):
    with pytest.raises(ValueError, match='outside'):
        provision.external(ROOT / 'state')
    link = tmp_path / 'inside'
    link.symlink_to(ROOT, target_is_directory=True)
    with pytest.raises(ValueError, match='outside'):
        provision.external(link)


def test_only_oracle_receives_real_torch_wheel(provision, tmp_path):
    for name in ('torch-' + provision.TORCH + '-cp311-cp311-manylinux_2_28_x86_64.whl',
                 'peft-0.17.1-py3-none-any.whl', 'numpy-1.26.4-cp311-cp311-linux_x86_64.whl'):
        (tmp_path / name).touch()
    calls = []
    provision.install_wheels(SimpleNamespace(run=lambda *args: calls.append(args)),
                             'candidate-python', 'oracle-python', tmp_path)
    assert len(calls) == 2
    oracle, candidate = calls
    assert oracle[0] == 'oracle-python' and candidate[0] == 'candidate-python'
    assert '--no-index' in oracle and '--no-deps' in candidate
    assert sum(Path(str(value)).name.startswith('torch-') for value in oracle) == 1
    assert not any(Path(str(value)).name.startswith('torch-') for value in candidate)
    assert set(oracle[6:]) - set(candidate[6:]) == {next(tmp_path.glob('torch-*.whl'))}


@pytest.mark.parametrize('names', [[], ['torch-2.7.1+cpu-old.whl'],
                                   ['torch-2.11.0+cpu-a.whl', 'torch-2.11.0+cpu-b.whl']])
def test_wrong_or_ambiguous_torch_wheel_refuses_before_install(provision, tmp_path, names):
    for name in names:
        (tmp_path / name).touch()
    with pytest.raises(ValueError, match='Exactly one'):
        provision.install_wheels(SimpleNamespace(run=lambda *args: pytest.fail('must not install')),
                                 'candidate', 'oracle', tmp_path)


def test_official_download_verifies_hash_before_wheel_becomes_available(provision, tmp_path, monkeypatch):
    expected = hashlib.sha256(b'official-wheel').hexdigest()
    release = {'urls': [{'packagetype': 'bdist_wheel', 'digests': {'sha256': expected},
                        'url': 'https://files.pythonhosted.org/pinned.whl', 'filename': 'pinned.whl'}]}
    responses = iter((json.dumps(release).encode(), b'tampered-wheel'))
    monkeypatch.setattr(provision.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(next(responses)))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        provision.official_wheel('peft', '0.17.1', expected, tmp_path)
    assert not (tmp_path / 'pinned.whl').exists()


def test_pinned_purelib_install_preserves_record_and_full_payload(provision, tmp_path):
    wheel, site = tmp_path / 'package.whl', tmp_path / 'site'
    payload = {'peft/runtime.py': b'actual source', 'peft/kernels/kernel.cu': b'kernel',
               'peft-0.17.1.dist-info/METADATA': b'Name: peft\nVersion: 0.17.1\n',
               'peft-0.17.1.dist-info/RECORD': b'original wheel RECORD\n'}
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, value in payload.items():
            archive.writestr(name, value)
    provision.unpack_pinned_wheel(wheel, site)
    assert {path.relative_to(site).as_posix(): path.read_bytes()
            for path in site.rglob('*') if path.is_file()} == payload


@pytest.mark.parametrize('unsafe', ['../escape', '/absolute', 'pkg/unsafe\\name',
                                   'distribution.data/scripts/entry'])
def test_unsafe_wheel_is_refused_before_extracting_any_member(provision, tmp_path, unsafe):
    wheel, site = tmp_path / 'bad.whl', tmp_path / 'site'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('safe.py', b'safe')
        archive.writestr(unsafe, b'unsafe')
    with pytest.raises(ValueError, match='layout'):
        provision.unpack_pinned_wheel(wheel, site)
    assert not site.exists()


@pytest.mark.parametrize('wrong', ['remote', 'commit', 'dirty'])
def test_dirty_or_unpinned_existing_upstream_is_rejected(provision, tmp_path, monkeypatch, wrong):
    sha = 'a' * 40
    remote = 'https://github.com/official/repo.git'
    values = {'remote': remote, 'rev-parse': sha, 'status': ''}
    values[{'remote': 'remote', 'commit': 'rev-parse', 'dirty': 'status'}[wrong]] = 'wrong'
    monkeypatch.setattr(provision.subprocess, 'check_output',
                        lambda argv, **kw: values[argv[3]])
    with pytest.raises(ValueError, match='exactly pinned'):
        provision.checkout(SimpleNamespace(), tmp_path, remote, sha)


def test_failed_command_is_retained_and_resume_appends_log(provision, tmp_path, monkeypatch):
    monkeypatch.setattr(provision.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=5))
    commands = provision.Commands(tmp_path, {})
    with pytest.raises(ValueError, match='Command failed'):
        commands.run('git', 'fetch')
    assert json.loads((tmp_path / 'commands.json').read_text())[0]['returncode'] == 5
    (tmp_path / 'logs/000.log').write_text('original failure')
    resumed = provision.Commands(tmp_path, {})
    with pytest.raises(ValueError):
        resumed.run('git', 'fetch')
    assert (tmp_path / 'logs/000.log').read_text() == 'original failure'
    assert (tmp_path / 'logs/001.log').exists()


def test_successful_state_requires_fresh_destination(provision, tmp_path):
    (tmp_path / 'provision.json').write_text('{}')
    with pytest.raises(ValueError, match='new state root'):
        provision.provision('peft', tmp_path, 'must-not-launch')
