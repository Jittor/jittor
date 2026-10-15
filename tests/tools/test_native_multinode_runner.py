"""Host-only execution/provenance checks for the two-host acceptance entry."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from _helpers.child_process import PYTHON, child_env, run_python_child
from _helpers.nox_probe import _FakeSession, _load_noxfile


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'agent/skills/jittor-distributed-verification'


@pytest.fixture(scope='module')
def runner():
    spec = importlib.util.spec_from_file_location('native_multinode_runner', DIRECTORY / 'run_native_multinode.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'regular').write_text('content')
    (root / 'linked').symlink_to('regular')
    for command in (['init', '-q'], ['add', 'regular', 'linked'],
                    ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture']):
        subprocess.run(['git', '-C', str(root), *command], check=True, capture_output=True)
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    return root, sha


def arguments(source, tmp_path, **values):
    root, sha = source
    defaults = dict(source_root=root, source_sha=sha, python=Path(PYTHON), oracle_python=Path(PYTHON),
                    nvcc=Path(PYTHON), state_root=tmp_path / 'state', shared_output=tmp_path / 'output',
                    job_id='test-job', node_rank=0, master_addr='127.0.0.1', master_port=29573,
                    device_id='allocated-device', stage='plan', timeout=30, rendezvous_timeout=20)
    defaults.update(values)
    return argparse.Namespace(**defaults)


def test_source_snapshot_includes_symlink_target_text(runner, source):
    root, sha = source
    snapshot = runner.source_snapshot(root, sha)
    assert snapshot['git_modes']['linked'] == '120000'
    assert snapshot['files']['linked'] == hashlib.sha256(b'regular').hexdigest()
    assert snapshot['files']['regular'] == hashlib.sha256(b'content').hexdigest()
    (root / 'regular').write_text('changed')
    with pytest.raises(ValueError, match='clean checkout'):
        runner.source_snapshot(root, sha)


@pytest.mark.parametrize('values,message', [
    ({'device_id': '0,1'}, 'exactly one'), ({'master_port': 0}, 'nonzero'),
    ({'job_id': '../job'}, 'invalid form'), ({'timeout': float('nan')}, 'finite'),
    ({'node_rank': 1, 'stage': 'seal'}, 'node zero'),
])
def test_invalid_protocol_declarations_refuse(runner, source, tmp_path, values, message):
    with pytest.raises(ValueError, match=message):
        runner.validate(arguments(source, tmp_path, **values))


def test_state_output_and_checkout_must_be_disjoint(runner, source, tmp_path):
    with pytest.raises(ValueError, match='disjoint'):
        runner.validate(arguments(source, tmp_path, state_root=tmp_path / 'output/cache'))
    with pytest.raises(ValueError, match='outside'):
        runner.validate(arguments(source, tmp_path, shared_output=source[0] / 'output'))


def test_plan_is_an_actual_host_command_without_device_claim(source, tmp_path):
    root, sha = source
    result = run_python_child([str(DIRECTORY / 'run_native_multinode.py'), '--stage', 'plan',
        '--source-root', str(root), '--source-sha', sha, '--python', PYTHON,
        '--oracle-python', PYTHON, '--nvcc', PYTHON, '--state-root', str(tmp_path / 'state'),
        '--shared-output', str(tmp_path / 'output'), '--node-rank', '1',
        '--master-addr', '127.0.0.1', '--master-port', '29573', '--job-id', 'host-plan',
        '--device-id', 'explicit-device'], merge_stderr=True)
    assert result.returncode == 0, result.stdout
    plan = json.loads(result.stdout)
    assert plan['world_size'] == 2 and plan['local_world_size'] == 1 and plan['node_rank'] == 1
    assert plan['host_id'] and plan['scope'] == 'plan only; no device or connectivity claim'
    assert not (tmp_path / 'output').exists()


def test_public_launcher_and_singleton_warm_rank_are_distinct(runner, source, tmp_path):
    args = arguments(source, tmp_path, stage='parity', node_rank=1)
    command = runner.launcher_command(args)
    assert command[1:3] == ['-m', 'jittor.distributed.launch']
    assert '--nnodes=2' in command and '--node-rank=1' in command
    assert '--nproc-per-node=1' in command and '--rdzv-id=test-job-parity' in command
    args.stage = 'warm'
    command = runner.launcher_command(args)
    assert '--standalone' in command and '--master-addr=127.0.0.1' in command
    assert '--master-port=0' in command and '--nnodes=2' not in command
    env = runner.environment(args)
    assert env['HOME'] != env['JITTOR_HOME'] != env['TMPDIR']
    assert 'RANK' not in env and 'JT_NCCL_RANK' not in env


def test_missing_node_proof_cannot_seal(runner, source, tmp_path):
    args = arguments(source, tmp_path, stage='seal')
    with pytest.raises(FileNotFoundError):
        runner.seal(args)
    assert not (args.shared_output / 'raw-bindings.json').exists()


def test_nox_preserves_explicit_per_host_arguments(monkeypatch, tmp_path):
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, '/usr/bin/python3-config')
    session.posargs = ('--stage', 'plan', '--node-rank', '1')
    module['native_multinode'](session)
    command, options = session.calls[-1]
    assert command[-4:] == session.posargs
    assert command[1].endswith('run_native_multinode.py')
    assert options['external'] is True


def test_actual_child_record_excludes_unrelated_environment(runner, source, tmp_path, monkeypatch):
    monkeypatch.setenv('MULTINODE_TEST_PRIVATE_VALUE', 'do-not-record')
    args = arguments(source, tmp_path)
    runner.run_command(args, [PYTHON, '-c', 'print("child-completed")'],
                       child_env(runner.environment(args), inherit=False), 'host-child')
    record = json.loads((args.shared_output / 'commands/host-child.json').read_text())
    assert record['returncode'] == 0
    assert 'MULTINODE_TEST_PRIVATE_VALUE' not in record['env']
    assert record['env']['HOME'] == str(args.state_root / 'worker/home')
    with pytest.raises(FileExistsError):
        runner.run_command(args, [PYTHON, '-c', 'raise RuntimeError("must not execute")'],
                           child_env(runner.environment(args), inherit=False), 'host-child')


def test_actual_host_identity_disambiguates_local_paths_and_devices():
    spec = importlib.util.spec_from_file_location('multinode_recheck', DIRECTORY / 'recheck_native_fsdp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = {}
    for index, mode in enumerate(('parity', 'baseline', 'resume')):
        for rank in (0, 1):
            records[mode, rank] = {
                'host_id': 'actual-host-' + str(rank), 'pid': 100 + index,
                'parent_pid': 200 + index, 'start_ns': index + 1,
                'home': '/local/home', 'jittor_home': '/local/jittor', 'tmpdir': '/local/tmp',
                'visible_devices': '0',
            }
    module._worker_isolation(records)
    for value in records.values():
        value['host_id'] = 'same-actual-host'
    with pytest.raises(ValueError, match='rank isolation'):
        module._worker_isolation(records)
    records['resume', 0].pop('host_id')
    with pytest.raises(ValueError, match='partial host'):
        module._worker_isolation(records)
    for value in records.values():
        value.pop('host_id', None)
    with pytest.raises(ValueError, match='rank isolation'):
        module._worker_isolation(records)
