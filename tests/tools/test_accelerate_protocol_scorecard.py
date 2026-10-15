"""Negative protocol admission contracts; these fixtures make no hardware claim."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'))
import audit_coverage as audit  # noqa: E402
import _accelerate_scorecard as accelerate  # noqa: E402
from _accelerate_scorecard import measurements  # noqa: E402
import _protocol_scorecards as protocols  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(accelerate, "measurement_sources", lambda _baseline: {str(index): "d" * 64 for index in range(4)})
    root = tmp_path / 'source'
    (root / 'docs/results').mkdir(parents=True)
    (root / 'docs/results/protocol.md').write_text('fixture only')
    (root / 'gate.py').write_text('def test_real():\n    pass\n')
    binary = tmp_path / 'torch_C.so'
    binary.write_bytes(b'binary identity fixture only')
    source_hash = 'c' * 64
    source = {'repo': 'accelerate', 'axes': [],
              'upstream': {'commit': 'b' * 40, 'remote': 'https://github.com/huggingface/accelerate.git', 'version': '1.10.1'},
              'package_sources': [{'path': 'src/accelerate/__init__.py', 'sha256': source_hash}]}
    precision = {'matmul': False, 'cudnn': False, 'cudnn_benchmark': False, 'matmul_precision': 'highest'}
    oracle = {'executable': 'independent-python', 'pid': 42, 'is_shim': False, 'torch_version': '2.11',
              'binary_extension': str(binary), 'binary_sha256': digest(binary)}
    platform = {'os': 'fixture', 'cpu': 'fixture', 'accelerator_model': 'not-applicable',
                'card_count': 0, 'card_indices': [], 'physical_devices': [], 'driver': 'not-applicable',
                'cuda': 'not-applicable', 'cann': 'not-applicable', 'nvcc': 'not-applicable',
                'gcc': 'fixture', 'nvidia_wheels': {}, 'precision': precision,
                'jittor_flags': {'use_cuda': 0, 'use_acl': 0, 'graph_replay': 'eager', 'tf32': False, 'hf32': False},
                'oracle': {**oracle, 'torch_npu_version': 'not-applicable'}}
    artifacts = []

    def bind(role, path):
        artifacts.append({'role': role, 'path': str(path), 'sha256': digest(path)})

    platform_path = tmp_path / 'platform.json'
    platform_path.write_text(json.dumps(platform))
    bind('platform', platform_path)
    row = {'record_kind': 'protocol', 'repo': 'accelerate', 'case': 'world1-no-checkpoint',
           'protocol': {'id': 'accelerate-checkpoint', 'world_size': 1, 'precision': 'no', 'ranks': [0]},
           'registry_refs': [], 'jittor_commit': 'a' * 40, 'upstream': source['upstream'],
           'status': 'passed', 'level': 3, 'tier': 'core', 'downstream_versions': {'candidate': '1.10.1', 'oracle': '1.10.1'},
           'device': 'cpu', 'observed_device': 'cpu', 'platform': platform,
           'inputs': {'seed': 0, 'hf_hub_offline': True, 'shape_kind': 'tiny'},
           'evidence': {'command': 'fixture', 'report': 'docs/results/protocol.md',
                        'gate_nodeid': 'gate.py::test_real', 'artifacts': artifacts}}
    for label, runtime in (('reference', 'torch'), ('candidate', 'jittor')):
        for scenario, seed in (('base', 0), ('perturb', 7)):
            arrays = {'initial/model/weight': np.ones((2, 3), dtype=np.float32),
                      'initial/model/bias': np.ones(2, dtype=np.float32)}
            for step, effective in enumerate((1, 2, 3, 4, 5, 4, 5)):
                prefix = 'step/%d/' % step
                for name, value in (('input', [[1. + seed * .001] * 3] * 4),
                                    ('target', [[.5] * 2] * 4), ('output', [[1. + seed * .001 - effective * .01] * 2] * 4),
                                    ('loss', 1. - effective * .1), ('input_gradient', [[.1 + seed * .001] * 3] * 4)):
                    arrays[prefix + name] = np.asarray(value, dtype=np.float32)
                for name, shape in (('weight', (2, 3)), ('bias', (2,))):
                    arrays[prefix + 'model/' + name] = np.full(shape, 1. - effective * .01, dtype=np.float32)
                    arrays[prefix + 'parameter_gradients/' + name] = np.full(shape, .1 + seed * .001, dtype=np.float32)
                    for field, value in (('step', effective), ('exp_avg', .02 * effective), ('exp_avg_sq', .003 * effective)):
                        arrays[prefix + 'optimizer/' + name + '/' + field] = np.full(() if field == 'step' else shape, value, dtype=np.float32)
                for group in ('0', '1'):
                    for field, value in (('lr', .01), ('betas/0', .7), ('betas/1', .93), ('eps', 1e-4),
                                         ('weight_decay', .02), ('amsgrad', False), ('maximize', False)):
                        arrays[prefix + 'hyperparameters/' + group + '/' + field] = np.asarray(value)
                    arrays[prefix + 'learning_rates/' + group] = np.asarray(.01)
            prefix = '%s-%s-rank0'  % (label, scenario)
            npz = tmp_path / (prefix + '.npz')
            np.savez(npz, **arrays)
            value = {'schema_version': 1, 'runtime': runtime, 'case': 'single_checkpoint', 'rank': 0,
                     'world_size': 1, 'mixed_precision': 'no', 'jittor_commit': row['jittor_commit'],
                     'source_descriptor': source, 'upstream': source['upstream'], 'device': 'cpu', 'backend': 'none',
                     'measurement_source_sha256': {str(index): 'd' * 64 for index in range(4)},
                     'downstream_distribution': {'version': '1.10.1', 'source_sha256': {'src/accelerate/__init__.py': source_hash},
                                                 'installed_files_sha256': {'__init__.py': source_hash}},
                     'interpreter': oracle if runtime == 'torch' else {**oracle, 'is_shim': True},
                     'native_runtime_flags': {'use_cuda': 0}, 'fallback_policy': 'error', 'fallback_count': 0,
                     'seed': seed, 'hf_hub_offline': True, 'tf32': precision,
                     'runtime_conditions': {'precision': precision}, 'platform': platform,
                     'array_manifest': {key: {'shape': list(array.shape), 'dtype': str(array.dtype), 'device': 'host-metadata' if '/hyperparameters/' in key or '/learning_rates/' in key else 'cpu',
                          'torch_dtype': 'metadata.' + str(array.dtype) if '/hyperparameters/' in key or '/learning_rates/' in key else 'torch.float32'} for key, array in arrays.items()},
                     'arrays': str(npz), 'arrays_sha256': digest(npz),
                     'result': {'before_save': [1, 2, 3], 'expected_resume': [4, 5], 'actual_resume': [4, 5],
                                'saved': {'state': 3}, 'after_restore': {'state': 3},
                                'expected_random': dict.fromkeys(('python', 'numpy', 'torch_cpu', 'torch_device'), .1),
                                'resumed_random': dict.fromkeys(('python', 'numpy', 'torch_cpu', 'torch_device'), .1)}}
            path = tmp_path / (prefix + '.json')
            path.write_text(json.dumps(value))
            bind(prefix + '-metadata', path)
            bind(prefix + '-arrays', npz)
            if seed == 0:
                bind(label, path)
    actual = measurements(row, artifacts, source)
    row.update(actual)
    comparison = tmp_path / 'comparison.json'
    comparison.write_text(json.dumps(actual))
    bind('comparison', comparison)
    return row, source, root


def result(protocol):
    row, source, root = protocol
    return audit.scorecard({'accelerate': source}, [row], row['jittor_commit'], root)


def mutate_metadata(protocol, role, change):
    row, _, _ = protocol
    artifacts = row['evidence']['artifacts']
    item = next(item for item in artifacts if item['role'] == role)
    path = Path(item['path'])
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))
    for artifact in artifacts:
        if artifact['path'] == str(path):
            artifact['sha256'] = digest(path)


def test_protocol_accepted_without_inventing_registry_denominator(protocol):
    value = result(protocol)
    assert not value['rejected'] and len(value['accepted']) == 1
    assert value['coverage'] == []
    assert value['protocols'][0]['matching_cpu_verified'] is True


@pytest.mark.parametrize('change', [
    lambda row: row.pop('record_kind'),
    lambda row: row.update(record_kind='anything'),
    lambda row: row['protocol'].update(world_size=2),
    lambda row: row['protocol'].update(ranks=[0, 0]),
    lambda row: row['protocol'].update(precision='bf16'),
    lambda row: row['protocol'].update(optimizer='SGD'),
    lambda row: row.update(registry_refs=[{'repo': 'accelerate', 'axis': 'fake', 'member': 'fake'}]),
])
def test_protocol_does_not_bypass_registry_or_protocol_identity(protocol, change):
    change(protocol[0])
    assert result(protocol)['rejected']


@pytest.mark.parametrize('change', [
    lambda value: value.update(rank=1),
    lambda value: value.update(world_size=2),
    lambda value: value.update(device='cuda'),
    lambda value: value.update(seed=7),
    lambda value: value.update(jittor_commit='e' * 40),
    lambda value: value.update(fallback_count=1),
    lambda value: value['measurement_source_sha256'].update({'0': 'e' * 64}),
    lambda value: value['downstream_distribution'].pop('installed_files_sha256'),
    lambda value: value['downstream_distribution']['source_sha256'].update({'src/accelerate/__init__.py': 'e' * 64}),
    lambda value: value['result'].update(actual_resume=[4, 6]),
    lambda value: value['result']['resumed_random'].update(python=.2),
])
def test_rehashed_invalid_actual_metadata_is_refused(protocol, change):
    mutate_metadata(protocol, 'candidate-base-rank0-metadata', change)
    assert result(protocol)['rejected']


def test_rehashed_missing_gradient_refused(protocol):
    mutate_metadata(protocol, 'candidate-base-rank0-metadata', lambda value: value['array_manifest'].pop('step/0/input_gradient'))
    assert result(protocol)['rejected']


def test_claimed_own_resume_does_not_override_independent_gradient_failure(protocol):
    row, _, _ = protocol
    item = next(item for item in row['evidence']['artifacts'] if item['role'] == 'candidate-base-rank0-arrays')
    path = Path(item['path'])
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['step/0/parameter_gradients/weight'] += .01
    np.savez(path, **arrays)
    item['sha256'] = digest(path)
    mutate_metadata(protocol, 'candidate-base-rank0-metadata', lambda value: value.update(arrays_sha256=item['sha256']))
    assert 'cross-runtime' in result(protocol)['rejected'][0]['reason']


def test_existing_registry_rows_keep_identical_output(protocol):
    row, _, root = protocol
    # A registry row must retain the old empty-denominator rejection.
    registry_row = copy.deepcopy(row)
    registry_row.pop('record_kind')
    value = audit.scorecard({'accelerate': protocol[1]}, [registry_row], row['jittor_commit'], root)
    assert 'protocols' not in value
    assert value['rejected'] == [{'index': 0, 'case': row['case'], 'reason': 'Missing registered denominator member'}]


def test_cpu_world_one_does_not_pair_cuda_world_two(protocol, monkeypatch):
    row, source, root = protocol
    # Isolate pairing only; measured admission has separate negative tests.
    monkeypatch.setattr(audit, 'validate_record', lambda *_args: 3)
    distributed = copy.deepcopy(row)
    distributed.update(device='cuda', case='world2-no-checkpoint')
    distributed['protocol'] = {'id': 'accelerate-fsdp2-full', 'world_size': 2, 'precision': 'no', 'ranks': [0, 1]}
    single_cuda = dict(row, device='cuda')
    value = audit.scorecard({'accelerate': source}, [row, distributed, single_cuda], row['jittor_commit'], root)
    assert [item['matching_cpu_verified'] for item in value['protocols']] == [True, False, True]
    assert [item['status'] for item in value['protocols']] == ['passed', 'blocked', 'passed']
    assert value['protocols'][1]['level'] is None
    assert value['protocols'][1]['independent_verified_level'] == 3
    assert [item['case'] for item in value['accepted']] == [row['case'], row['case']]
    assert value['blocked'][0]['verified_level'] is None
    assert value['blocked'][0]['level'] is None
    assert value['blocked'][0]['independent_verified_level'] == 3
    assert value['coverage'] == []


def test_unpaired_protocol_is_not_admitted_and_failed_cpu_cannot_pair(protocol, monkeypatch):
    row, source, root = protocol
    def validate(item, *_args):
        if item['device'] == 'cpu':
            raise ValueError('actual CPU gradient mismatch')
        return 3
    monkeypatch.setattr(audit, 'validate_record', validate)
    value = audit.scorecard({'accelerate': source}, [dict(row, device='cuda'), row], row['jittor_commit'], root)
    assert value['accepted'] == []
    assert len(value['blocked']) == len(value['rejected']) == 1
    assert value['protocols'][0]['status'] == 'blocked'
    assert value['protocols'][0]['level'] is None
    assert value['tiers'] == {'core': 0, 'broad': 0, 'tail': 0}


def test_fsdp_snapshot_compares_the_actual_restored_model_optimizer_scope():
    saved = {'model': {'weight': [1.]}, 'optimizer': {'exp_avg': [.2]}, 'loss': .4, 'gradients': [.3]}
    restored = {key: saved[key] for key in ('model', 'optimizer')}
    accelerate.validate_restoration({'saved': saved, 'after_restore': restored}, 2)
    wrong = copy.deepcopy(restored)
    wrong['optimizer']['exp_avg'][0] += .1
    with pytest.raises(ValueError, match='restoration differs'):
        accelerate.validate_restoration({'saved': saved, 'after_restore': wrong}, 2)
    with pytest.raises(ValueError, match='Missing FULL'):
        accelerate.validate_restoration({'saved': saved, 'after_restore': {'model': saved['model']}}, 2)


def test_cpu_environment_overrides_inherited_cuda_and_pins_source(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('accelerate_gate', ROOT / 'tools/run_accelerate_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    monkeypatch.setenv('JT_USE_CUDA', '1')
    monkeypatch.setenv('use_cuda', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'unallocated')
    monkeypatch.setenv('PYTHONPATH', 'different-tree')
    env = gate.environment('jittor', 'cpu', tmp_path, 0, [], tmp_path)
    assert env['JT_USE_CUDA'] == env['use_cuda'] == '0'
    assert env['CUDA_VISIBLE_DEVICES'] == env['JT_BUILD_NVCC_PATH'] == env['nvcc_path'] == ''
    assert env['PYTHONPATH'] == str(ROOT / 'python')


def test_cuda_warm_preserves_launcher_nccl_configuration(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('accelerate_gate', ROOT / 'tools/run_accelerate_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    monkeypatch.setenv('JT_BUILD_NVCC_PATH', 'explicit-nvcc')
    env = gate.environment('jittor', 'cuda', tmp_path, 0, ['first', 'second'], tmp_path)
    assert env['JT_BUILD_USE_NCCL'] == '1'
    assert env['JT_BUILD_USE_MPI'] == '0'


def test_cuda_warm_worker_keeps_its_assigned_cache_rank_without_bootstrap(tmp_path):
    spec = importlib.util.spec_from_file_location('accelerate_gate_warm', ROOT / 'tools/run_accelerate_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    for rank in (0, 1):
        args = SimpleNamespace(warm_only=True, warm_rank=rank)
        env = {}
        selected = gate.worker_rank(args, env)
        gate.rank_environment(env, tmp_path, 'jittor', 'cuda', selected)
        assert env['JT_BUILD_CACHE_NAME'].endswith('rank%d' % rank)
        assert Path(env['JITTOR_HOME']).parent.name == 'rank%d' % rank
        assert Path(env['HOME']).is_dir()
        assert Path(env['HOME']).parent == Path(env['JITTOR_HOME']).parent
        assert Path(env['HOME']) != Path(env['JITTOR_HOME'])
        assert 'RANK' not in env and 'JT_NCCL_RANK' not in env
    assert gate.worker_rank(SimpleNamespace(warm_only=False, warm_rank=0), {'JT_NCCL_RANK': '1'}) == 1


def test_gate_passes_each_serial_warm_its_explicit_cache_rank(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('accelerate_gate_launcher', ROOT / 'tools/run_accelerate_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    monkeypatch.setenv('JT_BUILD_NVCC_PATH', 'fixture-nvcc')
    monkeypatch.setattr(gate.subprocess, 'check_output', lambda *_args, **_kwargs: '')
    warmed = []
    def execute(command, env, _log):
        if '--warm-only' in command:
            rank = int(command[command.index('--warm-rank') + 1])
            worker_args = SimpleNamespace(warm_only=True, warm_rank=rank)
            assert gate.worker_rank(worker_args, env) == rank
            assert env['CUDA_VISIBLE_DEVICES'] == ('first', 'second')[rank]
            assert env['JT_BUILD_CACHE_NAME'].endswith('rank%d' % rank)
            warmed.append((command[command.index('--runtime') + 1], rank))
        else:
            measurement = Path(command[command.index('--measurement') + 1])
            measurement.mkdir(parents=True)
            world = int(command[command.index('--world-size') + 1])
            for rank in range(world):
                (measurement / ('junit-rank%d.xml' % rank)).write_text('<testsuite><testcase name="fixture"/></testsuite>')
        return 0
    monkeypatch.setattr(gate, 'run', execute)
    args = SimpleNamespace(output=tmp_path / 'measurements', cache_root=tmp_path / 'cache',
                           device='cuda', devices='first,second', upstream=tmp_path / 'upstream',
                           oracle_python='oracle-fixture', jittor_python='candidate-fixture')
    assert gate.gate(args) == 0
    assert warmed == [('torch', 0), ('torch', 1), ('jittor', 0), ('jittor', 1)]


def test_dispatch_never_borrows_another_repos_protocol(protocol):
    row, source, _ = protocol
    row, source = copy.deepcopy(row), copy.deepcopy(source)
    row['repo'] = source['repo'] = 'deepspeed'
    with pytest.raises(ValueError, match='No maintained validator'):
        protocols.validate_spec(row, source)


def test_missing_dedicated_validator_refuses_admission(protocol, monkeypatch):
    row, source, _ = protocol
    monkeypatch.setitem(protocols.VALIDATORS, ('accelerate', row['protocol']['id']), '_no_such_protocol_validator')
    with pytest.raises(ValueError, match='No maintained validator'):
        protocols.validate_spec(row, source)


def test_protocol_pairing_keeps_all_semantic_dimensions(protocol):
    row, _, _ = protocol
    first = copy.deepcopy(row)
    first['protocol']['zero_stage'] = 1
    second = copy.deepcopy(first)
    second['protocol']['zero_stage'] = 2
    assert protocols.pairing_key(first) != protocols.pairing_key(second)
