"""Negative protocol admission contracts; these fixtures make no hardware claim."""

import copy
import ast
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
    monkeypatch.setenv('JITTOR_ACCELERATE_CHECKPOINT_PHASE', 'consume')
    monkeypatch.setenv('JITTOR_ACCELERATE_CHECKPOINT_DIRECTORY', 'another-checkpoint')
    env = gate.environment('jittor', 'cpu', tmp_path, 0, [], tmp_path)
    assert env['JT_USE_CUDA'] == env['use_cuda'] == '0'
    assert env['CUDA_VISIBLE_DEVICES'] == env['JT_BUILD_NVCC_PATH'] == env['nvcc_path'] == ''
    assert env['PYTHONPATH'] == str(ROOT / 'python')
    assert 'JITTOR_ACCELERATE_CHECKPOINT_PHASE' not in env
    assert 'JITTOR_ACCELERATE_CHECKPOINT_DIRECTORY' not in env


@pytest.mark.parametrize('producer_status', [0, 1])
def test_fresh_checkpoint_consumer_follows_only_a_successful_terminated_producer(tmp_path, monkeypatch, producer_status):
    spec = importlib.util.spec_from_file_location('accelerate_fresh_gate', ROOT / 'tools/run_accelerate_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    monkeypatch.setattr(gate.subprocess, 'check_output', lambda *_args, **_kwargs: '')
    launched = {}

    def execute(command, env, _log):
        if '--warm-only' in command:
            return 0
        phase = command[command.index('--checkpoint-phase') + 1]
        checkpoint = command[command.index('--checkpoint-directory') + 1]
        if phase == 'consume':
            assert launched[checkpoint] == 'producer-returned-successfully'
        elif producer_status:
            return producer_status
        folder = Path(command[command.index('--measurement') + 1])
        folder.mkdir(parents=True)
        (folder / 'junit-rank0.xml').write_text('<testsuite><testcase name="fixture"/></testsuite>')
        launched[checkpoint] = 'producer-returned-successfully' if phase == 'produce' else 'consumer-returned'
        return 0

    monkeypatch.setattr(gate, 'run', execute)
    args = SimpleNamespace(output=tmp_path / 'measurements', cache_root=tmp_path / 'cache',
                           device='cpu', devices='', upstream=tmp_path / 'upstream',
                           oracle_python='oracle-fixture', jittor_python='candidate-fixture', fresh_process=True)
    assert gate.gate(args) == int(bool(producer_status))
    execution = json.loads((args.output / 'execution.json').read_text())
    assert len(execution['results']) == 4
    for result in execution['results']:
        phases = result['phase_launches']
        assert [row['phase'] for row in phases] == (['produce'] if producer_status else ['produce', 'consume'])
        if not producer_status:
            assert phases[1]['producer_exited'] is True
    if producer_status:
        assert not launched


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
            if command[command.index('--runtime') + 1] == 'jittor':
                assert command[1:3] == ['-m', 'jittor.distributed.launch']
                assert '--nproc-per-node=1' in command
                assert '--device-ids=' + ('first', 'second')[rank] in command
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


@pytest.fixture
def fresh_protocol(protocol, tmp_path, monkeypatch):
    from _accelerate_fresh_scorecard import protocol_spec
    import uuid

    row, source, root = protocol
    row['case'] = 'world1-no-fresh-process-checkpoint'
    row['protocol'] = protocol_spec(1, 'no')
    hashes = {str(i): 'd' * 64 for i in range(5)}
    monkeypatch.setattr(accelerate, 'measurement_sources', lambda _baseline, fresh=False: hashes)
    old = row['evidence']['artifacts']
    artifacts = []
    execution = {'results': []}

    def bind(role, path):
        artifacts.append({'role': role, 'path': str(path), 'sha256': digest(path)})

    for label, runtime in (('reference', 'torch'), ('candidate', 'jittor')):
        for scenario, seed in (('base', 0), ('perturb', 7)):
            prefix = '%s-%s-rank0' % (label, scenario)
            metadata = json.loads(Path(next(x['path'] for x in old if x['role'] == prefix + '-metadata')).read_text())
            with np.load(next(x['path'] for x in old if x['role'] == prefix + '-arrays')) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            saved = {'model': {name: arrays['step/2/model/' + name].tolist() for name in ('weight', 'bias')},
                     'optimizer': {'state': {}, 'param_groups': []},
                     'scheduler': {'step_size': 1, 'gamma': .8, 'base_lrs': [.01, .015], 'last_epoch': 3,
                                   '_step_count': 4, '_is_initial': False, '_get_lr_called_within_step': False,
                                   '_last_lr': [float(arrays['step/2/learning_rates/' + i]) for i in ('0', '1')]}}
            for index, name in enumerate(('weight', 'bias')):
                saved['optimizer']['state'][name] = {
                    key: arrays['step/2/optimizer/' + name + '/' + key].tolist() for key in ('step', 'exp_avg', 'exp_avg_sq')}
                group = {'params': [name], 'betas': [.7, .93]}
                for key in ('lr', 'eps', 'weight_decay', 'amsgrad', 'maximize'):
                    group[key] = arrays['step/2/hyperparameters/%d/%s' % (index, key)].tolist()
                saved['optimizer']['param_groups'].append(group)
            checkpoint = tmp_path / (prefix + '-checkpoint')
            checkpoint.mkdir()
            files = {}
            for filename in ('pytorch_model.bin', 'optimizer.bin', 'scheduler.bin', 'random_states_0.pkl'):
                path = checkpoint / filename
                path.write_bytes(b'host identity fixture only')
                files[filename] = digest(path)
                bind(prefix + '-checkpoint-' + filename, path)
            rng = {'python': [3, list(range(625)), None], 'numpy': ['MT19937', list(range(624)), 1, 0, 0.],
                   'torch_cpu': list(range(16))}
            phase_launches = []
            for phase in ('produce', 'consume'):
                value = copy.deepcopy(metadata)
                selected = {key: array for key, array in arrays.items() if key.startswith('initial/')
                            or int(key.split('/')[1]) in (range(5) if phase == 'produce' else range(5, 7))}
                npz = tmp_path / (prefix + '-' + phase + '.npz')
                np.savez(npz, **selected)
                value.update(arrays=str(npz), arrays_sha256=digest(npz),
                             array_manifest={key: value['array_manifest'][key] for key in selected},
                             measurement_source_sha256=hashes)
                if phase == 'consume':
                    value['interpreter']['pid'] += 1000
                value['result'] = {'phase': phase, 'pid': value['interpreter']['pid'],
                                   'process_nonce': str(uuid.uuid5(uuid.NAMESPACE_URL, prefix + phase)),
                                   'checkpoint_directory': str(checkpoint), 'checkpoint_sha256': files,
                                   'observed_monotonic_ns': 15 if phase == 'produce' else 35}
                environment = {}
                for key in ('HOME', 'JITTOR_HOME', 'TMPDIR'):
                    directory = tmp_path / prefix / key
                    directory.mkdir(parents=True, exist_ok=True)
                    environment[key] = str(directory)
                value['result']['process_environment'] = environment
                if phase == 'produce':
                    value['result'].update(before_save=[1, 2, 3], expected_resume=[4, 5], saved=saved,
                                           rng_saved=rng, expected_random=metadata['result']['expected_random'])
                else:
                    value['result'].update(actual_resume=[4, 5], after_restore=saved, before_restore={},
                                           rng_restored=rng, rng_before={**rng, 'torch_cpu': list(range(1, 17))},
                                           resumed_random=metadata['result']['expected_random'])
                path = tmp_path / (prefix + '-' + phase + '.json')
                path.write_text(json.dumps(value))
                bind(prefix + '-' + phase + '-metadata', path)
                bind(prefix + '-' + phase + '-arrays', npz)
                if phase == 'produce' and seed == 0:
                    bind(label, path)
                phase_launches.append({'phase': phase, 'returncode': 0, 'producer_exited': phase == 'consume',
                    'command': [value['interpreter']['executable'], 'gate.py', '--checkpoint-phase', phase,
                                '--checkpoint-directory', str(checkpoint)],
                    'started_monotonic_ns': 10 if phase == 'produce' else 30,
                    'finished_monotonic_ns': 20 if phase == 'produce' else 40})
            execution['results'].append({'runtime': runtime, 'seed': seed, 'world_size': 1, 'precision': 'no',
                                        'checkpoint_mode': 'fresh-process', 'returncode': 0, 'status': 'passed',
                                        'phase_launches': phase_launches})
    execution_path = tmp_path / 'execution.json'
    execution_path.write_text(json.dumps(execution))
    bind('execution', execution_path)
    artifacts.extend(x for x in old if x['role'] == 'platform')
    row['evidence']['artifacts'] = artifacts
    actual = measurements(row, artifacts, source)
    row.update(actual)
    path = tmp_path / 'fresh-comparison.json'
    path.write_text(json.dumps(actual))
    bind('comparison', path)
    return row, source, root


def test_separate_process_protocol_accepts_only_its_own_semantic_scope(fresh_protocol):
    value = result(fresh_protocol)
    assert not value['rejected'] and len(value['accepted']) == 1
    assert value['accepted'][0]['protocol']['restoration'] == 'fresh-process'
    assert value['coverage'] == []
    old = copy.deepcopy(fresh_protocol[0])
    old['protocol']['id'] = 'accelerate-checkpoint'
    assert protocols.pairing_key(old) != protocols.pairing_key(fresh_protocol[0])


@pytest.mark.parametrize('change', [
    lambda value: value['interpreter'].update(pid=42),
    lambda value: (value['interpreter'].update(pid=42), value['result'].update(pid=42)),
    lambda value: value['result'].update(phase='produce'),
    lambda value: value['result'].update(process_nonce='not-a-process-nonce'),
    lambda value: value['result'].update(observed_monotonic_ns=19),
    lambda value: value['result']['checkpoint_sha256'].update({'optimizer.bin': '0' * 64}),
    lambda value: value['result']['after_restore']['optimizer']['state']['weight'].pop('exp_avg_sq'),
    lambda value: value['result']['after_restore']['scheduler'].update(last_epoch=2),
    lambda value: value['result']['rng_restored']['torch_cpu'].append(12),
    lambda value: value['result'].update(rng_before=value['result']['rng_restored']),
    lambda value: value['result']['resumed_random'].update(python=.2),
    lambda value: value['result'].update(actual_resume=[4]),
    lambda value: value.update(fallback_count=1),
    lambda value: value['result']['process_environment'].update(HOME=''),
    lambda value: value['result']['process_environment'].update(HOME=value['result']['process_environment']['TMPDIR']),
])
def test_fresh_process_contract_rejects_rebound_metadata_tampering(fresh_protocol, change):
    mutate_metadata(fresh_protocol, 'candidate-base-rank0-consume-metadata', change)
    assert result(fresh_protocol)['rejected']


def test_missing_adam_moment_in_both_phase_snapshots_is_not_hidden_by_equal_json(fresh_protocol):
    for phase, field in (('produce', 'saved'), ('consume', 'after_restore')):
        mutate_metadata(fresh_protocol, 'candidate-base-rank0-' + phase + '-metadata',
                        lambda value: value['result'][field]['optimizer']['state']['weight'].pop('exp_avg_sq'))
    assert result(fresh_protocol)['rejected']


def test_fresh_checkpoint_failed_partial_output_cannot_be_overwritten(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('accelerate_partial_gate', ROOT / 'tools/run_accelerate_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    monkeypatch.setattr(gate.subprocess, 'check_output', lambda *_args, **_kwargs: '')
    output = tmp_path / 'old-failed-output'
    output.mkdir()
    raw = output / 'producer.log'
    raw.write_text('retained partial failure')
    with pytest.raises(ValueError, match='empty output'):
        gate.gate(SimpleNamespace(output=output, fresh_process=True))
    assert raw.read_text() == 'retained partial failure'


def test_maintained_nox_entry_launches_the_separate_process_protocol(tmp_path, monkeypatch):
    import os
    tree = ast.parse((ROOT / 'noxfile.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'ecosystem_accelerate')
    function.decorator_list = []
    namespace = {'os': os, 'NOX_STATE_ROOT': tmp_path,
                 '_session_env': lambda *_args: (tmp_path, {}), '_hardware_python': lambda: 'candidate-python',
                 '_set_hardware_python_config': lambda *_args: None}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / 'noxfile.py'), 'exec'), namespace)
    monkeypatch.setenv('REAL_TORCH_PYTHON', 'binary-oracle-python')
    monkeypatch.setenv('JITTOR_ACCELERATE_CHECKOUT', 'official-pinned-checkout')
    monkeypatch.setenv('JITTOR_TEST_DEVICES', 'cpu')
    commands = []
    namespace['ecosystem_accelerate'](SimpleNamespace(posargs=[], run=lambda *args, **kwargs: commands.append(args)))
    assert len(commands) == 1
    assert commands[0][:2] == ('candidate-python', 'tools/run_accelerate_gate.py')
    assert '--fresh-process' in commands[0]
