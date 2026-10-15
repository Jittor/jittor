"""Read actual producer/consumer artifacts before shared numerical validation."""

import copy
import hashlib
import json
from pathlib import Path
import uuid

from _accelerate_scorecard import require


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_spec(world, precision):
    return {'id': 'accelerate-checkpoint-fresh-process' if world == 1 else 'accelerate-fsdp2-full-fresh-process',
            'world_size': world, 'precision': precision, 'ranks': list(range(world)),
            'restoration': 'fresh-process', 'checkpoint': 'FULL_STATE_DICT' if world == 2 else 'Accelerator',
            'model': 'Linear(4,3)' if world == 2 else 'Linear(3,2)', 'optimizer': 'AdamW',
            'scheduler': 'none' if world == 2 else 'StepLR(step_size=1,gamma=0.8)',
            'pre_save_steps': 3, 'resume_steps': 2, 'input_seeds': [0, 7],
            'rng_state': 'full-advanced-streams'}


def validate_rng(value, device):
    keys = {'python', 'numpy', 'torch_cpu'} | ({'torch_cuda'} if device == 'cuda' else set())
    require(isinstance(value, dict) and set(value) == keys, 'Missing full checkpoint RNG streams')
    py, np = value['python'], value['numpy']
    require(isinstance(py, list) and len(py) == 3 and py[0] == 3 and isinstance(py[1], list) and len(py[1]) == 625,
            'Incomplete Python random state')
    require(isinstance(np, list) and len(np) == 5 and np[0] == 'MT19937' and isinstance(np[1], list) and len(np[1]) == 624,
            'Incomplete NumPy random state')
    require(isinstance(value.get('torch_cuda', []), list), 'Malformed CUDA generator states')
    for stream in [value['torch_cpu']] + value.get('torch_cuda', []):
        require(isinstance(stream, list) and len(stream) >= 16
                and all(type(item) is int and 0 <= item <= 255 for item in stream),
                'Missing actual opaque Torch generator state')
    if device == 'cuda':
        require(value['torch_cuda'], 'Missing actual CUDA generator states')


def observation(role, load, record, runtime, seed, rank):
    import numpy as np

    spec = record['protocol']
    execution = json.loads(load('execution').read_text())
    entries = [row for row in execution.get('results', []) if row.get('runtime') == runtime
               and row.get('world_size') == spec['world_size'] and row.get('precision') == spec['precision']
               and row.get('seed') == seed]
    require(len(entries) == 1 and entries[0].get('checkpoint_mode') == 'fresh-process'
            and entries[0].get('status') == 'passed' and entries[0].get('returncode') == 0,
            'Missing successful separate-process launcher execution')
    launches = entries[0].get('phase_launches', [])
    require(len(launches) == 2 and [row.get('phase') for row in launches] == ['produce', 'consume']
            and all(row.get('returncode') == 0 for row in launches) and launches[1].get('producer_exited') is True,
            'Consumer was not launched after successful producer termination')
    require(all(type(row.get(key)) is int for row in launches
                for key in ('started_monotonic_ns', 'finished_monotonic_ns'))
            and launches[0]['started_monotonic_ns'] < launches[0]['finished_monotonic_ns']
            <= launches[1]['started_monotonic_ns'] < launches[1]['finished_monotonic_ns'],
            'Producer/consumer launcher lifetime overlaps or is missing')
    values, arrays = [], []
    for phase, launch in zip(('produce', 'consume'), launches):
        prefix = role + '-' + phase
        value = json.loads(load(prefix + '-metadata').read_text())
        result = value.get('result', {})
        identity = value.get('interpreter', {})
        require(result.get('phase') == phase and type(result.get('pid')) is int
                and result['pid'] == identity.get('pid') and result['pid'] > 0,
                'Actual process phase/PID is missing')
        nonce = result.get('process_nonce', '')
        try:
            require(str(uuid.UUID(nonce)) == nonce, 'Invalid process nonce')
        except (ValueError, AttributeError, TypeError) as error:
            raise ValueError('Invalid process nonce') from error
        observed = result.get('observed_monotonic_ns')
        require(type(observed) is int and launch['started_monotonic_ns'] <= observed <= launch['finished_monotonic_ns'],
                'Observed worker does not belong to actual launcher lifetime')
        environment = result.get('process_environment', {})
        directories = [environment.get(key) for key in ('HOME', 'JITTOR_HOME', 'TMPDIR')]
        require(all(isinstance(path, str) and Path(path).is_absolute() and Path(path).is_dir() for path in directories)
                and len(set(directories)) == 3, 'Missing independent actual HOME/JITTOR_HOME/TMPDIR')
        if spec['world_size'] == 2:
            require(environment.get('RANK') == str(rank) and environment.get('WORLD_SIZE') == '2',
                    'Actual rank environment differs')
            peer = role.rsplit('-rank', 1)[0] + '-rank%d-' % (1 - rank) + phase + '-metadata'
            peer_env = json.loads(load(peer).read_text()).get('result', {}).get('process_environment', {})
            require(all(environment[key] != peer_env.get(key) for key in ('HOME', 'JITTOR_HOME', 'TMPDIR')),
                    'Ranks share a private cache or home directory')
            control_keys = ('JITTOR_DIST_RENDEZVOUS_DIR',) if runtime == 'jittor' else ('MASTER_ADDR', 'MASTER_PORT')
            require(all(environment.get(key) and environment[key] == peer_env.get(key) for key in control_keys),
                    'Ranks do not share their actual communicator control plane')
        command = launch.get('command', [])
        require(isinstance(command, list) and len(command) >= 3 and all(isinstance(item, str) for item in command)
                and isinstance(identity.get('executable'), str) and identity['executable']
                and '--checkpoint-phase' in command and command[command.index('--checkpoint-phase') + 1] == phase
                and '--checkpoint-directory' in command
                and Path(command[command.index('--checkpoint-directory') + 1]).resolve()
                    == Path(result.get('checkpoint_directory', '')).resolve()
                and Path(command[0]).resolve() == Path(identity.get('executable', '')).resolve(),
                'Observed process is not bound to actual checkpoint launcher command')
        if spec['world_size'] == 2:
            module = 'torch.distributed.run' if runtime == 'torch' else 'jittor.distributed.launch'
            require(len(command) > 2 and command[1:3] == ['-m', module], 'Fresh FSDP requires the public launcher')
        data = load(prefix + '-arrays')
        require(value.get('arrays_sha256') == digest(data) and Path(value.get('arrays', '')).resolve() == data,
                'Phase NPZ binding differs')
        with np.load(data, allow_pickle=False) as archive:
            phase_arrays = {key: archive[key].copy() for key in archive.files}
        require(set(phase_arrays) == set(value.get('array_manifest', {})), 'Phase tensor manifest differs')
        expected_steps = range(5) if phase == 'produce' else range(5, 7)
        require(all(key.startswith('initial/model/') or any(key.startswith('step/%d/' % i) for i in expected_steps)
                    for key in phase_arrays), 'Tensor belongs to another process phase')
        values.append(value)
        arrays.append(phase_arrays)
    producer, consumer = values
    p, c = producer['result'], consumer['result']
    require(p['pid'] != c['pid'] and p['process_nonce'] != c['process_nonce'], 'Checkpoint restore reused a process')
    ignored = {'result', 'interpreter', 'arrays', 'arrays_sha256', 'array_manifest'}
    require({k: v for k, v in producer.items() if k not in ignored}
            == {k: v for k, v in consumer.items() if k not in ignored}, 'Consumer runtime/source/platform differs')
    require({k: v for k, v in producer['interpreter'].items() if k != 'pid'}
            == {k: v for k, v in consumer['interpreter'].items() if k != 'pid'}, 'Consumer interpreter/binary differs')
    require(isinstance(p.get('checkpoint_sha256'), dict)
            and p.get('checkpoint_directory') == c.get('checkpoint_directory')
            and p.get('checkpoint_sha256') == c.get('checkpoint_sha256') and p.get('checkpoint_sha256'),
            'Persisted checkpoint changed between processes')
    for filename, sha256 in p['checkpoint_sha256'].items():
        path = load(role + '-checkpoint-' + filename)
        require(path == (Path(p['checkpoint_directory']) / filename).resolve()
                and digest(path) == sha256, 'Persisted checkpoint file differs')
    files = set(p['checkpoint_sha256'])
    require('random_states_%d.pkl' % rank in files and 'optimizer.bin' in files
            and ('pytorch_model_fsdp.bin' if spec['world_size'] == 2 else 'pytorch_model.bin') in files
            and (spec['world_size'] == 2 or 'scheduler.bin' in files), 'Missing complete actual checkpoint files')
    require(set(p.get('saved', {})) == {'model', 'optimizer', 'scheduler'}
            and p['saved'] == c.get('after_restore') and c.get('before_restore') != c['after_restore'],
            'New model/optimizer/scheduler did not restore the complete saved state')
    require((p['saved']['scheduler'] is None) == (spec['world_size'] == 2), 'Protocol scheduler scope differs')
    saved, observed = p['saved'], arrays[0]
    names = ('weight', 'bias')
    require(isinstance(saved['model'], dict) and set(saved['model']) == set(names)
            and isinstance(saved['optimizer'], dict) and set(saved['optimizer']) == {'state', 'param_groups'}
            and isinstance(saved['optimizer']['state'], dict) and set(saved['optimizer']['state']) == set(names),
            'Saved FULL model/optimizer schema is incomplete')
    for name in names:
        entry = saved['optimizer']['state'][name]
        require(isinstance(entry, dict) and set(entry) == {'step', 'exp_avg', 'exp_avg_sq'},
                'Missing complete saved Adam moments/step')
        for suffix, value in [('model/' + name, saved['model'][name])] + [
                ('optimizer/' + name + '/' + key, entry[key]) for key in entry]:
            key = 'step/2/' + suffix
            require(key in observed and np.array_equal(np.asarray(value), observed[key]),
                    'Saved checkpoint snapshot differs from observed model/optimizer tensors')
    groups = saved['optimizer']['param_groups']
    require(isinstance(groups, list) and len(groups) == 2
            and all(group.get('params') == [name] for group, name in zip(groups, names)),
            'Saved optimizer parameter FQN groups differ')
    for index, group in enumerate(groups):
        for key in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize'):
            expected = (np.asarray([observed['step/2/hyperparameters/%d/betas/%d' % (index, i)] for i in (0, 1)])
                        if key == 'betas' else observed.get('step/2/hyperparameters/%d/%s' % (index, key)))
            require(key in group and expected is not None and np.array_equal(np.asarray(group[key]), expected),
                    'Saved optimizer hyperparameters differ from observed training')
    if spec['world_size'] == 1:
        scheduler = saved['scheduler']
        expected = {'step_size': 1, 'gamma': .8, 'base_lrs': [.01, .015], 'last_epoch': 3,
                    '_step_count': 4, '_is_initial': False, '_get_lr_called_within_step': False,
                    '_last_lr': [float(observed['step/2/learning_rates/%d' % i]) for i in (0, 1)]}
        require(scheduler == expected, 'Saved complete StepLR state differs from observed learning rates')
    for value in (p.get('rng_initial'), p.get('rng_saved'), c.get('rng_restored'), c.get('rng_before')):
        validate_rng(value, record['device'])
    require(all(p['rng_initial'][key] != p['rng_saved'][key] for key in p['rng_saved'])
            and isinstance(p.get('advanced_random'), dict)
            and set(p['advanced_random']) == set(p.get('expected_random', {})),
            'Each saved RNG stream must actually advance before checkpointing')
    require(p['rng_saved'] == c['rng_restored'] and c['rng_before'] != c['rng_restored'],
            'Full generator state was not restored in the new process')
    require(p.get('expected_random') == c.get('resumed_random'), 'Fresh-process next random samples differ')
    require(len(p.get('before_save', [])) == 3 and len(p.get('expected_resume', [])) == 2
            and p['expected_resume'] == c.get('actual_resume'), 'Fresh-process training trajectory differs')
    if spec['world_size'] == 2:
        observations = [value.get('distributed_observation', {}) for value in (p, c)]
        for observation, phase in zip(observations, (p, c)):
            require(observation.get('pid') == phase['pid'], 'FSDP shard observation belongs to another process')
        excluded = {'pid', 'parameter_pointers'}
        require({k: v for k, v in observations[0].items() if k not in excluded}
                == {k: v for k, v in observations[1].items() if k not in excluded},
                'Producer and new consumer FSDP shard/provider semantics differ')
        for observation in observations:
            require(all(observation.get('parameter_pointers', {}).get(name, 0) > 0
                        for name, shape in observation.get('parameter_shapes', {}).items() if all(shape)),
                    'Producer/consumer FSDP shard lacks physical storage')
    if spec['precision'] == 'bf16':
        require(len(p.get('compute_observations', [])) == 5 and len(c.get('compute_observations', [])) == 2,
                'BF16 compute observations do not belong to their actual process phases')
    overlap = arrays[0].keys() & arrays[1].keys()
    require(overlap and all(key.startswith('initial/model/') and np.array_equal(arrays[0][key], arrays[1][key])
                           and producer['array_manifest'][key] == consumer['array_manifest'][key]
                           for key in overlap), 'Producer/consumer fixture overlap differs')
    # Union actual disjoint phase observations in memory for the existing
    # seven-step numerical checker. No combined measurement is written or
    # presented as if a single process observed both phases.
    result = copy.deepcopy(producer)
    result['array_manifest'] = {**producer['array_manifest'], **consumer['array_manifest']}
    result['result'] = {**p, **c, 'compute_observations': p.get('compute_observations', []) + c.get('compute_observations', [])}
    return result, {**arrays[0], **arrays[1]}
