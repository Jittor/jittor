"""Fail-closed, non-registry Accelerate checkpoint protocol evidence.

This validates measured tensors, rather than trusting a declared level or a
successful own-runtime checkpoint assertion. Existing registry rows never use
this path. Tolerances are the original checkpoint oracle budget.
"""

import hashlib
import json
import math
from pathlib import Path
import subprocess

from registry_inventory import ROOT


ATOL, RTOL = 2e-6, 2e-5
MEASUREMENT_SOURCES = ('compat/tests/torch/_accelerate_protocol.py',
                       'compat/tests/torch/test_accelerate_distributed_integration.py',
                       'compat/tests/torch/test_optimizer_checkpoint_state.py',
                       'tools/run_accelerate_gate.py')


def measurement_sources(baseline, fresh=False):
    paths = MEASUREMENT_SOURCES + (('compat/tests/torch/_accelerate_fresh_process.py',) if fresh else ())
    try:
        return {path: hashlib.sha256(subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', baseline + ':' + path], stderr=subprocess.PIPE)).hexdigest()
                for path in paths}
    except subprocess.CalledProcessError as error:
        raise ValueError('Unknown measured protocol source commit/files') from error


PROTOCOLS = {'accelerate-checkpoint' : (1, ('no',)), 'accelerate-fsdp2-full': (2, ('no', 'bf16'))}
FRESH_PROTOCOLS = {'accelerate-checkpoint-fresh-process': (1, ('no',)),
                   'accelerate-fsdp2-full-fresh-process': (2, ('no', 'bf16'))}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_restoration(result, world_size):
    saved, restored = result.get('saved'), result.get('after_restore')
    if world_size == 2:
        # The maintained distributed node saves a complete training observation,
        # then observes only model/optimizer immediately after load. Loss and
        # gradients belong to the separate exact continuation checks.
        require(isinstance(saved, dict) and {'model', 'optimizer'} <= saved.keys()
                and isinstance(restored, dict) and set(restored) == {'model', 'optimizer'},
                'Missing FULL model/optimizer restoration snapshot')
        saved = {key: saved[key] for key in ('model', 'optimizer')}
    require(saved is not None and saved == restored, 'FULL model/optimizer restoration differs')


def validate_spec(record, source):
    require(record.get('record_kind') == 'protocol', 'Unknown protocol record kind')
    require(source.get('axes') == [] and record.get('registry_refs') == [],
            'Protocol rows require explicit empty axes and registry_refs')
    spec = record.get('protocol', {})
    require(isinstance(spec, dict), 'Protocol descriptor must be an object')
    if spec.get('id') in FRESH_PROTOCOLS:
        from _accelerate_fresh_scorecard import protocol_spec
        world, precisions = FRESH_PROTOCOLS[spec['id']]
        require(spec == protocol_spec(world, spec.get('precision')) and spec['precision'] in precisions,
                'Fresh-process semantic protocol differs')
        require(all(type(spec[key]) is int for key in ('world_size', 'pre_save_steps', 'resume_steps'))
                and all(type(item) is int for key in ('ranks', 'input_seeds') for item in spec[key]),
                'Fresh-process dimensions must use actual integers')
        require(record.get('case') == 'world%d-%s-fresh-process-checkpoint' % (world, spec['precision'])
                and record.get('repo') == 'accelerate' and record.get('device') in ('cpu', 'cuda')
                and (world == 1 or record['device'] == 'cuda'), 'Fresh-process device/world/case mismatch')
        return spec
    require(isinstance(spec, dict) and set(spec) == {'id', 'world_size', 'precision', 'ranks'},
            'Unknown/missing Accelerate protocol semantic dimension')
    require(spec.get('id') in PROTOCOLS, 'Unknown maintained protocol')
    world, precisions = PROTOCOLS[spec['id']]
    require(type(spec.get('world_size')) is int and spec['world_size'] == world
            and spec.get('precision') in precisions and spec.get('ranks') == list(range(world)),
            'Protocol world/precision/rank mismatch')
    require(record.get('case') == 'world%d-%s-checkpoint' % (world, spec['precision']), 'Protocol case label mismatch')
    require(record.get('repo') == 'accelerate', 'Protocol repo mismatch')
    require(record.get('device') in ('cpu', 'cuda') and (world == 1 or record['device'] == 'cuda'),
            'Unvalidated protocol device/world scope')
    return spec


def measurements(record, artifacts, source):
    import numpy as np

    spec = validate_spec(record, source)
    fresh = spec['id'] in FRESH_PROTOCOLS
    roles = {row['role']: row for row in artifacts}
    require(len(roles) == len(artifacts), 'Duplicate protocol artifact role')

    def load(role):
        require(role in roles, 'Missing protocol artifact: ' + role)
        row = roles[role]
        path = Path(row['path']).expanduser().resolve()
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'],
                'Protocol artifact SHA256 mismatch: ' + role)
        return path

    expected_sources = {row['path']: row['sha256'] for row in source.get('package_sources', [])}
    require(expected_sources and len(expected_sources) == len(source['package_sources']),
            'Missing full upstream package source hashes')
    platform = json.loads(load('platform').read_text())
    require(platform == record['platform'], 'Protocol platform artifact mismatch')
    require(platform['card_count'] == (0 if record['device'] == 'cpu' else spec['world_size']),
            'Protocol allocated card count differs from actual world')
    expected_precision = {'matmul': False, 'cudnn': False, 'cudnn_benchmark': False, 'matmul_precision': 'highest'}
    require(platform.get('precision') == expected_precision, 'Protocol requires the original full-FP32 precision switches')
    all_values = {}
    layouts = {}
    source_hashes = measurement_sources(record["jittor_commit"], fresh=True) if fresh else measurement_sources(record["jittor_commit"])
    installed_hashes = None
    conditions = None
    parameter_count = input_count = checked = 0
    max_abs = max_scaled = 0.
    passed = True
    steps = None
    for label, runtime in (('reference', 'torch'), ('candidate', 'jittor')):
        for scenario, seed in (('base', 0), ('perturb', 7)):
            for rank in spec['ranks']:
                role = '%s-%s-rank%d' % (label, scenario, rank)
                if fresh:
                    from _accelerate_fresh_scorecard import observation
                    value, arrays = observation(role, load, record, runtime, seed, rank)
                else:
                    value = json.loads(load(role + '-metadata').read_text())
                require(value.get('schema_version') == 1 and value.get('runtime') == runtime
                        and type(value.get('rank')) is int and value.get('rank') == rank
                        and type(value.get('world_size')) is int and value.get('world_size') == spec['world_size']
                        and value.get('mixed_precision') == spec['precision'], 'Actual protocol runtime/rank mismatch')
                expected_case = 'single_checkpoint' if spec['world_size'] == 1 else 'fsdp2_full'
                require(value.get('case') == expected_case and value.get('device') == record['device'],
                        'Actual protocol case/device mismatch')
                require(value.get('jittor_commit') == record['jittor_commit']
                        and value.get('source_descriptor') == source and value.get('upstream') == record['upstream'],
                        'Stale protocol framework/upstream sources')
                hashes = value.get('measurement_source_sha256')
                require(hashes == source_hashes, 'Measured protocol source hashes differ from committed maintained source')
                distribution = value.get('downstream_distribution', {})
                require(distribution.get('source_sha256') == expected_sources
                        and distribution.get('version') == record['upstream']['version'],
                        'Actual installed Accelerate source/version mismatch')
                installed = distribution.get('installed_files_sha256', {})
                require(isinstance(installed, dict) and installed, 'Missing actual full installed package hashes')
                installed_hashes = installed if installed_hashes is None else installed_hashes
                require(installed == installed_hashes, 'Actually installed package differs across runtimes/ranks')
                for path, digest in expected_sources.items():
                    require(installed.get('/'.join(Path(path).parts[2:])) == digest,
                            'Actual installed package omits/differs from pinned source: ' + path)
                interpreter = value.get('interpreter', {})
                require(interpreter.get('is_shim') is (runtime == 'jittor')
                        and interpreter.get('executable') and type(interpreter.get('pid')) is int,
                        'Missing actual independent interpreter identity')
                if runtime == 'torch':
                    binary = Path(interpreter.get('binary_extension') or '')
                    require(binary.suffix in ('.so', '.pyd', '.dylib') and binary.is_file()
                            and hashlib.sha256(binary.read_bytes()).hexdigest() == interpreter.get('binary_sha256'),
                            'Oracle binary identity/hash missing')
                    if rank == 0 and seed == 0:
                        require(platform['oracle'] == {**interpreter, 'torch_npu_version': 'not-applicable'},
                                'Platform oracle identity differs from measured process')
                else:
                    flags = value.get('native_runtime_flags', {})
                    require(flags.get('use_cuda') == int(record['device'] == 'cuda')
                            and value.get('fallback_policy') == 'error' and type(value.get('fallback_count')) is int and value.get('fallback_count') == 0,
                            'Protocol backend/zero-fallback guard missing')
                require(value.get('hf_hub_offline') is True and value.get('seed') == seed,
                        'Protocol inputs are not fixed offline seed0/seed7 observations')
                require(value.get('tf32') == expected_precision
                        and value.get('runtime_conditions', {}).get('precision') == expected_precision,
                        'Actual protocol precision mismatch')
                conditions = value['runtime_conditions'] if conditions is None else conditions
                require(value['runtime_conditions'] == conditions, 'Actual protocol thread/affinity conditions mismatch')
                measured_platform = value.get('platform', {})
                require(measured_platform.get('physical_devices') == platform.get('physical_devices')
                        and measured_platform.get('card_count') == platform['card_count'],
                        'Protocol physical device allocation mismatch')
                if record['device'] == 'cuda':
                    allocation = measured_platform.get('physical_devices', [])
                    physical = {item['index']: item['uuid'] for item in allocation}
                    expected_device = measured_platform['card_indices'][rank]
                    require(value.get('rank_device_uuid') == physical.get(expected_device, expected_device)
                            and type(value.get('logical_device_index')) is int,
                            'Actual CUDA UUID/rank allocation mismatch')
                require(value.get('backend') == 'none'  if spec['world_size'] == 1
                        else 'nccl' in str(value.get('backend')), 'Missing actual collective backend')
                if not fresh:
                    array_path = load(role + '-arrays')
                    require(value.get('arrays_sha256') == roles[role + '-arrays']['sha256']
                            and Path(value.get('arrays', '')).resolve() == array_path, 'Actual protocol NPZ binding mismatch')
                    with np.load(array_path, allow_pickle=False) as archive:
                        arrays = {key: archive[key].copy() for key in archive.files}
                manifest = value.get('array_manifest', {})
                require(arrays and set(arrays) == set(manifest), 'Missing/extra protocol tensor observations')
                for key, array in arrays.items():
                    require(array.dtype.kind in 'fibu' and np.isfinite(array).all(), 'Nonfinite/non-numeric protocol tensor: ' + key)
                    observed = manifest[key]
                    is_metadata = ('/hyperparameters/' in key or '/learning_rates/' in key)
                    require(observed.get('shape') == list(array.shape) and observed.get('dtype') == str(array.dtype)
                            and ((is_metadata and observed.get('device') == 'host-metadata')
                                 or observed.get('device') == record['device']
                                 or (record['device'] == 'cuda' and key.endswith('/step') and '/optimizer/' in key
                                     and array.shape == () and observed.get('device') == 'cpu'))
                            and observed.get('torch_dtype') == ('metadata.' + str(array.dtype) if is_metadata else 'torch.bfloat16' if spec['precision'] == 'bf16' and key.endswith('/output') else 'torch.float32'), 'Protocol tensor dtype/shape/device mismatch')
                    if observed['device'] == 'cuda':
                        require(observed.get('pointer_device') == value['logical_device_index'],
                                'Protocol tensor allocation differs from actual CUDA device')
                result = value.get('result', {})
                before, expected, actual = (result.get(name, []) for name in ('before_save', 'expected_resume', 'actual_resume'))
                require(len(before) == 3 and len(expected) == len(actual) == 2 and expected == actual,
                        'Own-runtime checkpoint continuation is incomplete/not exact')
                if not fresh:
                    validate_restoration(result, spec['world_size'])
                rng = result.get('expected_random', {})
                require(isinstance(rng, dict) and set(rng) == {'python', 'numpy', 'torch_cpu', 'torch_device' if spec['world_size'] == 1 else 'torch_cuda'} and rng == result.get('resumed_random'),
                        'Four-stream own-runtime RNG restoration differs')
                steps = len(before) + len(expected)
                names = sorted(key[len('initial/model/'):] for key in arrays if key.startswith('initial/model/'))
                require(names == ['bias', 'weight'], 'Maintained Linear parameter scope differs')
                rows, columns = (2, 3) if spec['world_size'] == 1 else (3, 4)
                require(arrays['initial/model/weight'].shape == (rows, columns)
                        and arrays['initial/model/bias'].shape == (rows,), 'Maintained model shape differs')
                for step in range(7):
                    prefix = 'step/%d/' % step
                    required = {prefix + name for name in ('input', 'target', 'output', 'loss', 'input_gradient')}
                    for name in names:
                        required.update({prefix + 'model/' + name, prefix + 'parameter_gradients/' + name})
                        required.update(prefix + 'optimizer/' + name + '/' + field for field in ('step', 'exp_avg', 'exp_avg_sq'))
                    for group in ('0', '1'):
                        required.update(prefix + 'hyperparameters/' + group + '/' + field
                                        for field in ('lr', 'betas/0', 'betas/1', 'eps', 'weight_decay', 'amsgrad', 'maximize'))
                    if spec['world_size'] == 1:
                        required.update(prefix + 'learning_rates/' + group for group in ('0', '1'))
                    require(required <= arrays.keys(), 'Missing complete step/input/parameter gradients or Adam moments')
                    for field, shape in (('input', (4, columns)), ('input_gradient', (4, columns)),
                                         ('target', (4, rows)), ('output', (4, rows)), ('loss', ())):
                        require(arrays[prefix + field].shape == shape, 'Maintained input/output/gradient shape differs')
                    for name, shape in (('weight', (rows, columns)), ('bias', (rows,))):
                        for field in ('model', 'parameter_gradients'):
                            require(arrays[prefix + field + '/' + name].shape == shape, 'Complete parameter/gradient shape differs')
                        for field in ('exp_avg', 'exp_avg_sq'):
                            require(arrays[prefix + 'optimizer/' + name + '/' + field].shape == shape,
                                    'Complete Adam moment shape differs')
                        require(arrays[prefix + 'optimizer/' + name + '/step'].shape == (), 'Optimizer step must be scalar')
                for actual_step, expected_step in ((5, 3), (6, 4)):
                    prefix = 'step/%d/' % actual_step
                    for key in arrays:
                        if key.startswith(prefix):
                            peer = 'step/%d/' % expected_step + key[len(prefix):]
                            require(peer in arrays and np.array_equal(arrays[key], arrays[peer]),
                                    'Measured own-runtime resumed tensors are not exact')
                require(any(not np.array_equal(arrays['initial/model/' + name], arrays['step/0/model/' + name]) for name in names),
                        'Protocol optimizer did not update any actual parameter')
                require(not np.array_equal(arrays['step/0/loss'], arrays['step/2/loss']),
                        'Protocol loss trajectory is constant')
                if spec['world_size'] == 2:
                    observation = result.get('distributed_observation', {})
                    require(observation.get('rank') == rank and observation.get('world_size') == 2
                            and observation.get('torch_compat_active') is (runtime == 'jittor')
                            and 'nccl' in str(observation.get('backend')),
                            'Missing actual distributed rank/backend observation')
                    local = observation.get('parameter_shapes', {})
                    global_shapes = observation.get('global_parameter_shapes', {})
                    require(set(local) == set(global_shapes) == {'weight', 'bias'}
                            and global_shapes == {'weight': [3, 4], 'bias': [3]},
                            'Missing actual FULL/global or owned-shard parameter shapes')
                    require(all(value == 'torch.float32' for value in observation.get('master_parameter_dtypes', {}).values())
                            and set(observation.get('master_parameter_dtypes', {})) == set(local),
                            'FSDP master parameters are not FP32')
                    require(sum(math.prod(shape) for shape in local.values()) < 15,
                            'FSDP rank owns a complete replicated parameter allocation')
                    for name, shape in local.items():
                        require(all(type(size) is int and size >= 0 for size in shape)
                                and str(observation.get('parameter_devices', {}).get(name, '')).startswith('cuda')
                                and (math.prod(shape) == 0 or observation.get('parameter_pointers', {}).get(name, 0) > 0),
                                'Invalid observed FSDP local CUDA parameter allocation')
                    layouts[(runtime, seed, rank)] = local
                if spec['precision'] == 'bf16':
                    observations = result.get('compute_observations', [])
                    require(len(observations) == 7 and all(set(row) == {'input', 'weight', 'bias', 'output'}
                            and all(item.get('dtype') == 'torch.bfloat16' and item.get('device') == 'cuda'
                                    and type(item.get('pointer_device')) is int and item['pointer_device'] >= 0
                                    for item in row.values()) for row in observations),
                            'BF16 protocol lacks actual inner compute dtype/allocation observations')
                all_values[(runtime, seed, rank)] = arrays
    if spec['world_size'] == 2:
        for runtime in ('torch', 'jittor'):
            for seed in (0, 7):
                for name, expected_numel in (('weight', 12), ('bias', 3)):
                    require(sum(math.prod(layouts[(runtime, seed, rank)][name]) for rank in spec['ranks']) == expected_numel,
                            'FSDP owned shards do not cover each complete global parameter exactly')
    for seed in (0, 7):
        for rank in spec['ranks']:
            reference, candidate = (all_values[(runtime, seed, rank)] for runtime in ('torch', 'jittor'))
            require(set(reference) == set(candidate), 'Cross-runtime protocol tensor names differ')
            for key, ref in reference.items():
                actual = candidate[key]
                require(ref.shape == actual.shape and ref.dtype == actual.dtype, 'Cross-runtime protocol tensor shape/dtype differs')
                delta = np.abs(actual.astype(np.float64) - ref.astype(np.float64))
                absolute = float(delta.max(initial=0))
                scaled = float((delta / (np.abs(ref.astype(np.float64)) + ATOL / RTOL)).max(initial=0))
                max_abs, max_scaled = max(max_abs, absolute), max(max_scaled, scaled)
                passed = passed and bool(np.all(delta <= ATOL + RTOL * np.abs(ref)))
                if key.startswith('initial/') or key.endswith(('/input', '/target')):
                    require(np.array_equal(actual, ref), 'Oracle/candidate initial weights or inputs differ')
                parameter_count += '/parameter_gradients/' in key
                input_count += key.endswith('/input_gradient')
                checked += 1
    for runtime in ('torch', 'jittor'):
        for rank in spec['ranks']:
            base, perturb = (all_values[(runtime, seed, rank)] for seed in (0, 7))
            require(not np.array_equal(base['step/0/input'], perturb['step/0/input'])
                    and not np.array_equal(base['step/0/output'], perturb['step/0/output'])
                    and not np.array_equal(base['step/0/input_gradient'], perturb['step/0/input_gradient']),
                    'Protocol input perturbation did not change real outputs/gradients')
            require(all(np.array_equal(base[key], perturb[key]) for key in base if key.startswith('initial/')),
                    'Perturbation changed initial weights')
    result = {'numerics': {'passed': passed, 'finite': True, 'checked_tensors': checked,
                        'missing_tensors': [], 'input_perturbation_verified': True,
                        'max_abs': max_abs, 'max_scaled_relative': max_scaled, 'atol': ATOL, 'rtol': RTOL,
                        'parameter_gradients': {'requested': parameter_count, 'checked': parameter_count},
                        'input_gradients': {'requested': input_count, 'checked': input_count}},
            'training': {'steps': steps, 'loss_trajectory_passed': passed, 'parameter_trajectory_passed': passed},
            'own_continuation': {'model_optimizer_exact': True, 'rng_exact': True, 'trajectory_exact': True}}
    if fresh:
        result['fresh_process'] = {'distinct_producer_consumer': True, 'producer_terminated': True,
                                   'checkpoint_files_immutable': True, 'full_rng_state_exact': True,
                                   'scheduler_exact': True}
    return result


def validate_measurements(record, artifacts, source):
    actual = measurements(record, artifacts, source)
    require(actual['numerics']['passed'], 'Independent cross-runtime protocol numerics failed')
    keys = ('numerics', 'training', 'own_continuation') + (('fresh_process',) if 'fresh_process' in actual else ())
    for key in keys:
        require(record.get(key) == actual[key], 'Declared/recomputed protocol ' + key + ' mismatch')
    comparison = next(row for row in artifacts if row['role'] == 'comparison')
    require(json.loads(Path(comparison['path']).read_text()) == actual, 'Protocol comparison artifact mismatch')
