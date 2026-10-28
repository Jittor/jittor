"""Original parameter-free vLLM scope with source and physical-device proof."""

from collections import Counter
import json
import math
from pathlib import Path

from _installed_source_proof import installation_proofs, require, validate_installations
from _vllm_installation import UPSTREAM_COMMIT, WHEEL_SHA256, framework_sources, validate_adapter_installation
from component_measure import digest, package_files
from collect_component_scorecard import validate_initial_weights

CASE = 'vllm_decoder_primitives'
FIELDS = {'__output__': (320,), 'ingrad::gates': (5, 32),
          'ingrad::hidden_states': (5, 16), 'ingrad::query': (5, 16), 'ingrad::key': (5, 16)}


def validate_metadata(value, source, baseline, runtime, device, seed, proof, root):
    require(value.get('case') == CASE and value.get('device') == device and value.get('seed') == seed,
            'Changed vLLM case/device/seed')
    require(value.get('jittor_commit') == baseline and value.get('upstream') == source['upstream']
            and source['upstream']['commit'] == UPSTREAM_COMMIT, 'Changed vLLM measurement source')
    require(value.get('hf_hub_offline') is True, 'vLLM offline control missing')
    controls = value.get('device_controls', {})
    if device == 'cpu':
        require(controls.get('cuda_visible_devices') == '' and controls.get('cuda_available') is False,
                'vLLM CPU did not explicitly disable CUDA')
    else:
        actual = controls.get('cuda_device', {})
        require(controls.get('cuda_available') is True and controls.get('cuda_visible_devices')
                and actual.get('logical_index') == 0 and type(actual.get('physical_index')) is int
                and actual.get('uuid', '').startswith('GPU-'), 'Missing physical vLLM CUDA allocation')
    conditions = value.get('runtime_conditions', {})
    require(set(conditions) >= {'affinity', 'runtime_threads', 'thread_env', 'precision'}
            and conditions['affinity'] and conditions['runtime_threads'] == 1
            and all(conditions['thread_env'].get(name) == '1'
                    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
            'Missing original vLLM runtime conditions')
    seconds = value.get('seconds')
    require(isinstance(seconds, (float, int)) and math.isfinite(seconds) and seconds > 0,
            'Invalid actual vLLM execution timing')
    interpreter = value.get('interpreter', {})
    require(interpreter.get('is_shim') is (runtime == 'jittor') and type(interpreter.get('pid')) is int
            and interpreter['pid'] > 0 and interpreter.get('executable'), 'Missing vLLM interpreter identity')
    if runtime == 'torch':
        binary = interpreter.get('binary_extension')
        require(binary and Path(binary).is_file() and digest(binary) == interpreter.get('binary_sha256')
                and interpreter.get('torch_version', '').split('+')[0] == '2.13.0', 'Missing genuine vLLM Torch binary')
    else:
        require(value.get('fallback_count') == 0 and value.get('fallback_policy') == 'error', 'vLLM fallback observed')
        flags = value.get('native_runtime_flags', {})
        require(flags.get('use_cuda') == int(device == 'cuda'), 'Wrong native vLLM device')
        validate_adapter_installation(value.get('adapter_distribution'), root)
    validate_installations(value, proof)
    binding = value['downstream_distribution']
    require(binding.get('distribution_name') == 'vllm' and binding.get('distribution_version') == '0.29.0'
            and binding.get('wheel_sha256') == WHEEL_SHA256 and binding.get('metadata_sha256')
            and binding.get('record_sha256'), 'Missing actual vLLM wheel metadata proof')
    for name in ('metadata', 'record'):
        path = Path(binding.get(name, ''))
        require(path.is_file() and digest(path) == binding[name + '_sha256'], 'Changed actual vLLM ' + name)
    require(Path(binding['metadata']).name == 'METADATA' and Path(binding['record']).name == 'RECORD'
            and Path(binding['metadata']).parent == Path(binding['record']).parent
            and Path(binding['metadata']).parent.parent == Path(binding['package']).parent,
            'Actual vLLM metadata belongs to another package prefix')
    dependencies = value.get('dependencies', {})
    require(set(dependencies) == {'vllm', 'transformers'}, 'Incomplete actual vLLM dependency scope')
    for name, version in (('vllm', '0.29.0'), ('transformers', '5.13.0')):
        dependency = dependencies[name]
        origin = Path(dependency.get('origin', ''))
        require(origin.is_file() and origin.name == '__init__.py' and dependency.get('version') == version,
                'Wrong actual vLLM dependency identity: ' + name)
        require(dependency.get('installed_files_sha256') == package_files(origin.parent),
                'Changed actual vLLM dependency package bytes: ' + name)
        if name == 'vllm':
            require(origin.parent == Path(binding['package'])
                    and dependency['installed_files_sha256'] == binding['installed_files_sha256'],
                    'Actual vLLM dependency and installed source disagree')
    captures = value.get('measurement_source_sha256', {})
    require(set(captures) == set(framework_sources(root)),
            'Incomplete vLLM framework source capture')
    import hashlib
    import os

    for relative, wanted in captures.items():
        path = root / relative
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                'Invalid vLLM source path')
        actual = hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest() if path.is_symlink() else digest(path)
        require(actual == wanted, 'Changed measured vLLM source bytes: ' + relative)
    observations = value.get('tensor_observations', [])
    require(isinstance(observations, list) and len(observations) == len(FIELDS),
            'Missing physical vLLM tensor observations')
    require(Counter(tuple(row.get('shape', ())) for row in observations) == Counter(FIELDS.values()),
            'Wrong physical vLLM tensor shapes')
    for observation in observations:
        require(observation.get('device') == device and observation.get('dtype') in ('float32', 'torch.float32'),
                'Wrong actual vLLM tensor device/dtype')
        if device == 'cuda':
            require(type(observation.get('pointer_device')) is int and observation['pointer_device'] == 0,
                    'Missing actual vLLM CUDA pointer identity')
        if runtime == 'jittor':
            require(observation.get('location') == ('device' if device == 'cuda' else 'cpu')
                    and observation.get('dispatch') == device, 'Wrong physical vLLM tensor residency')


def compare_arrays(reference, candidate):
    import numpy as np

    rows = []
    with np.load(reference, allow_pickle=False) as expected, np.load(candidate, allow_pickle=False) as actual:
        require(set(expected.files) == set(actual.files) == set(FIELDS), 'Changed complete vLLM tensor scope')
        for name, shape in FIELDS.items():
            a, b = expected[name], actual[name]
            require(a.shape == b.shape == shape and a.dtype == b.dtype == np.dtype('float32')
                    and np.isfinite(a).all() and np.isfinite(b).all(), 'Invalid vLLM tensor: ' + name)
            error = float(np.abs(a.astype('float64') - b.astype('float64')).max())
            scale = max(float(np.abs(a.astype('float64')).max()), 1e-8)
            limit = .002 if name == '__output__' else .01
            require(error / scale <= limit, 'Original vLLM numerical budget failed: ' + name)
            rows.append({'tensor': name, 'max_abs': error, 'reference_scale': scale,
                         'normalized_error': error / scale, 'limit': limit, 'passed': True})
    return rows


def compare_pair(reference, candidate, source, baseline, device, seed, wheel, root):
    import numpy as np

    root = Path(root)
    paths = [Path(reference), Path(candidate)]
    values = [json.loads(path.with_suffix('.json').read_text()) for path in paths]
    require(digest(wheel) == WHEEL_SHA256, 'Changed official vLLM wheel')
    proof = installation_proofs({'repo': 'vllm'}, {'vllm': source},
                               [{'role': 'upstream-wheel', 'path': str(wheel), 'sha256': WHEEL_SHA256}])
    for runtime, value in zip(('torch', 'jittor'), values):
        validate_metadata(value, source, baseline, runtime, device, seed, proof, root)
    require(values[0]['interpreter']['pid'] != values[1]['interpreter']['pid'],
            'vLLM runtimes did not use distinct worker processes')
    for name in ('weights_sha256', 'runtime_conditions', 'tf32', 'measurement_source_sha256', 'device_controls'):
        require(values[0].get(name) and values[0].get(name) == values[1].get(name), 'Changed vLLM pair ' + name)
    normalized = [{name: {key: data for key, data in value.items() if key != 'origin'}
                   for name, value in report['dependencies'].items()} for report in values]
    require(normalized[0] == normalized[1], 'Changed actual vLLM dependency bytes or identity fields')
    require(values[0]['tf32'] == {'matmul': False, 'cudnn': False, 'cudnn_benchmark': False,
                                 'matmul_precision': 'highest'}, 'Changed original vLLM TF32 policy')
    require(values[0]['runtime_conditions'].get('precision') == values[0]['tf32'], 'Wrong actual vLLM precision')
    validate_initial_weights(values, values[0]['weights_sha256'])
    with np.load(values[0]['weights'], allow_pickle=False) as weights:
        require(weights.files == ['rotary.cos_sin_cache'], 'Changed parameter-free vLLM state scope')
        cache = weights['rotary.cos_sin_cache']
        require(cache.shape == (32, 8) and cache.dtype == np.dtype('float32') and np.isfinite(cache).all(),
                'Invalid original vLLM rotary state')
    rows = compare_arrays(*paths)
    return {'case': CASE, 'status': 'passed', 'seed': seed, 'device': device,
            'reference_result': values[0], 'candidate_result': values[1], 'rows': rows,
            'max_abs': max(row['max_abs'] for row in rows),
            'max_scaled_relative': max(row['normalized_error'] for row in rows), 'missing': [],
            'parameter_gradients': {'requested': 0, 'checked': 0}, 'input_gradients': {'requested': 4, 'checked': 4}}


def validate_perturbation(original, perturbed):
    import numpy as np

    with np.load(original, allow_pickle=False) as a, np.load(perturbed, allow_pickle=False) as b:
        require(set(a.files) == set(b.files) == set(FIELDS), 'Changed complete vLLM perturbation scope')
        require(not np.array_equal(a['__output__'], b['__output__']), 'Unchanged vLLM input perturbation output')
        changed = sorted(name for name in FIELDS if name.startswith('ingrad::') and not np.array_equal(a[name], b[name]))
        require(changed, 'Unchanged vLLM input perturbation gradients')
    return changed
