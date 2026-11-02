"""Common runtime evidence validation, independent of a downstream library."""
from pathlib import Path
import hashlib
from _installed_source_proof import require


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_runtime_metadata(value, baseline, expected_platform=None, *, source_validator):
    source_validator(value, baseline)
    runtime, device = value.get('runtime'), value.get('device')
    require(runtime in ('torch', 'jittor') and device in ('cpu', 'cuda'), 'Wrong component runtime/device')
    identity = value.get('interpreter', {})
    require(identity.get('is_shim') is (runtime == 'jittor')
            and type(identity.get('pid')) is int and identity['pid'] > 0,
            'Missing independent actual component process identity')
    if runtime == 'torch':
        binary = identity.get('binary_extension')
        require(binary and Path(binary).is_file() and digest(binary) == identity.get('binary_sha256'),
                'Missing actual independent binary Torch extension')
    paths = value.get('runtime_environment', {})
    require(set(paths) == {'HOME', 'JITTOR_HOME', 'TMPDIR', 'XDG_CACHE_HOME'}
            and all(isinstance(path, str) and Path(path).is_absolute() for path in paths.values()),
            'Missing actual isolated component runtime paths')
    platform = value.get('platform', {})
    controls = value.get('device_controls', {})
    observations = value.get('tensor_observations', [])
    require(platform.get('precision') == value.get('tf32') and observations
            and all(row.get('device') == device for row in observations),
            'Missing actual component precision/tensor device observations')
    if runtime == 'jittor':
        require(value.get('fallback_policy') == 'error' and value.get('fallback_count') == 0
                and value.get('native_runtime_flags', {}).get('use_cuda') == int(device == 'cuda')
                and all(row.get('dispatch') == device and row.get('location') ==
                        ('device' if device == 'cuda' else 'cpu') for row in observations),
                'Missing native component residency/zero-fallback proof')
    if device == 'cpu':
        require(controls.get('cuda_visible_devices') == '' and controls.get('cuda_available') is False
                and platform.get('card_count') == 0 and platform.get('card_indices') == [],
                'CPU component did not explicitly hide CUDA')
    else:
        physical = platform.get('physical_devices', [])
        require(controls.get('cuda_available') is True and len(physical) == platform.get('card_count') == 1
                and controls.get('cuda_device', {}).get('uuid') == physical[0]['uuid']
                and all(row.get('pointer_device') == 0 for row in observations),
                'Component tensor pointers differ from the allocated physical CUDA device')
    if expected_platform is not None:
        if runtime == 'jittor':
            require(platform == {key: item for key, item in expected_platform.items() if key != 'oracle'},
                    'Record platform differs from actual candidate worker')
        else:
            for key in ('os', 'cpu', 'accelerator_model', 'card_count', 'card_indices', 'physical_devices',
                        'driver', 'cann', 'precision'):
                require(platform.get(key) == expected_platform.get(key), 'Oracle/candidate physical platform mismatch')
            oracle = expected_platform.get('oracle', {})
            require(oracle.get('is_shim') is False and oracle.get('torch_version') == identity['torch_version']
                    and oracle.get('binary_extension') == identity['binary_extension'],
                    'Record oracle differs from actual independent process')
