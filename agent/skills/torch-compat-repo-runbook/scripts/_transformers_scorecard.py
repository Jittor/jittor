"""Source-bound replay of the original nine Transformers cases and 32K Longformer.

Only validation/provenance lives here. Models, AD and SGD remain in the public
case builders, ecosystem runner and training_probe.
"""
import hashlib
import json
from pathlib import Path
import zipfile

from _peft_scorecard import PINS, digest, git, require
from _peft_scorecard import installed_payload as installed_payload
from _peft_scorecard import source_files as peft_source_files
from _installed_source_proof import installation_proofs, validate_installations

# (parameter count, trainable count, base arrays, three-step arrays, model family)
SPECS = {
    'transformers_bert': (39, 39, 40, 240, 'bert'),
    'transformers_clip_multimodal': (78, 78, 80, 477, 'clip'),
    'transformers_gpt2': (28, 28, 29, 174, 'gpt2'),
    'transformers_llama': (21, 21, 22, 132, 'llama'),
    'transformers_llama_rope_2048': (12, 12, 13, 78, 'llama'),
    'transformers_mixtral_moe': (41, 41, 42, 252, 'mixtral'),
    'transformers_t5': (47, 47, 48, 288, 't5'),
    'transformers_vit': (40, 40, 42, 249, 'vit'),
    'transformers_whisper': (89, 88, 91, 540, 'whisper'),
    'transformers_longformer_32768': (29, 29, 31, 183, 'longformer'),
}
OUTPUTS = {'transformers_bert': [1152],
 'transformers_clip_multimodal': [2, 2],
 'transformers_gpt2': [2, 8, 128],
 'transformers_llama': [2, 8, 128],
 'transformers_llama_rope_2048': [1, 2048, 64],
 'transformers_longformer_32768': [1048608],
 'transformers_mixtral_moe': [2, 32, 64],
 'transformers_t5': [2, 8, 64],
 'transformers_vit': [2304],
 'transformers_whisper': [2, 8, 64]}
CAUSAL = ('gpt2', 'llama', 'mixtral')
PHASES = (('base', 0), ('perturb7', 7), ('perturb19', 19), ('train', 0))
INPUTS = {
    'transformers_bert': {'input_ids': ['int64', [2, 8], 128]},
    'transformers_clip_multimodal': {'input_ids': ['int64', [2, 8], 64],
                                    'pixel_values': ['float32', [2, 3, 16, 16], None]},
    'transformers_gpt2': {'input_ids': ['int64', [2, 8], 128]},
    'transformers_llama': {'input_ids': ['int64', [2, 8], 128]},
    'transformers_llama_rope_2048': {'input_ids': ['int64', [1, 2048], 64]},
    'transformers_mixtral_moe': {'input_ids': ['int64', [2, 32], 64]},
    'transformers_t5': {'input_ids': ['int64', [2, 8], 128],
                         'decoder_input_ids': ['int64', [2, 8], 128]},
    'transformers_vit': {'pixel_values': ['float32', [2, 3, 32, 32], None]},
    'transformers_whisper': {'input_features': ['float32', [2, 16, 64], None],
                              'decoder_input_ids': ['int64', [2, 8], 128]},
    'transformers_longformer_32768': {'input_ids': ['int64', [1, 32768], 64],
                                      'embedding_noise': ['float32', [1, 32768, 32], None]},
}


def source_descriptor(checkout, wheel):
    from registry_inventory import inventory
    checkout, wheel = Path(checkout).resolve(), Path(wheel).resolve()
    version, sha, remote, wheel_hash = PINS['transformers']
    source = inventory('transformers', checkout)
    require(not git(checkout, 'status', '--porcelain'), 'Dirty official Transformers source')
    require(source['upstream'] == {'commit': sha, 'version': version, 'remote': git(checkout, 'remote', 'get-url', 'origin')}
            and source['upstream']['remote'] in ('git@github.com:' + remote + '.git',
                                                'https://github.com/' + remote + '.git'), 'Wrong official Transformers source')
    require(digest(wheel) == wheel_hash, 'Wrong official Transformers wheel')
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate official wheel member')
        payload = {n[len('transformers/'):]: hashlib.sha256(archive.read(n)).hexdigest()
                   for n in names if n.startswith('transformers/') and not n.endswith('/')}
    included, excluded = [], []
    for name in git(checkout, 'ls-files', 'src/transformers').splitlines():
        relative = name[len('src/transformers/'):]
        row = {'path': name, 'sha256': digest(checkout / name)}
        if relative not in payload:
            require(name in ('src/transformers/models/gpt2/CONVERSION.md',
                             'src/transformers/models/t5/download_from_gcp.sh'), 'Wheel omits runtime source')
            excluded.append(dict(row, reason='Official wheel omits this non-runtime file'))
        else:
            require(payload[relative] == row['sha256'], 'Official wheel/source mismatch')
            included.append(row)
    require(payload == {r['path'][len('src/transformers/'):]: r['sha256'] for r in included},
            'Unbound complete official package payload')
    source.update(package_sources=included, excluded_package_sources=excluded,
                  wheel={'path': str(wheel), 'sha256': wheel_hash, 'package_files_sha256': payload})
    return {'transformers': source}


def source_files(root):
    root = Path(root)
    values = peft_source_files(root)
    for name in ('tools/run_transformers_gate.py',
                 'agent/skills/torch-compat-repo-runbook/scripts/_transformers_scorecard.py',
                 'agent/skills/torch-compat-repo-runbook/scripts/collect_transformers_scorecards.py',
                 'agent/skills/torch-compat-repo-runbook/scripts/_accelerate_protocol.py',
                 'agent/skills/torch-compat-repo-runbook/scripts/_installed_source_proof.py'):
        values[name] = digest(root / name)
    return dict(sorted(values.items()))


def capture_scope(model, inputs):
    configs = {}
    for name, module in model.named_modules():
        config = getattr(module, 'config', None)
        if config is not None and hasattr(config, 'to_dict'):
            configs[name] = json.loads(json.dumps(config.to_dict(), default=str))
    return {'parameters': sorted(n for n, _ in model.named_parameters()),
            'buffers': sorted(n for n, _ in model.named_buffers()),
            'trainable_parameters': sorted(n for n, p in model.named_parameters() if p.requires_grad),
            'frozen_parameters': sorted(n for n, p in model.named_parameters() if not p.requires_grad),
            'input_spec': json.loads(json.dumps(inputs)), 'configs': configs}


def validate_binding(value, sources, baseline, runtime, device, case):
    require(value.get('case') == case and case in SPECS and value.get('jittor_commit') == baseline,
            'Wrong original case/source')
    identity = value.get('interpreter', {})
    require(value.get('runtime') == runtime and value.get('device') == device
            and identity.get('is_shim') is (runtime == 'jittor') and type(identity.get('pid')) is int,
            'Wrong actual runtime/device identity')
    if runtime == 'torch':
        binary = identity.get('binary_extension')
        require(isinstance(binary, str) and Path(binary).is_file()
                and digest(binary) == identity.get('binary_sha256'), 'Missing genuine Torch binary')
    source = sources['transformers']
    proof = installation_proofs({'repo': 'transformers'}, sources,
                                [{'role': 'transformers-official-wheel', **source['wheel']}])
    validate_installations(value, proof)
    actual = value['downstream_distributions']['transformers']
    with zipfile.ZipFile(source['wheel']['path']) as wheel:
        prefix = next(n.rsplit('/', 1)[0] for n in wheel.namelist() if n.endswith('.dist-info/METADATA'))
        require(all(actual.get(key + '_sha256') == hashlib.sha256(wheel.read(prefix + '/' + member)).hexdigest()
                    for key, member in (('metadata', 'METADATA'), ('record', 'RECORD'))),
                'Missing official METADATA/RECORD proof')
    dependency = value.get('dependencies', {}).get('transformers', {})
    require(dependency.get('version') == actual['version']
            and dependency.get('installed_files_sha256') == actual['installed_files_sha256']
            and Path(dependency.get('origin', '')).parent == Path(actual['package']), 'Runtime dependency mismatch')
    require(value.get('hf_hub_offline') is True and value.get('measurement_source_sha256'), 'Missing offline/source proof')
    if runtime == 'jittor':
        require(value.get('fallback_policy') == 'error' and value.get('fallback_count') == 0, 'Missing real zero-fallback')
    paths = value.get('actual_paths', {})
    require(paths and paths == value.get('requested_paths') and all(
        isinstance(paths.get(k), str) and Path(paths[k]).is_absolute() for k in
        ('HOME', 'REAL_HOME', 'JITTOR_HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'JITTOR_TORCH_CACHE_ROOT')),
        'Wrong private runtime path routing')
    scope = value['model_scope']
    require(scope['input_spec'] == INPUTS[case] and len(scope['parameters']) == SPECS[case][0]
            and len(scope['trainable_parameters']) == SPECS[case][1] and scope['configs'], 'Reduced original model/input scope')
    observations = value.get('device_observations', [])
    require(observations and all(r['device'] == device for r in observations), 'Missing real tensor residency')
    labels = {r['label'] for r in observations}
    require({'output'} | {'input/' + n for n in INPUTS[case]} | {'parameter/' + n for n in scope['parameters']} <= labels,
            'Missing input/output/parameter residency')
    expected = scope['trainable_parameters'] if value.get('mode') == 'train' else scope['parameters']
    expected_inputs = {n for n, spec in INPUTS[case].items() if spec[0] != 'int64'}
    require({r['label'][9:] for r in observations if r['label'].startswith('gradient/')} == set(expected)
            and {r['label'][15:] for r in observations if r['label'].startswith('input_gradient/')} == expected_inputs,
            'Missing requested gradient residency')
    if device == 'cuda':
        allocation = value.get('cuda_device', {})
        require(allocation.get('logical_index') == 0 and type(allocation.get('physical_index')) is int
                and str(allocation.get('uuid', '')).startswith('GPU-'), 'Missing CUDA driver allocation identity')
        require(str(allocation['physical_index']) in value['platform']['card_indices']
                or allocation['uuid'] in value['platform']['card_indices'], 'Wrong physical CUDA device')
        mapped = value['platform']['cuda'].get('mapped_runtime_api', {})
        require(mapped and set(mapped) <= {r['path'] for r in value['mapped_libraries']}
                and all(type(v) is int and v > 0 for v in mapped.values()), 'Missing actual CUDA runtime version')
        require(all(r.get('pointer_device') == 0 for r in observations), 'Not a real CUDA allocation')


def compare_arrays(reference, candidate, expected, case, phase, device):
    """Keep the original phase-global TF9 and per-block Longformer budgets."""
    import numpy as np
    with np.load(reference) as ref, np.load(candidate) as cand:
        require(set(ref.files) == set(cand.files) == set(expected), 'Missing/unexpected original numerical field')
        longformer = case == 'transformers_longformer_32768'
        atol, rtol = ((.005, .02) if device == 'cuda' and phase != 'base' else (.002, .01))
        global_scale = max([1e-12, *[float(np.abs(ref[k].astype('float64')).max()) for k in ref.files if ref[k].size]])
        rows = []
        for name in sorted(ref.files):
            a, b = ref[name], cand[name]
            require(a.shape == b.shape and a.dtype == b.dtype and np.isfinite(a).all() and np.isfinite(b).all(),
                    'Invalid exact-shaped finite array: ' + name)
            error = float(np.abs(a.astype('float64') - b.astype('float64')).max()) if a.size else 0.
            scale = global_scale
            if longformer:
                prefix = name.split('/', 2)[:2] if phase == 'train' else []
                block = '/'.join(prefix) + '/' if prefix else ''
                gradient = name.startswith(block + ('grad::')) or name.startswith(block + 'ingrad::')
                keys = [k for k in ref.files if k.startswith(block + 'grad::') or k.startswith(block + 'ingrad::')]
                floor = .001 * max([0., *[float(np.abs(ref[k].astype('float64')).max()) for k in keys]]) if gradient else .001 * float(np.abs(a.astype('float64')).max())
                scale = max(float(np.abs(a.astype('float64')).max()), floor, 1e-6)
                tolerance = (.02 if device == 'cuda' else .01) if gradient else (.005 if device == 'cuda' else .002)
                require(error / scale < tolerance, 'Original Longformer normalized budget failed: ' + name)
            else:
                require(error <= atol or error / scale <= rtol, 'Original Transformers stage budget failed: ' + name)
                tolerance = rtol
            rows.append({'name': name, 'max_abs': error, 'max_scaled_relative': error / scale,
                         'normalization': 'longformer-field-gradient-block' if longformer else 'phase-global',
                         'tolerance': tolerance})
    return rows
