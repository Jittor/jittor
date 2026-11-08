"""Validate the bounded real DeepSpeed Engine protocol, with no registry claim.

Only measurements of ZeRO1/2 AdamW BF16 world-two training and strict fresh
Engine continuation are admitted. MPI CPU evidence is not native Gloo or CPU
jtrun support. Numerical budgets remain those of the maintained Engine gate.
"""
import ast
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import subprocess
from types import SimpleNamespace
import zipfile

from registry_inventory import ROOT

UPSTREAM_COMMIT = 'f5af15c2b4da10411e340850c00ba0cb5b14fd87'
MEASUREMENT_SOURCES = tuple('compat/tests/torch/' + name for name in (
    '_ecosystem_deepspeed_runner.py', '_ecosystem_deepspeed_multirank.py',
    '_ecosystem_deepspeed_cases.py', '_ecosystem_deepspeed_comparison.py'))
PARAMETERS = {'up.weight': (16, 8), 'up.bias': (16,), 'down.weight': (8, 16), 'down.bias': (8,),
              'norm.weight': (8,), 'norm.bias': (8,), 'head.weight': (4, 8), 'head.bias': (4,)}
GENERATED = 'deepspeed/git_version_info_installed.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(checkout, *arguments):
    try:
        return subprocess.check_output(['git', '-C', str(checkout), *arguments], stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('Cannot resolve committed protocol sources') from error


@lru_cache(maxsize=4)
def _tree(checkout, revision):
    rows = git(checkout, 'ls-tree', '-r', '-z', revision).split(b'\0')
    tree = {}
    for row in rows:
        if not row:
            continue
        header, path = row.split(b'\t', 1)
        mode, kind, object_id = header.decode().split()
        require(kind == 'blob' and mode in ('100644', '100755', '120000'),
                'Unresolved upstream gitlink/object')
        tree[path.decode()] = (mode, object_id)
    identifiers = sorted({value[1] for value in tree.values()})
    content = subprocess.run(['git', '-C', str(checkout), 'cat-file', '--batch'],
                             input=('\n'.join(identifiers) + '\n').encode(), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, check=True).stdout
    blobs, offset = {}, 0
    for identifier in identifiers:
        end = content.index(b'\n', offset)
        observed, kind, size = content[offset:end].decode().split()
        require(observed == identifier and kind == 'blob', 'Git object identity mismatch')
        size = int(size)
        blobs[identifier] = content[end + 1:end + 1 + size]
        offset = end + 2 + size
    require(offset == len(content), 'Trailing Git source content')
    return tree, blobs


def _package_closure(tree, blobs):
    """Expand official directory symlinks without trusting installed packages."""
    result = {}
    def expand(source, destination, ancestors):
        require(source not in ancestors, 'Cyclic upstream package symlink')
        if source in tree:
            mode, identifier = tree[source]
            value = blobs[identifier]
            if mode == '120000':
                target = value.decode()
                require(not target.startswith('/'), 'Absolute upstream package symlink')
                target = posixpath.normpath(posixpath.join(posixpath.dirname(source), target))
                require(target not in ('.', '..') and not target.startswith('../'),
                        'Upstream package symlink escapes source checkout')
                expand(target, destination, ancestors | {source})
            else:
                require(destination not in result, 'Duplicate exported upstream source')
                result[destination] = {'path': destination, 'upstream_path': source,
                    'sha256': hashlib.sha256(value).hexdigest()}
            return
        prefix = source + '/'
        children = sorted({name[len(prefix):].split('/')[0] for name in tree if name.startswith(prefix)})
        require(children, 'Unresolved upstream package symlink target')
        for child in children:
            expand(prefix + child, destination + '/' + child, ancestors)
    expand('deepspeed', 'deepspeed', set())
    return result


def _exported(path, closure):
    # Official setup.py find_packages + MANIFEST.in recursive package data.
    if Path(path).suffix in ('.cpp', '.h', '.hpp', '.cu', '.hip', '.cuh', '.cc', '.json'):
        return True
    if not path.endswith('.py'):
        return False
    parent = PurePosixPath(path).parent
    while str(parent) != '.':
        if str(parent / '__init__.py') not in closure:
            return False
        parent = parent.parent
    return True


def _generated_fields(value, version, commit):
    try:
        tree = ast.parse(value)
        require(all(isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) for node in tree.body),
                'Executable/unresolved generated DeepSpeed metadata')
        fields = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body}
    except (SyntaxError, UnicodeError, TypeError, ValueError) as error:
        raise ValueError('Invalid generated DeepSpeed metadata') from error
    require(len(fields) == len(tree.body) == 6 and set(fields) == {
        'version', 'git_hash', 'git_branch', 'installed_ops', 'accelerator_name', 'torch_info'},
        'Missing/extra generated DeepSpeed field')
    require(fields['version'] == version + '+' + commit[:7]
            and fields['git_hash'] == commit[:7] and fields['git_branch'] == 'HEAD',
            'Generated DeepSpeed SCM identity mismatch')
    require(fields['accelerator_name'] in ('cpu', 'cuda') and isinstance(fields['installed_ops'], dict)
            and fields['installed_ops'] and all(value is False for value in fields['installed_ops'].values()),
            'Protocol requires the actual uncompiled upstream wheel')
    require(isinstance(fields['torch_info'], dict) and fields['torch_info'].get('version')
            and fields['torch_info'].get('cuda_version'), 'Missing actual wheel build Torch identity')
    return fields


def source_descriptor(checkout, wheel):
    """Derive the full official wheel source closure, including linked C++ files."""
    checkout, wheel = Path(checkout).resolve(), Path(wheel).resolve()
    require(not git(checkout, 'status', '--porcelain').strip(), 'DeepSpeed upstream checkout must be clean')
    remote = git(checkout, 'remote', 'get-url', 'origin').decode().strip()
    require(remote in ('https://github.com/deepspeedai/DeepSpeed.git',
                       'https://github.com/deepspeedai/DeepSpeed',
                       'git@github.com:deepspeedai/DeepSpeed.git'), 'Expected official DeepSpeed upstream')
    commit = git(checkout, 'rev-parse', 'HEAD').decode().strip()
    tree, blobs = _tree(str(checkout), commit)
    closure = _package_closure(tree, blobs)
    files = sorted((item for name, item in closure.items() if _exported(name, closure)), key=lambda row: row['path'])
    excluded = sorted((item for name, item in closure.items() if not _exported(name, closure)), key=lambda row: row['path'])
    version = blobs[tree['version.txt'][1]].decode().strip()
    require(files and {'setup.py', 'MANIFEST.in', 'op_builder/all_ops.py'} <= tree.keys(),
            'Missing official packaging source')
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate upstream wheel members')
        payload = {name for name in names if name.startswith('deepspeed/') and not name.endswith('/')}
        require(payload == {row['path'] for row in files} | {GENERATED},
                'Official wheel package closure differs from setup.py/MANIFEST.in sources')
        for row in files:
            require(hashlib.sha256(archive.read(row['path'])).hexdigest() == row['sha256'],
                    'Upstream wheel differs from Git source: ' + row['path'])
        generated = archive.read(GENERATED)
        fields = _generated_fields(generated, version, commit)
    packaging = [{'path': path, 'sha256': hashlib.sha256(blobs[tree[path][1]]).hexdigest()}
                 for path in ('setup.py', 'MANIFEST.in', 'version.txt', 'op_builder/all_ops.py')]
    return {'repo': 'deepspeed', 'upstream': {'remote': remote, 'commit': commit,
            'version': 'source@' + commit}, 'axes': [], 'checkout': str(checkout),
            'package_sources': files, 'packaging_sources': packaging,
            'excluded_package_sources': excluded,
            'exclusion_rule': 'Official find_packages and recursive package-data suffixes; excludes docs/non-package tests/MPS metal.',
            'generated_source': {'path': GENERATED, 'sha256': hashlib.sha256(generated).hexdigest(),
                                 'fields': fields}}


def protocol(stage):
    return {'id': 'deepspeed-zero-training', 'world_size': 2, 'precision': 'bf16', 'ranks': [0, 1],
            'zero_stage': stage, 'optimizer': 'AdamW', 'gradient_accumulation_steps': 2,
            'updates': 3, 'microsteps': 6, 'checkpoint_after_microstep': 4,
            'resume_microsteps': [4, 5], 'seed': 0, 'input_perturbations': [0., .125],
            'model': {'id': 'residual-linear-gelu-layernorm-v1', 'micro_batch': 2,
                      'input_features': 8, 'hidden_features': 16, 'output_features': 4}}


def validate_spec(record, source):
    require(isinstance(record, dict) and isinstance(source, dict), 'Protocol record/source must be objects')
    require(record.get('record_kind') == 'protocol' and record.get('repo') == source.get('repo') == 'deepspeed'
            and record.get('registry_refs') == [] and source.get('axes') == [],
            'DeepSpeed protocol cannot claim registry breadth')
    spec = record.get('protocol')
    require(isinstance(spec, dict) and type(spec.get('zero_stage')) is int and spec['zero_stage'] in (1, 2)
            and spec == protocol(spec['zero_stage']), 'Unknown/missing DeepSpeed protocol semantic dimension')
    require(all(type(spec.get(key)) is int for key in ('world_size', 'gradient_accumulation_steps',
                'updates', 'microsteps', 'checkpoint_after_microstep', 'seed'))
            and all(type(value) is int for key in ('ranks', 'resume_microsteps') for value in spec[key])
            and all(type(value) is float for value in spec['input_perturbations']),
            'Protocol numeric semantic axes must not be booleans/coerced labels')
    require(record.get('device') in ('cpu', 'cuda') and record.get('case') ==
            'zero%d-world2-bf16-adamw' % spec['zero_stage'], 'Unverified DeepSpeed case/device scope')
    require(source.get('upstream', {}).get('commit') == UPSTREAM_COMMIT,
            'Unverified DeepSpeed upstream source version')
    return spec


@lru_cache(maxsize=128)
def _committed_source(baseline, path):
    return git(ROOT, 'show', baseline + ':' + path)


def _measured_function(baseline, path, name, namespace):
    """Reuse the actual committed comparator; do not invent a looser budget."""
    value = _committed_source(baseline, path)
    nodes = [node for node in ast.parse(value).body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(nodes) == 1, 'Missing committed protocol validator: ' + name)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)
    return namespace[name]


def expected_shapes(stage):
    result = {'initial/parameter/' + name: shape for name, shape in PARAMETERS.items()}
    for prefix in ('saved', 'final', 'restored', 'resumed'):
        for name in ('exp_avg', 'exp_avg_sq', 'step'):
            result[prefix + '_optimizer/0/' + name] = () if name == 'step' else (166,)
        result[prefix + '_master_partition/0/0'] = (166,)
    for label, index in [('step%d' % i, i) for i in range(6)] + [('resumed4', 4), ('resumed5', 5)]:
        for name, shape in {'output': (2, 4), 'loss': (), 'input/inputs': (2, 8), 'input/target': (2, 4),
                            'input_grad/inputs': (2, 8), 'input_grad/target': (2, 4)}.items():
            result[label + '/' + name] = shape
        for name, shape in PARAMETERS.items():
            for category in ('parameter', 'incoming_parameter_grad'):
                result[label + '/' + category + '/' + name] = shape
            if stage == 1 or index % 2 == 1:
                result[label + '/parameter_grad/' + name] = shape
    return result


def _bf16(value):
    import numpy as np
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)


def _assert_arrays(arrays, metadata, stage, offset, rank):
    import numpy as np
    shapes = expected_shapes(stage)
    require(set(arrays) == set(shapes), 'Missing/extra DeepSpeed observation or gradient')
    for key, shape in shapes.items():
        require(arrays[key].shape == shape and arrays[key].dtype == np.dtype('float32')
                and np.isfinite(arrays[key]).all(), 'Nonfinite/invalid DeepSpeed dtype or shape: ' + key)
    for index, (name, shape) in enumerate(PARAMETERS.items()):
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * np.float32(.001)
        expected = _bf16(values + np.float32(.01 * (index + 1)))
        require(np.array_equal(arrays['initial/parameter/' + name], expected),
                'Actually trained initial BF16 weights differ from fixed fixture')
    for label, index in [('step%d' % i, i) for i in range(6)] + [('resumed4', 4), ('resumed5', 5)]:
        perturbation = offset + .125 * rank
        values = np.arange(16, dtype=np.float32).reshape(2, 8) * np.float32(.03)
        inputs = _bf16(values + np.float32(.01 * index) + np.float32(perturbation))
        target = _bf16(np.arange(8, dtype=np.float32).reshape(2, 4) * np.float32(-.007)
                       - np.float32(perturbation))
        require(np.array_equal(arrays[label + '/input/inputs'], inputs)
                and np.array_equal(arrays[label + '/input/target'], target), 'Actual fixed input/target formula differs')
    expected_gradients = {key: ('float32' if stage == 2 and '/parameter_grad/' in key else 'bfloat16')
                          for key in shapes if '_grad/' in key}
    require(metadata.get('gradient_dtypes') == expected_gradients,
            'Actual stage1 BF16 shortcut/stage2 FP32 full-gradient dtype mismatch')
    namespace = {'np': np, 'hashlib': hashlib, 'json': json}
    weights_digest = _measured_function(metadata['jittor_commit'], MEASUREMENT_SOURCES[0], '_weights_digest', namespace)
    require(weights_digest({name: arrays['initial/parameter/' + name] for name in PARAMETERS})
            == metadata.get('initial_weights_sha256'), 'Actual initial training weight digest differs')
    strict = _measured_function(metadata['jittor_commit'], MEASUREMENT_SOURCES[0], '_assert_strict_resume', namespace)
    try:
        checked, changed = strict(arrays, metadata['gradient_dtypes'])
    except (AssertionError, KeyError) as error:
        raise ValueError('DeepSpeed strict fresh Engine continuation failed') from error
    require(checked == metadata.get('strict_resume_fields') == (68 if stage == 1 else 60)
            and changed == metadata.get('updated_master_elements') == 166, 'Incomplete/no-op DeepSpeed continuation')
    for prefix, step in (('saved', 2), ('restored', 2), ('final', 3), ('resumed', 3)):
        require(float(arrays[prefix + '_optimizer/0/step']) == step, 'Actual AdamW step count differs')
    require(any(not np.array_equal(arrays['initial/parameter/' + name], arrays['step5/parameter/' + name])
                for name in PARAMETERS), 'No actual model parameter update')


def verify_perturbation(before, after, bm, pm):
    import numpy as np
    require(bm['initial_weights_sha256'] == pm['initial_weights_sha256']
            and all(np.array_equal(before['initial/parameter/' + name], after['initial/parameter/' + name]) for name in PARAMETERS),
            'Input perturbation changed initial weights')
    require(any(not np.array_equal(before['step%d/output' % index], after['step%d/output' % index])
                for index in range(6))
            and all(not np.array_equal(before['step%d/loss' % index], after['step%d/loss' % index])
                    for index in range(6))
            and any(not np.array_equal(before[key], after[key]) for key in before if '_grad/' in key),
            'Perturbation did not affect actual outputs/gradients')


def measurements(record, artifacts, source):
    import numpy as np
    spec = validate_spec(record, source)
    roles = {item['role']: item for item in artifacts}
    require(len(roles) == len(artifacts), 'Duplicate DeepSpeed artifact role')
    def load(role):
        require(role in roles, 'Missing DeepSpeed artifact: ' + role)
        item = roles[role]
        path = Path(item['path']).expanduser().resolve()
        require(path.is_file() and digest(path) == item['sha256'], 'DeepSpeed artifact SHA256 mismatch: ' + role)
        return path
    wheel = load('upstream-wheel')
    require(source_descriptor(source['checkout'], wheel) == source, 'Regenerated official DeepSpeed source closure differs')
    require(json.loads(load('source-inventory').read_text()) == source, 'Bound upstream source descriptor differs')
    with zipfile.ZipFile(wheel) as first, zipfile.ZipFile(load('upstream-rebuilt-wheel')) as rebuilt:
        names = {row['path'] for row in source['package_sources']} | {GENERATED}
        require(names == {name for name in rebuilt.namelist() if name.startswith('deepspeed/') and not name.endswith('/')}
                and all(first.read(name) == rebuilt.read(name) for name in names),
                'Independent official wheel rebuild package bytes differ')
    platform = json.loads(load('platform').read_text())
    require(platform == record.get('platform'), 'Actual platform artifact differs')
    require(platform['card_count'] == (2 if record['device'] == 'cuda' else 0), 'Actual card count differs')
    expected_package = {row['path'][len('deepspeed/'):]: row['sha256'] for row in source['package_sources']}
    expected_package[GENERATED[len('deepspeed/'):]] = source['generated_source']['sha256']
    namespace = {'np': np, 'json': json, 'os': os,
                 'EcosystemComparison': SimpleNamespace(forward_tolerance=.002, backward_tolerance=.01)}
    for name in ('_versions', '_divergence', '_comparison_floor'):
        _measured_function(record['jittor_commit'], 'compat/tests/torch/_ecosystem_harness.py', name, namespace)
    compare = _measured_function(record['jittor_commit'], MEASUREMENT_SOURCES[3], '_compare_engine_outputs', namespace)
    source_hashes = {path: hashlib.sha256(_committed_source(record['jittor_commit'], path)).hexdigest()
                     for path in MEASUREMENT_SOURCES}
    tree, blobs = _tree(str(ROOT), record['jittor_commit'])
    framework_sources = {path: hashlib.sha256(blobs[identifier]).hexdigest()
        for path, (mode, identifier) in tree.items() if mode != '120000'
        and (path == 'noxfile.py' or path.split('/')[0] in ('src', 'python', 'compat', 'backends', 'adapters'))}
    values, conditions, identities = {}, None, set()
    counts = {'checked_tensors': 0, 'parameter_gradients': 0, 'input_gradients': 0,
              'bitwise_gradients': 0, 'strict_resume_fields': 0, 'max_abs': 0.,
              'max_scaled_relative': 0., 'forward_and_state': 0., 'gradients': 0.}
    for scenario, offset in (('base', 0.), ('perturb', .125)):
        manifest = json.loads(load(scenario + '-sources').read_text())
        require(manifest.get('commit') == record['jittor_commit']
                and manifest.get('sha256') == framework_sources, 'Incomplete/stale committed framework source manifest')
        gate_path = load('gate-' + scenario + '-comparison')
        case = gate_path.parent
        gate = json.loads(gate_path.read_text())
        require(gate.get('status') == 'passed' and gate.get('commit') == record['jittor_commit']
                and gate.get('device') == record['device'] and gate.get('world_size') == 2
                and gate.get('dtype') == 'bfloat16' and gate.get('optimizer') == 'AdamW'
                and gate.get('zero_stage') == spec['zero_stage'] and gate.get('perturbation') == offset,
                'Actual maintained gate protocol/result differs')
        protocol_path = case / 'protocol.json'
        require(json.loads(protocol_path.read_text()) == {'device': record['device'], 'world_size': 2,
                'dtype': 'bfloat16', 'optimizer': 'AdamW', 'zero_stage': spec['zero_stage'], 'perturbation': offset},
                'Actual reserved case protocol differs')
        raw = gate.get('raw_sha256', {})
        require({'protocol.json', 'source.json', 'jittor/rank0.npz', 'jittor/rank1.npz',
                'torch/rank0.npz', 'torch/rank1.npz', 'jittor/launcher.json',
                'jittor/rank0-launch.json', 'jittor/rank1-launch.json'} <= raw.keys(),
                'Incomplete maintained gate raw evidence')
        for relative, checksum in raw.items():
            path = (case / relative).resolve()
            require(case.resolve() in path.parents and path.is_file() and digest(path) == checksum,
                    'Maintained gate raw artifact changed: ' + relative)
        require(load(scenario + '-sources') == (case / 'source.json').resolve(), 'Gate/source manifest path mismatch')
        for label, runtime in (('reference', 'torch'), ('candidate', 'jittor')):
            for rank in (0, 1):
                role = '%s-%s-rank%d' % (label, scenario, rank)
                path = load(role)
                require(path == (case / runtime / ('rank%d.npz' % rank)).resolve(), 'Gate/rank numerical artifact mismatch')
                with np.load(path, allow_pickle=False) as archive:
                    require(len(archive.files) == len(set(archive.files)) and 'metadata' in archive.files,
                            'Duplicate/missing DeepSpeed NPZ metadata')
                    metadata = json.loads(str(archive['metadata']))
                    arrays = {key: archive[key].copy() for key in archive.files if key != 'metadata'}
                require(type(metadata.get('rank')) is int and type(metadata.get('world_size')) is int
                        and type(metadata.get('seed')) is int, 'Actual process rank/world/seed types differ')
                require(metadata.get('runtime') == runtime and metadata.get('device') == record['device']
                        and metadata.get('rank') == rank and metadata.get('world_size') == 2
                        and metadata.get('dtype') == 'bfloat16' and metadata.get('zero_stage') == spec['zero_stage']
                        and metadata.get('optimizer') == 'AdamW' and metadata.get('perturbation') == offset,
                        'Actual DeepSpeed process protocol mismatch')
                require(metadata.get('protocol') == spec and metadata.get('jittor_commit') == record['jittor_commit']
                        and metadata.get('measurement_source_sha256') == source_hashes
                        and metadata.get('seed') == 0 and metadata.get('hf_hub_offline') is True,
                        'Stale/unfixed actual DeepSpeed sources or inputs')
                distribution = metadata.get('dependencies', {}).get('deepspeed', {})
                require(distribution.get('version') == record['downstream_versions'][label == 'reference' and 'oracle' or 'candidate']
                        == '0.19.7+f5af15c' and distribution.get('installed_files_sha256') == expected_package,
                        'Actually installed full DeepSpeed package differs from official source/wheel')
                interpreter = metadata.get('interpreter', {})
                require(interpreter.get('is_shim') is (runtime == 'jittor')
                        and interpreter.get('executable') and type(interpreter.get('pid')) is int,
                        'Missing actual independent interpreter identity')
                identity = (interpreter['executable'], interpreter['pid'])
                require(identity not in identities, 'Repeated process used for distinct protocol ranks/scenarios')
                identities.add(identity)
                if runtime == 'torch':
                    binary = Path(interpreter.get('binary_extension') or '')
                    require(binary.is_file() and binary.suffix in ('.so', '.pyd')
                            and digest(binary) == interpreter.get('binary_sha256'), 'Missing actual binary PyTorch oracle')
                    if rank == 0 and scenario == 'base':
                        require(platform['oracle'] == {**interpreter, 'torch_npu_version': 'not-applicable'}
                                and platform['oracle_platform'] == metadata.get('platform'), 'Recorded oracle platform differs')
                else:
                    require(metadata.get('fallback_policy') == 'error' and type(metadata.get('fallback_count')) is int
                            and metadata['fallback_count'] == 0
                            and metadata.get('native_runtime_flags', {}).get('use_cuda') == int(record['device'] == 'cuda'),
                            'Missing actual requested-device/zero-fallback enforcement')
                actual_platform = metadata.get('platform', {})
                declared = platform['oracle_platform'] if runtime == 'torch' else {key: platform[key] for key in actual_platform}
                require(actual_platform and actual_platform == declared, 'Actual process platform differs across ranks/scenarios')
                require(actual_platform['precision'] == metadata['runtime_conditions']['precision'], 'Actual precision switches differ')
                conditions = metadata['runtime_conditions'] if conditions is None else conditions
                require(metadata['runtime_conditions'] == conditions, 'Actual thread/affinity/precision conditions differ')
                expected_backend = 'nccl' if record['device'] == 'cuda' else ('mpi' if runtime == 'jittor' else 'gloo')
                require(metadata.get('process_group_backend') == expected_backend, 'Actual transport/provider differs')
                if runtime == 'jittor':
                    require(metadata.get('launch_entrypoint') == ('jtrun-module' if record['device'] == 'cuda'
                            else 'mpi-backend-verification'), 'Actual public CUDA/backend-only CPU launcher differs')
                if record['device'] == 'cuda':
                    physical = actual_platform['physical_devices'][rank]
                    require(metadata.get('rank_device_uuid') == physical['uuid']
                            and metadata.get('cuda_visible_devices') == actual_platform['card_indices'][rank]
                            and metadata.get('logical_device_index') == 0, 'Actual CUDA rank UUID/allocation mismatch')
                else:
                    require(metadata.get('cuda_visible_devices') == '' and metadata.get('rank_device_uuid') is None,
                            'CPU protocol used a CUDA allocation')
                require(metadata.get('engine_class') == 'deepspeed.runtime.engine.DeepSpeedEngine'
                        and metadata.get('optimizer_class') == ('deepspeed.runtime.bf16_optimizer.BF16_Optimizer'
                             if spec['zero_stage'] == 1 else 'deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer')
                        and metadata.get('optimizer_partition_sizes') == [166]
                        and metadata.get('model_parameter_elements') == 332 and metadata.get('global_steps') == 3
                        and metadata.get('microsteps') == 6 and metadata.get('parameter_gradients_per_microstep') == 8
                        and metadata.get('input_gradients_per_microstep') == 2
                        and metadata.get('trainable_parameters') == sorted(PARAMETERS)
                        and metadata.get('frozen_parameters') == [], 'Incomplete actual Engine/optimizer/gradient scope')
                _assert_arrays(arrays, metadata, spec['zero_stage'], offset, rank)
                counts['strict_resume_fields'] += metadata['strict_resume_fields']
                values[label, scenario, rank] = (arrays, metadata)
        for rank in (0, 1):
            actual, am = values['candidate', scenario, rank]
            reference, rm = values['reference', scenario, rank]
            class Archive(dict):
                @property
                def files(self):
                    return list(self)
            aa, rr = Archive(actual, metadata=np.asarray(json.dumps(am))), Archive(reference, metadata=np.asarray(json.dumps(rm)))
            try:
                errors = compare(aa, rr, device=record['device'], world_size=2, rank=rank, dtype='bfloat16',
                                 optimizer='AdamW', zero_stage=spec['zero_stage'], perturbation=offset)
            except (AssertionError, KeyError) as error:
                raise ValueError('Independent DeepSpeed numerical comparison failed: ' + str(error)) from error
            for key, value in errors.items():
                counts[key] = max(counts[key], value)
            for key, value in actual.items():
                counts['checked_tensors'] += 1
                counts['max_abs'] = max(counts['max_abs'], float(np.max(np.abs(value.astype('float64') - reference[key]))))
                if '_grad/' in key:
                    category = 'input_gradients' if '/input_grad/' in key else 'parameter_gradients'
                    counts[category] += 1
                    counts['bitwise_gradients'] += int(np.array_equal(value, reference[key]))
    for label in ('candidate', 'reference'):
        for rank in (0, 1):
            before, bm = values[label, 'base', rank]
            after, pm = values[label, 'perturb', rank]
            verify_perturbation(before, after, bm, pm)
    counts['max_scaled_relative'] = max(counts['forward_and_state'], counts['gradients'])
    counts['input_perturbation_verified'] = True
    return counts


def validate_measurements(record, artifacts, source):
    try:
        observed = measurements(record, artifacts, source)
        numeric = record['numerics']
        require(numeric.get('passed') is True and numeric.get('finite') is True
                and numeric.get('missing_tensors') == [] and numeric.get('atol') == .002 and numeric.get('rtol') == .01,
                'Changed original DeepSpeed numerical budget')
        for name in ('checked_tensors', 'max_abs', 'max_scaled_relative', 'input_perturbation_verified'):
            require(numeric.get(name) == observed[name], 'Claimed DeepSpeed numerical result differs: ' + name)
        for axis in ('parameter_gradients', 'input_gradients'):
            require(numeric.get(axis) == {'requested': observed[axis], 'checked': observed[axis]},
                    'Claimed DeepSpeed gradient scope differs')
        require(record.get('training') == {'steps': 3, 'loss_trajectory_passed': True,
                    'parameter_trajectory_passed': True, 'trainable': 8, 'frozen': 0,
                    'trainable_parameters': sorted(PARAMETERS), 'frozen_parameters': []},
                'Claimed DeepSpeed training scope differs')
        rows = {row['role']: row for row in artifacts}
        comparison = Path(rows['comparison']['path'])
        require(digest(comparison) == rows['comparison']['sha256']
                and json.loads(comparison.read_text()) == {'status': 'passed', 'protocol': record['protocol'],
                    'jittor_commit': record['jittor_commit'], 'device': record['device'], 'observed': observed},
                'Bound DeepSpeed numerical comparison differs from recomputation')
    except (OSError, KeyError, TypeError, AttributeError, AssertionError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        raise ValueError('Invalid DeepSpeed protocol measurement evidence') from error
    return observed
