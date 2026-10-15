"""Validate the fixed real Swift Trainer protocol; never infer library breadth."""
from pathlib import Path
import hashlib
import json

from collect_component_scorecard import compare_arrays, require
from swift_training_probe import UPSTREAM_CALLS

CASE = 'ms_swift_trainer_full_llama'


def read(path):
    return json.loads(Path(path).read_text())


def fields(value):
    names = value.get('trainable_parameters', [])
    buffers = value.get('training_protocol', {}).get('identity', {}).get('buffer_names', [])
    require(len(names) == len(set(names)) == 12 and names == sorted(names),
            'Expected the complete twelve-parameter full tiny Llama scope')
    require(len(buffers) == len(set(buffers)) and not set(names) & set(buffers), 'Invalid model buffers')
    initial = ['initial::' + name for name in names + buffers]
    step = ['output', 'loss', 'observed_grad::embedding_output',
            *['grad::' + name for name in names], *['parameter::' + name for name in names]]
    return initial + ['step/%d/%s' % (i, name) for i in range(3) for name in step]


def compare_pair(reference, candidate, source, baseline, *, seed):
    import numpy as np
    paths = [Path(reference), Path(candidate)]
    values = [read(path.with_suffix('.json')) for path in paths]
    expected_sources = {row['path']: row['sha256'] for row in source['package_sources']}
    require(bool(expected_sources), 'Missing complete Swift source inventory')
    for runtime, value in zip(('torch', 'jittor'), values):
        require(value.get('case') == CASE and value.get('runtime') == runtime,
                'Actual runtime/case mismatch')
        require(value.get('jittor_commit') == baseline and value.get('upstream') == source['upstream'],
                'Stale measured source identity')
        require(value.get('steps') == 3 and value.get('seed') == seed and value.get('learning_rate') == .001
                and value.get('mode') == 'train', 'Changed actual Trainer protocol')
        require(value.get('frozen_parameters') == [] and value.get('input_gradient_names') == []
                and value.get('integer_inputs') == ['attention_mask', 'input_ids', 'labels']
                and value.get('observed_intermediate_gradients') == ['embedding_output'],
                'Changed gradient/trainable scope')
        distribution = value.get('downstream_distribution', {})
        require(distribution.get('repo') == 'ms-swift'
                and distribution.get('version') == source['upstream']['version']
                and distribution.get('source_sha256') == expected_sources,
                'Incomplete actual installed Swift source binding')
        require(distribution.get('installed_files_sha256') and distribution.get('wheel_sha256'),
                'Missing installed payload/wheel identity')
        identity = value.get('training_protocol', {}).get('identity', {})
        expected_identity = {
            'model_family': 'llama', 'trainer': 'causal_lm', 'task': 'causal_lm',
            'trainer_class': 'swift.trainers.seq2seq_trainer.Seq2SeqTrainer',
            'trainer_registry_target': 'swift.trainers.Seq2SeqTrainer',
            'model_class': 'transformers.models.llama.modeling_llama.LlamaForCausalLM',
            'loader_class': 'swift.model.models.llama.LlamaLoader', 'template_type': 'llama'}
        require(all(identity.get(key) == expected for key, expected in expected_identity.items()),
                'Changed real registered model/loader/template/Trainer identity')
        refs = [{'repo': 'ms-swift', 'axis': axis, 'member': identity.get(axis)}
                for axis in ('model_family', 'trainer', 'task')]
        require(value.get('registry_refs') == refs and all(any(
            row['axis'] == ref['axis'] and ref['member'] in row['members']
            for row in source['axes']) for ref in refs), 'Unregistered actual loader/task/Trainer identity')
        require(identity.get('optimizer_updates') == 3 and identity.get('trainable_parameters') == value['trainable_parameters']
                and identity.get('frozen_parameters') == [], 'Incomplete real optimizer observations')
        require(value['training_protocol'].get('owner') == 'swift.pipelines.train.sft.SwiftSft.main'
                and value['training_protocol'].get('trainer_owner') == 'swift.trainers.Seq2SeqTrainer.train'
                and value['training_protocol'].get('optimizer') == 'torch.optim.SGD'
                and value['training_protocol'].get('scheduler') == 'constant'
                and value['training_protocol'].get('gradient_clipping') is False,
                'Changed downstream-owned training path')
        require(value['training_protocol'].get('upstream_calls') == UPSTREAM_CALLS,
                'Missing actual registered loader/template/Trainer call observations')
        require(value.get('hf_hub_offline') is True, 'Offline guard missing')
        require(value.get('interpreter', {}).get('is_shim') == (runtime == 'jittor'), 'Invalid independent runtime')
        inputs = value.get('observed_inputs', {})
        require(set(inputs) == {'step/%d/observed_input::%s' % (step, name)
                for step in range(3) for name in value['integer_inputs']}, 'Incomplete real model integer inputs')
        for key, entry in inputs.items():
            dtype = np.dtype(entry['dtype'])
            data = np.array(entry['values'], dtype=dtype)
            require(dtype.kind in 'biu' and list(data.shape) == entry['shape']
                    and hashlib.sha256(data.tobytes()).hexdigest() == entry['sha256'],
                    'Invalid observed integer input: ' + key)
        require(set(value.get('observed_tensor_devices', {})) == set(fields(value)) | set(inputs)
                and set(value['observed_tensor_devices'].values()) == {value.get('device')},
                'Incomplete actual tensor residency observations')
    a, b = values
    distributions = [value['downstream_distribution'] for value in values]
    require(all(isinstance(value.get('package'), str) and value['package'] for value in distributions),
            'Missing actual installed package location')
    for name in ('registry_refs', 'trainable_parameters', 'training_protocol', 'downstream_distribution',
                 'tf32', 'runtime_conditions', 'dependencies', 'model_artifacts', 'device',
                 'measurement_source_sha256', 'observed_inputs', 'integer_inputs'):
        left, right = a.get(name), b.get(name)
        if name == 'downstream_distribution':
            # Each worker already hashes its actual package. Independent venvs
            # have different locations; all content and identity fields remain
            # exact comparisons, and the raw locations remain in both reports.
            left, right = ({key: value for key, value in item.items() if key != 'package'}
                           for item in (left, right))
        require(left == right, 'Runtime condition/source/protocol mismatch: ' + name)
    require(a['interpreter'].get('binary_sha256'), 'Oracle binary extension identity missing')
    require(b.get('fallback_policy') == 'error' and b.get('fallback_count') == 0,
            'Candidate zero-fallback guard missing')
    require(set(fields(a)) == set(fields(b)), 'Asymmetric requested field set')
    rows = compare_arrays(*paths, fields(a), .002, .01)
    with np.load(paths[0]) as ref, np.load(paths[1]) as cand:
        require(all(ref[name].dtype.kind == cand[name].dtype.kind == 'f' for name in fields(a)),
                'Only floating numerical fields may contribute to the original comparison scale')
        for name in fields(a):
            if name.startswith('initial::'):
                require(np.array_equal(ref[name], cand[name]), 'Different actual initial weights: ' + name)
        names = a['trainable_parameters']
        require(any(not np.array_equal(ref['initial::' + name], ref['step/2/parameter::' + name])
                    for name in names), 'No actual real-Trainer parameter update')
    return {'case': CASE, 'status': 'passed', 'seed': seed, 'rows': rows,
            'reference_result': a, 'candidate_result': b,
            'max_abs': max(row['max_abs'] for row in rows),
            'max_scaled_relative': max(row['max_scaled_relative'] for row in rows)}
