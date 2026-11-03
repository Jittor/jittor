"""Fail-closed comparisons for the method-aware full-Qwen LoRA protocol."""
import argparse
import json
from pathlib import Path

from trl_final_logits_probe import checked_array
from trl_practical_compare import expected_adapters, validate_resume_binding
from trl_rl_probe import compare_arrays
from trl_sft_probe import digest, write_json


INPUTS = {
    'sft': {'input_ids', 'attention_mask', 'labels'},
    'dpo': {'prompt_input_ids', 'prompt_attention_mask', 'chosen_input_ids',
            'chosen_attention_mask', 'rejected_input_ids', 'rejected_attention_mask'},
    'grpo': {'prompt_ids', 'prompt_mask', 'completion_ids', 'completion_mask', 'advantages'},
}


def validate(manifest):
    if manifest.get('status') != 'complete' or manifest.get('fallback_count') != 0:
        raise AssertionError('Require completed zero-fallback training')
    if manifest.get('runtime') not in ('torch', 'jittor'):
        raise AssertionError('Unknown tensor runtime')
    protocol = manifest['protocol']
    method = protocol.get('algorithm')
    if method not in INPUTS or protocol.get('steps') != 10 or protocol.get('checkpoint_step') != 5:
        raise AssertionError('Unknown ten-update method protocol')
    if protocol.get('gradient_accumulation') not in (1, 2) or protocol.get('max_length') not in (32, 128):
        raise AssertionError('Unsupported configuration')
    fixed = dict(optimizer='AdamW', learning_rate=1e-4, betas=[.9, .999], epsilon=1e-8,
                 weight_decay=.01, scheduler='linear', warmup_steps=0, attention='eager',
                 lora=dict(r=8, alpha=16, dropout=0., targets=['q_proj', 'v_proj']))
    if any(protocol.get(key) != value for key, value in fixed.items()):
        raise AssertionError('Changed optimizer or adapter protocol')
    records = protocol.get('records', [])
    if len(records) != 10 or len({json.dumps(row, sort_keys=True) for row in records}) != 10:
        raise AssertionError('Require ten distinct records')
    precision = protocol.get('precision')
    expected_dtype = {'fp32': 'torch.float32', 'fp16': 'torch.float16', 'bf16': 'torch.bfloat16'}.get(precision)
    if expected_dtype is None or manifest.get('forward_dtypes') != [expected_dtype]:
        raise AssertionError('Missing actual compute precision')
    if manifest.get('device') not in ('cpu', 'cuda') or (precision != 'fp32' and manifest['device'] != 'cuda'):
        raise AssertionError('Invalid precision/device')
    adapters = expected_adapters()
    if manifest.get('trainable_parameters') != adapters or manifest.get('trainable_parameter_count') != 1146880:
        raise AssertionError('Incomplete Qwen LoRA inventory')
    if not set(adapters) <= manifest.get('initial', {}).keys() or not manifest.get('initial_adapters_sha256'):
        raise AssertionError('Missing initialization binding')
    resumed = manifest.get('resumed_from_step')
    if resumed not in (None, 5):
        raise AssertionError('Unsupported resume step')
    steps = list(range(6 if resumed == 5 else 1, 11))
    if manifest.get('optimizer_steps') != steps:
        raise AssertionError('Incomplete actual optimizer steps')
    if any(set(manifest.get(name, {})) != {str(step) for step in steps}
           for name in ('micro_steps', 'scheduler', 'scaler')):
        raise AssertionError('Incomplete step state inventory')
    for name in ('updates', 'gradient_norms'):
        values = manifest.get(name, [])
        if len(values) != len(steps) or not any(value > 0 for value in values):
            raise AssertionError('Missing nonzero training signal')
    arrays = manifest['arrays']
    required = set()
    rng_keys = set()
    for step in steps:
        count = manifest['micro_steps'][str(step)]
        if count != protocol['gradient_accumulation']:
            raise AssertionError('Incorrect actual microbatch count')
        for micro in range(count):
            prefix = 'step/%d/micro/%d/' % (step, micro)
            required.add(prefix+'loss')
            required.update(prefix+'input/'+name for name in INPUTS[method])
            rng_keys.add('%d/%d' % (step, micro))
        if method == 'sft' and protocol['max_length'] == 128 and count == 2:
            lengths = [arrays.get('step/%d/micro/%d/input/input_ids' % (step, micro), {}).get('shape', [0, 0])[-1]
                       for micro in range(count)]
            if len(set(lengths)) != 2 or max(lengths) != 128:
                raise AssertionError('Require actual unequal-length accumulated microbatches reaching 128')
        for name, shape in adapters.items():
            for category, suffixes in (('gradient/', ('',)), ('parameter/', ('',)),
                                       ('adam/', ('/step', '/exp_avg', '/exp_avg_sq'))):
                for suffix in suffixes:
                    field = 'step/%d/' % step + category + name + suffix
                    required.add(field)
                    info = arrays.get(field, {})
                    if info.get('shape') != ([] if suffix == '/step' else shape):
                        raise AssertionError('Missing adapter/gradient/state shape: '+field)
                    if suffix != '/step' and info.get('dtype') != 'float32':
                        raise AssertionError('Adapter state precision changed')
                    if suffix == '/step' and (info.get('exact') is not True or info.get('dtype') not in ('float32', 'float64')):
                        raise AssertionError('Invalid step-counter representation')
    if set(manifest.get('rng_before_loss', {})) != rng_keys:
        raise AssertionError('Incomplete RNG observation inventory')
    for key in rng_keys:
        expected_rng = {'cpu', 'python', 'numpy'} | ({'cuda'} if manifest['device'] == 'cuda' else set())
        if set(manifest['rng_before_loss'][key]) != expected_rng:
            raise AssertionError('Incomplete runtime RNG state')
    for step in steps:
        if manifest['scheduler'][str(step)].get('last_epoch') != step:
            raise AssertionError('Incorrect scheduler progress')
        scaler = manifest['scaler'][str(step)]
        if precision == 'fp16' and (not isinstance(scaler, dict) or scaler.get('scale', 0) <= 0):
            raise AssertionError('Missing real FP16 GradScaler state')
        if precision != 'fp16' and scaler is not None:
            raise AssertionError('Unexpected scaler')
    for index in range(3):
        required.update('final/%d/' % index + suffix for suffix in ('input_ids', 'logits', 'next_logits'))
    for index in range(3):
        prefix = 'final/%d/' % index
        if arrays.get(prefix+'next_logits', {}).get('shape') != [151936]:
            raise AssertionError('Missing complete final-vocabulary logits')
        shape = arrays.get(prefix+'logits', {}).get('shape', [])
        inputs = arrays.get(prefix+'input_ids', {}).get('shape', [])
        if len(shape) != 3 or shape[0] != 1 or shape[2] != 151936 or inputs != shape[:2] or shape[1] < 1:
            raise AssertionError('Invalid final prompt/logits shape')
    if not required <= arrays.keys():
        raise AssertionError('Missing required training/final observation')
    if method == 'grpo':
        calls = manifest.get('generation_calls', [])
        if {call['before_global_step']+1 for call in calls} != set(steps):
            raise AssertionError('Incomplete GRPO generation inventory')
        for step in steps:
            fields = [key for key in arrays if key.startswith('step/%d/generation/' % step)]
            if not fields or not any(key.endswith('/output_ids') for key in fields):
                raise AssertionError('Missing generated-token observations')


def compare(reference_path, candidate_path, resume=False, repeat=False, initial_adapters=None):
    import numpy as np
    reference_path, candidate_path = Path(reference_path).resolve(), Path(candidate_path).resolve()
    if reference_path == candidate_path or reference_path in candidate_path.parents or candidate_path in reference_path.parents:
        raise AssertionError('Require distinct trace directories')
    ref, cand = [json.loads((path/'manifest.json').read_text()) for path in (reference_path, candidate_path)]
    for manifest in (ref, cand):
        validate(manifest)
    if resume and repeat:
        raise AssertionError('Repeat and resume are separate controls')
    if resume:
        if ref['runtime'] != cand['runtime'] or cand['resumed_from_step'] != 5:
            raise AssertionError('Require own-runtime checkpoint continuation')
        validate_resume_binding(reference_path, ref, cand)
    elif repeat:
        if ref['runtime'] != cand['runtime'] or cand['resumed_from_step'] is not None:
            raise AssertionError('Invalid uninterrupted-repeat control')
    elif ref['runtime'] != 'torch' or cand['runtime'] != 'jittor' or cand['resumed_from_step'] is not None:
        raise AssertionError('Require an independent Torch oracle')
    if ref['resumed_from_step'] is not None:
        raise AssertionError('Require uninterrupted reference')
    for key in ('protocol', 'versions', 'trl_source', 'peft_source', 'model_artifacts', 'initial',
                'device', 'tool_sources', 'trainable_parameters', 'initial_adapters_sha256'):
        if ref[key] != cand[key]:
            raise AssertionError('Protocol/source mismatch: '+key)
    expected = {key for key in ref['arrays'] if not resume or not key.startswith('step/') or int(key.split('/')[1]) > 5}
    if expected != set(cand['arrays']):
        raise AssertionError('Observation inventory mismatch')
    if resume:
        for key in ('rng_before_loss', 'scheduler', 'scaler'):
            for name, value in cand[key].items():
                if ref[key].get(name) != value:
                    raise AssertionError('Resume changed '+key)
    rows, representation = {}, []
    for key in sorted(expected):
        a = checked_array(reference_path, ref['arrays'][key])
        b = checked_array(candidate_path, cand['arrays'][key])
        counter = '/adam/' in key and key.endswith('/step')
        if counter:
            if a.shape != () or b.shape != () or a.item() != int(key.split('/')[1]) or b.item() != a.item():
                raise AssertionError('Wrong live optimizer counter value')
            if a.dtype != b.dtype:
                representation.append(key)
                a, b = a.astype(np.float64), b.astype(np.float64)
        rows[key] = compare_arrays(a, b, exact=resume or repeat or counter or ref['arrays'][key]['exact'])
    result = dict(strict_passed=not representation and all(row['passed'] for row in rows.values()),
                  numerical_passed=all(row['passed'] for row in rows.values()),
                  failed_fields=sum(not row['passed'] for row in rows.values()),
                  max_abs=max(row['max_abs'] for row in rows.values()), fields=rows,
                  counter_representation_mismatches=representation, resume=resume, repeat=repeat,
                  byte_equal=all(ref['arrays'][key]['sha256'] == cand['arrays'][key]['sha256'] for key in expected),
                  reference_manifest=digest(reference_path/'manifest.json'), candidate_manifest=digest(candidate_path/'manifest.json'))
    if initial_adapters is not None:
        if digest(initial_adapters) != ref['initial_adapters_sha256']:
            raise AssertionError('Changed initial adapters')
        error_sq = update_sq = 0.
        with np.load(initial_adapters, allow_pickle=False) as initial:
            if set(initial.files) != set(expected_adapters()):
                raise AssertionError('Incomplete initial adapter archive')
            for name in expected_adapters():
                key = 'step/10/parameter/'+name
                a = checked_array(reference_path, ref['arrays'][key]).astype(np.float64)
                b = checked_array(candidate_path, cand['arrays'][key]).astype(np.float64)
                error_sq += float(np.square(b-a).sum())
                update_sq += float(np.square(a-initial[name].astype(np.float64)).sum())
        result['final_adapter_error_over_update'] = (error_sq/update_sq)**.5 if update_sq else None
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'candidate', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--repeat', action='store_true')
    parser.add_argument('--initial-adapters', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, compare(args.reference, args.candidate, args.resume, args.repeat, args.initial_adapters))
