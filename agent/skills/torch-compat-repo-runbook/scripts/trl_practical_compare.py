"""Fail-closed posthoc practical TRL comparison, distinct from measured producers.

Require the complete Qwen3-0.6B/rank-8 adapter protocol, all training observations
and checkpoint provenance. Only scalar Adam counters allow dtype normalization;
representation mismatches remain explicit and prevent strict API parity.
"""
import argparse
import json
import math
from pathlib import Path

from trl_final_logits_probe import checked_array
from trl_rl_probe import compare_arrays
from trl_sft_probe import digest, write_json


def expected_adapters():
    return {f'base_model.model.model.layers.{layer}.self_attn.{projection}.lora_{side}.default.weight': shape
            for layer in range(28) for projection in ('q_proj', 'v_proj')
            for side, shape in (('A', [8, 1024]), ('B', [2048 if projection == 'q_proj' else 1024, 8]))}


def validate_manifest(manifest):
    """Validate inventory against the producer protocol, not the other trace."""
    if manifest.get('status') != 'complete' or manifest.get('fallback_count') != 0:
        raise AssertionError('Require completed zero-fallback runs')
    if manifest.get('runtime') not in ('torch', 'jittor'):
        raise AssertionError('Unknown practical runtime')
    protocol = manifest['protocol']
    fixed = dict(steps=10, checkpoint_step=5, optimizer='AdamW', learning_rate=1e-4,
                 betas=[.9, .999], epsilon=1e-8, weight_decay=.01, scheduler='linear',
                 warmup_steps=0, batch_size=1, max_length=32, order='SequentialSampler',
                 attention='eager', lora=dict(r=8, alpha=16, dropout=0., targets=['q_proj', 'v_proj']))
    if any(protocol.get(key) != value for key, value in fixed.items()):
        raise AssertionError('Unsupported practical training protocol')
    if len(protocol.get('texts', [])) != 10 or len(set(protocol['texts'])) != 10 or len(protocol.get('eval_prompts', [])) != 3:
        raise AssertionError('Require ten distinct texts and three evaluation prompts')
    precision = protocol.get('precision')
    dtype = {'fp32': 'torch.float32', 'fp16': 'torch.float16', 'bf16': 'torch.bfloat16'}.get(precision)
    if dtype is None or manifest.get('forward_dtypes') != [dtype]:
        raise AssertionError('Missing actual forward precision evidence')
    if manifest.get('device') not in ('cpu', 'cuda') or (precision != 'fp32' and manifest['device'] != 'cuda'):
        raise AssertionError('Unsupported precision/device protocol')
    adapters = expected_adapters()
    if manifest.get('trainable_parameters') != adapters or manifest.get('trainable_parameter_count') != 1146880:
        raise AssertionError('Incomplete trainable adapter inventory/count')
    if not set(adapters).issubset(manifest.get('initial', {})) or not manifest.get('initial_adapters_sha256'):
        raise AssertionError('Missing initial adapter provenance')
    resumed = manifest.get('resumed_from_step')
    if resumed not in (None, 5):
        raise AssertionError('Invalid resumed step')
    steps = list(range(6 if resumed == 5 else 1, 11))
    if manifest.get('optimizer_steps') != steps:
        raise AssertionError('Incomplete actual optimizer updates')
    for name in ('updates', 'gradient_norms'):
        values = manifest.get(name, [])
        if len(values) != len(steps) or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in values):
            raise AssertionError('Missing/nonpositive actual ' + name)
    for name in ('rng_before_loss', 'scheduler', 'scaler'):
        if set(manifest.get(name, {})) != {str(step) for step in steps}:
            raise AssertionError('Incomplete ' + name + ' inventory')
    rng_keys = {'python', 'numpy', 'cpu'} | ({'cuda'} if manifest['device'] == 'cuda' else set())
    for step in steps:
        key = str(step)
        if set(manifest['rng_before_loss'][key]) != rng_keys or not all(manifest['rng_before_loss'][key].values()):
            raise AssertionError('Incomplete RNG state')
        scheduler = manifest['scheduler'][key]
        scheduler_keys = {'base_lrs', 'last_epoch', '_step_count', '_is_initial',
                          '_get_lr_called_within_step', '_last_lr', 'lr_lambdas'}
        if not isinstance(scheduler, dict) or set(scheduler) != scheduler_keys or scheduler['last_epoch'] != step:
            raise AssertionError('Incomplete scheduler state')
        if scheduler['_step_count'] != step + 1 or not scheduler['base_lrs'] or len(scheduler['_last_lr']) != len(scheduler['base_lrs']):
            raise AssertionError('Invalid scheduler state')
        scaler = manifest['scaler'][key]
        if precision == 'fp16':
            if not isinstance(scaler, dict) or not {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'} <= scaler.keys():
                raise AssertionError('Incomplete FP16 scaler state')
            if not math.isfinite(scaler['scale']) or scaler['scale'] <= 0:
                raise AssertionError('Invalid FP16 scaler scale')
        elif scaler is not None:
            raise AssertionError('Unexpected scaler for non-FP16 run')
    arrays = manifest['arrays']
    expected = set()
    for step in steps:
        prefix = f'step/{step}/'
        expected.add(prefix + 'loss')
        expected.update(prefix + 'input/' + name for name in ('input_ids', 'attention_mask', 'labels'))
        for name, shape in adapters.items():
            for category in ('gradient/', 'parameter/', 'adam/'):
                suffixes = ('/step', '/exp_avg', '/exp_avg_sq') if category == 'adam/' else ('',)
                for suffix in suffixes:
                    field = prefix + category + name + suffix
                    expected.add(field)
                    info = arrays.get(field, {})
                    if info.get('shape') != ([] if suffix == '/step' else shape):
                        raise AssertionError('Incomplete adapter observation inventory/shape: ' + field)
                    if suffix == '/step':
                        if info.get('exact') is not True or info.get('dtype') not in ('float32', 'float64', 'int64'):
                            raise AssertionError('Invalid Adam counter metadata')
                    elif info.get('dtype') != 'float32':
                        raise AssertionError('Invalid adapter observation dtype')
        inputs = [arrays.get(prefix + 'input/' + name, {}) for name in ('input_ids', 'attention_mask', 'labels')]
        shape = inputs[0].get('shape', [])
        if len(shape) != 2 or shape[0] != 1 or not 1 <= shape[1] <= 32:
            raise AssertionError('Incomplete input inventory/shape')
        if any(info.get('shape') != shape or info.get('dtype') != 'int64' or info.get('exact') is not True for info in inputs):
            raise AssertionError('Invalid exact input metadata')
        if arrays.get(prefix + 'loss', {}).get('shape') != []:
            raise AssertionError('Missing scalar loss inventory')
    for index in range(3):
        prefix = f'final/{index}/'
        expected.update(prefix + name for name in ('input_ids', 'logits', 'next_logits'))
        inputs = arrays.get(prefix + 'input_ids', {})
        shape = inputs.get('shape', [])
        if len(shape) != 2 or shape[0] != 1 or shape[1] < 1 or inputs.get('dtype') != 'int64' or inputs.get('exact') is not True:
            raise AssertionError('Missing final input inventory')
        if arrays.get(prefix + 'logits', {}).get('shape') != [1, shape[1], 151936] or arrays.get(prefix + 'next_logits', {}).get('shape') != [151936]:
            raise AssertionError('Missing full-vocabulary logit inventory')
    if set(arrays) != expected:
        raise AssertionError('Observation inventory mismatch')
    return steps


def validate_resume_binding(reference_path, ref, cand):
    required = {'adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt', 'rng_state.pth',
                'scheduler.pt', 'trainer_state.json', 'training_args.bin'}
    if ref['protocol']['precision'] == 'fp16':
        required.add('scaler.pt')
    prefix = 'checkpoint-5/'
    bound = {key[len(prefix):]: value for key, value in ref.get('checkpoint_files', {}).items() if key.startswith(prefix)}
    if not required <= bound.keys() or cand.get('resume_checkpoint_hashes') != bound:
        raise AssertionError('Resume checkpoint-five hash binding mismatch')
    root = (reference_path / 'trainer' / 'checkpoint-5').resolve()
    for name, sha in bound.items():
        path = (root / name).resolve()
        if root not in path.parents or digest(path) != sha:
            raise AssertionError('Resume checkpoint-five file hash mismatch')


def compare(reference_path, candidate_path, repeat=False, resume=False):
    import numpy as np
    reference_path, candidate_path = Path(reference_path).resolve(), Path(candidate_path).resolve()
    if reference_path == candidate_path or reference_path in candidate_path.parents or candidate_path in reference_path.parents:
        raise AssertionError('Require distinct nonoverlapping trace directories')
    if repeat and resume:
        raise AssertionError('Repeat and resume are separate protocols')
    ref = json.loads((reference_path / 'manifest.json').read_text())
    cand = json.loads((candidate_path / 'manifest.json').read_text())
    validate_manifest(ref)
    validate_manifest(cand)
    if ref['resumed_from_step'] is not None or (not resume and cand['resumed_from_step'] is not None):
        raise AssertionError('Require uninterrupted reference/full-run comparison')
    if resume:
        if ref['runtime'] != cand['runtime'] or cand['resumed_from_step'] != 5:
            raise AssertionError('Require same-runtime fresh process resumed from step five')
        validate_resume_binding(reference_path, ref, cand)
    elif repeat:
        if ref['runtime'] != cand['runtime']:
            raise AssertionError('Repeat control requires the same runtime')
    elif ref['runtime'] != 'torch' or cand['runtime'] != 'jittor':
        raise AssertionError('Require independent Torch/Jittor runtimes')
    for name in ('protocol', 'versions', 'trl_source', 'peft_source', 'model_artifacts', 'initial', 'device', 'tool_sources',
                 'trainable_parameters', 'trainable_parameter_count', 'initial_adapters_sha256'):
        if ref[name] != cand[name]:
            raise AssertionError('Protocol mismatch: ' + name)
    expected = {key for key in ref['arrays'] if not resume or not key.startswith('step/') or int(key.split('/')[1]) > 5}
    if expected != set(cand['arrays']):
        raise AssertionError('Observation inventory mismatch')
    if resume:
        for name in ('rng_before_loss', 'scheduler', 'scaler'):
            for step in range(6, 11):
                if ref[name][str(step)] != cand[name][str(step)]:
                    raise AssertionError('Resume changed ' + name + ' at step ' + str(step))
    rows, mismatches = {}, []
    for key in sorted(expected):
        a = checked_array(reference_path, ref['arrays'][key])
        b = checked_array(candidate_path, cand['arrays'][key])
        counter = '/adam/' in key and key.endswith('/step')
        if counter:
            step = int(key.split('/')[1])
            if a.shape != () or b.shape != () or a.item() != step or b.item() != step:
                raise AssertionError('Adam counter does not equal actual optimizer step')
            if a.dtype != b.dtype:
                mismatches.append(dict(field=key, reference_dtype=str(a.dtype), candidate_dtype=str(b.dtype)))
            if not resume:
                a, b = a.astype(np.float64), b.astype(np.float64)
        rows[key] = compare_arrays(a, b, exact=resume or counter or ref['arrays'][key]['exact'])
    numerical = all(row['passed'] for row in rows.values())
    byte_equal = all(ref['arrays'][key]['sha256'] == cand['arrays'][key]['sha256'] for key in rows)
    return dict(repeat=repeat, resume=resume, passed=numerical and not mismatches,
                byte_equal=byte_equal, numerical_passed=numerical, strict_passed=numerical and not mismatches,
                counter_representation_mismatches=mismatches,
                max_abs=max(row['max_abs'] for row in rows.values()),
                failed_fields=sum(not row['passed'] for row in rows.values()),
                failed_elements=sum(row['mismatched_elements'] for row in rows.values()), fields=rows,
                reference_manifest=digest(reference_path / 'manifest.json'),
                candidate_manifest=digest(candidate_path / 'manifest.json'),
                comparison_tool_sources={name: digest(Path(__file__).with_name(name)) for name in
                    ('trl_practical_compare.py', 'trl_final_logits_probe.py', 'trl_rl_probe.py', 'trl_sft_probe.py')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'candidate', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--repeat', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Do not overwrite previous comparison')
    write_json(args.output, compare(args.reference, args.candidate, args.repeat, args.resume))


if __name__ == '__main__':
    main()
