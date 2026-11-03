"""Audit a full-checkpoint TRL pair and optionally prune only verified oracle bulk.

Default operation is read-only. Intermediate samples are never represented as
complete gradient coverage. Candidate bulk payloads were intentionally not
retained: their recorded comparisons cannot be independently recomputed here.
A pruning plan is written exclusively before deletion, and an exclusive
completion receipt afterwards; original producer manifests stay immutable.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

from trl_final_logits_probe import aggregate_logits, checked_array, logit_metrics
from trl_rl_probe import compare_arrays
from trl_sft_probe import digest
from trl_trace_archive import ArchiveReader


BULK = re.compile(r'^step/(\d+)/(final_gradient|final_parameter)/(.+)/0$')
ARRAY_PATH = re.compile(r'^arrays/[0-9a-f]{64}\.npy$')
CAUSAL_PARAMETERS = 596049920
CLASSIFIER_PARAMETERS = 596050944


def write_exclusive(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def aggregate(rows):
    rows = list(rows)
    if not rows:
        return None
    delta = math.hypot(*(row['difference_l2'] for row in rows))
    reference = math.hypot(*(row['reference_l2'] for row in rows))
    return dict(fields=len(rows), elements=sum(row['elements'] for row in rows),
                failed_fields=sum(not row['passed'] for row in rows),
                failed_elements=sum(row['mismatched_elements'] for row in rows),
                max_abs=max(row['max_abs'] for row in rows),
                global_relative_l2=delta / reference if reference else None)


def validate_comparison(row, info):
    size = 1
    for extent in info['shape']:
        size *= extent
    if row['elements'] != size or row['exact'] != info['exact']:
        raise AssertionError('Comparison size/exactness mismatch')
    failed = row['mismatched_elements']
    if not isinstance(failed, int) or not 0 <= failed <= size or row['passed'] != (failed == 0):
        raise AssertionError('Invalid comparison failure count')
    for key in ('max_abs', 'reference_l2', 'candidate_l2', 'difference_l2'):
        if not math.isfinite(row[key]) or row[key] < 0:
            raise AssertionError('Invalid comparison magnitude')


def compare_recorded(actual, recorded):
    if actual.keys() != recorded.keys():
        raise AssertionError('Comparison fields changed')
    for key in actual:
        a, b = actual[key], recorded[key]
        if isinstance(a, float) and isinstance(b, float):
            if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15):
                raise AssertionError('Recorded comparison differs: ' + key)
        elif a != b:
            raise AssertionError('Recorded comparison differs: ' + key)


def validate_full_trainable(manifest):
    roles = ('policy', 'value') if manifest['protocol']['algorithm'] == 'ppo' else ('policy',)
    expected = {}
    for role in roles:
        tensors = manifest['initial'][role]['tensors']
        parameters = {name[len('parameter/'):]: value['shape']
                      for name, value in tensors.items() if name.startswith('parameter/')}
        count = 0
        for shape in parameters.values():
            size = 1
            for extent in shape:
                size *= extent
            count += size
        classifier = role == 'value' or manifest['protocol']['algorithm'] == 'reward'
        known_count = CLASSIFIER_PARAMETERS if classifier else CAUSAL_PARAMETERS
        if count != known_count or manifest['initial'][role]['parameter_count'] != count:
            raise AssertionError('Initial role is not the complete Qwen3-0.6B parameter inventory')
        prefix = ('policy.' if role == 'policy' else 'value_model.') if len(roles) == 2 else ''
        expected.update({prefix + name: shape for name, shape in parameters.items()})
    if manifest['trainable_parameters'] != expected:
        raise AssertionError('Trainable inventory does not include every full policy/value parameter')


def validate_pair(reference, candidate):
    if reference['runtime'] != 'torch' or candidate['runtime'] != 'jittor':
        raise AssertionError('Require independent Torch/Jittor pair')
    for manifest in (reference, candidate):
        if manifest['status'] != 'complete' or manifest['fallback_count'] != 0:
            raise AssertionError('Require complete zero-fallback runs')
        if 'trl_full_probe.py' not in manifest['tool_sources']:
            raise AssertionError('Not a full-checkpoint probe manifest')
        steps = manifest['protocol']['steps']
        if not 1 <= steps <= 10 or manifest['optimizer_steps'] != steps:
            raise AssertionError('Incomplete optimizer updates')
        if manifest['case']['expected_optimizer_steps'] != steps:
            raise AssertionError('Expected optimizer count differs')
        if len(manifest['gradient_norms']) != steps or len(manifest['parameter_updates']) != steps:
            raise AssertionError('Incomplete step statistics')
        if not all(math.isfinite(v) and v >= 0 for v in manifest['gradient_norms']):
            raise AssertionError('Invalid gradient norms')
        if not any(manifest['gradient_norms']) or not any(v['changed_elements'] for v in manifest['parameter_updates']):
            raise AssertionError('No genuine optimization observed')
    for key in ('device', 'protocol', 'versions', 'trl_source', 'tool_sources', 'initial',
                'trainable_parameters', 'frozen_parameters', 'case'):
        if reference[key] != candidate[key]:
            raise AssertionError('Provenance/inventory mismatch: ' + key)
    if reference['generation_calls'] != candidate['generation_calls']:
        raise AssertionError('Controlled generation protocol differs')
    if set(reference['arrays']) != set(candidate['arrays']) or set(candidate['comparison']) != set(candidate['arrays']):
        raise AssertionError('Missing observation or comparison')
    validate_full_trainable(reference)
    validate_full_trainable(candidate)
    names = reference['trainable_parameters']
    if not names:
        raise AssertionError('Empty trained parameter inventory')
    steps = reference['protocol']['steps']
    expected_bulk = {'step/%d/%s/%s/0' % (steps - 1, kind, name)
                     for kind in ('final_gradient', 'final_parameter') for name in names}
    actual_bulk = {name for name in reference['arrays'] if BULK.match(name)}
    if expected_bulk != actual_bulk:
        raise AssertionError('Missing or unexpected final bulk tensor')
    for key in expected_bulk:
        name = BULK.fullmatch(key).group(3)
        if reference['arrays'][key]['shape'] != names[name]:
            raise AssertionError('Final bulk shape differs from full parameter inventory')
    for step in range(steps):
        for kind in ('gradient_sample', 'gradient_norm', 'updated_parameter_sample'):
            expected = {'step/%d/%s/%s/0' % (step, kind, name) for name in names}
            observed = {name for name in reference['arrays'] if name.startswith('step/%d/%s/' % (step, kind))}
            if expected != observed:
                raise AssertionError('Incomplete intermediate parameter observation')
    expected_eval = {'step/%d/final_eval/%d/logits/0' % (steps, index) for index in range(10)}
    if not expected_eval.issubset(reference['arrays']):
        raise AssertionError('Incomplete ten-prompt final evaluation')
    expected_inputs = {'step/%d/final_eval/%d/input_ids/0' % (steps, index) for index in range(10)}
    if not expected_inputs.issubset(reference['arrays']) or any(
            not reference['arrays'][key]['exact'] for key in expected_inputs):
        raise AssertionError('Missing exact final evaluation input IDs')
    for step in range(steps):
        observations = [name[len('step/%d/' % step):] for name in reference['arrays']
                        if name.startswith('step/%d/' % step)]
        if not any(name.startswith('embedding_gradient/') for name in observations):
            raise AssertionError('Missing per-step embedding gradient observations')
        if not any(name.startswith(('forward/input/', 'forward/positional/', 'input::', 'input/'))
                   for name in observations):
            raise AssertionError('Missing per-step training input observations')
        if not any(name.startswith('forward/output/') for name in observations):
            raise AssertionError('Missing per-step training output observations')
        if reference['protocol']['algorithm'] not in ('ppo', 'rloo') and not any(
                name.startswith('loss/') for name in observations):
            raise AssertionError('Missing per-step loss observations')
    if candidate['numerical_passed'] != all(row['passed'] for row in candidate['comparison'].values()):
        raise AssertionError('Numerical completion flag disagrees with comparisons')


def audit(reference_path, candidate_path):
    sources = {name: digest(Path(__file__).with_name(name)) for name in
               ('trl_full_collect.py', 'trl_trace_archive.py')}
    with ArchiveReader(reference_path) as read_reference, ArchiveReader(candidate_path) as read_candidate:
        result = _audit(reference_path, candidate_path, read_reference, read_candidate)
    if any(digest(Path(__file__).with_name(name)) != sha for name, sha in sources.items()):
        raise AssertionError('Audit source changed during validation')
    result['audit_sources'] = sources
    return result


def _audit(reference_path, candidate_path, read_reference, read_candidate):
    import numpy as np
    reference_path, candidate_path = Path(reference_path).resolve(), Path(candidate_path).resolve()
    if reference_path == candidate_path or reference_path in candidate_path.parents or candidate_path in reference_path.parents:
        raise AssertionError('Trace roots must be distinct and non-nested')
    paths = [reference_path, candidate_path]
    manifests = [json.loads((path / 'manifest.json').read_text()) for path in paths]
    reference, candidate = manifests
    validate_pair(reference, candidate)
    records, bulk, final_logits, next_logits = [], [], [], []
    for role, info in reference['initial'].items():
        artifact = reference_path / 'initial' / (role + '-extras.npz')
        if digest(artifact) != info['extra_artifact_sha256']:
            raise AssertionError('Initial extras artifact changed')
    # Deleting a file shared by a non-bulk record is never allowed.
    path_counts = Counter(info.get('path') for info in reference['arrays'].values())
    for name, oracle_info in reference['arrays'].items():
        candidate_info = candidate['arrays'][name]
        if not all(isinstance(info.get('sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', info['sha256'])
                   for info in (oracle_info, candidate_info)):
            raise AssertionError('Invalid array SHA256 metadata')
        for key in ('shape', 'dtype', 'exact'):
            if candidate_info[key] != oracle_info[key]:
                raise AssertionError('Array metadata differs: ' + name)
        if not oracle_info.get('retained') or 'path' not in oracle_info:
            raise AssertionError('Oracle observation was not retained')
        a = read_reference(oracle_info)
        row = candidate['comparison'][name]
        validate_comparison(row, oracle_info)
        is_bulk = BULK.match(name) is not None
        if is_bulk:
            if candidate_info.get('retained') or 'path' in candidate_info:
                raise AssertionError('Unexpected candidate bulk retention')
            relative = oracle_info['path']
            file_path = reference_path / relative
            if not ARRAY_PATH.fullmatch(relative) or path_counts[relative] != 1:
                raise AssertionError('Unsafe or shared bulk path')
            if file_path.is_symlink() or file_path.parent.is_symlink() or file_path.resolve().parent != reference_path / 'arrays':
                raise AssertionError('Bulk path redirects outside owned arrays')
            storage = dict(storage='npy', bytes=file_path.stat().st_size) if file_path.exists() else None
            if storage is None:
                archived = read_reference.records[relative]
                storage = dict(storage='lossless_zstd', bytes=archived['original_bytes'],
                               archive_path=archived['archive_path'], archive_bytes=archived['archive_bytes'],
                               raw_sha256=archived['raw_sha256'])
            bulk.append(dict(key=name, path=relative, sha256=oracle_info['sha256'],
                             shape=oracle_info['shape'], dtype=oracle_info['dtype'], **storage))
        else:
            if not candidate_info.get('retained') or 'path' not in candidate_info:
                raise AssertionError('Missing retained candidate observation')
            b = read_candidate(candidate_info)
            actual = compare_arrays(a, b, exact=oracle_info['exact'])
            compare_recorded(actual, row)
            if '/final_eval/' in name and '/next_logits/' in name:
                next_logits.append(logit_metrics(a, b))
            elif '/final_eval/' in name and '/logits/' in name and a.ndim == 3:
                final_logits.append(logit_metrics(a, b))
            if oracle_info['exact'] and not np.array_equal(a, b):
                raise AssertionError('Controlled discrete input differs')
        records.append((name, row))
    steps = reference['protocol']['steps']
    if reference['case'].get('model_kind') != 'sequence_classification' and (len(final_logits) != 10 or len(next_logits) != 10):
        raise AssertionError('Incomplete causal final-logit observations')
    categories = {}
    for category in ('gradient_sample', 'gradient_norm', 'updated_parameter_sample',
                     'final_gradient', 'final_parameter', 'embedding_gradient', 'loss', 'final_eval'):
        categories[category] = aggregate(row for name, row in records if '/%s/' % category in name)
    result = dict(status='audited', algorithm=reference['protocol']['algorithm'], device=reference['device'],
                  optimizer_steps=steps, numerical_passed=candidate['numerical_passed'],
                  reference_manifest_sha256=digest(reference_path / 'manifest.json'),
                  candidate_manifest_sha256=digest(candidate_path / 'manifest.json'),
                  reference_path=str(reference_path), candidate_path=str(candidate_path),
                  observations=len(records), trained_tensors=len(reference['trainable_parameters']),
                  categories=categories, all_observations=aggregate(row for _, row in records),
                  next_logits=aggregate_logits(next_logits) if next_logits else None,
                  logits=aggregate_logits(final_logits) if final_logits else None,
                  final_logits_only=aggregate(row for name, row in records
                                              if '/final_eval/' in name and '/logits/' in name),
                  verified_oracle_bulk=bulk, bulk_bytes=sum(row['bytes'] for row in bulk),
                  unmeasured_categories=['scalar_training_loss']
                  if reference['protocol']['algorithm'] in ('ppo', 'rloo') else [],
                  parameter_comparison_scope='Absolute final parameter values; no full update-delta parity measurement',
                  scope='Limited intermediate gradient samples, complete final-step gradients and parameters; no registry admission',
                  bulk_verification='Oracle bytes/hash checked; candidate bulk SHA256 format and comparison metadata validated. Candidate bulk bytes were not retained, so their hashes and comparisons cannot be independently recomputed')
    # Protect against edits while reading a potentially large trace.
    for path, manifest in zip(paths, manifests):
        if json.loads((path / 'manifest.json').read_text()) != manifest:
            raise AssertionError('Manifest changed during audit')
    return result


def prune_verified_bulk(result, receipt_path):
    """Remove only this audited pair's oracle final arrays, with immutable receipts."""
    receipt_path = Path(receipt_path).resolve()
    completion_path = receipt_path.with_name(receipt_path.name + '.completed.json')
    root = Path(result['reference_path']).resolve()
    candidate = Path(result['candidate_path']).resolve()
    if result['status'] != 'audited' or not result['verified_oracle_bulk']:
        raise AssertionError('Require a nonempty completed audit')
    if receipt_path == root or receipt_path == candidate or receipt_path.parent in (root / 'arrays', candidate / 'arrays'):
        raise AssertionError('Receipt cannot be an array payload')
    if receipt_path.exists() or completion_path.exists():
        raise FileExistsError('Never overwrite a retention receipt')
    if digest(root / 'manifest.json') != result['reference_manifest_sha256'] or digest(candidate / 'manifest.json') != result['candidate_manifest_sha256']:
        raise AssertionError('Manifest changed after audit')
    original = json.loads((root / 'manifest.json').read_text())
    paths = []
    for row in result['verified_oracle_bulk']:
        if row.get('storage', 'npy') != 'npy':
            raise AssertionError('Cannot prune archived payloads')
        if not BULK.fullmatch(row['key']) or not ARRAY_PATH.fullmatch(row['path']):
            raise AssertionError('Only final gradient/parameter arrays may be pruned')
        info = original['arrays'][row['key']]
        if any(info[key] != row[key] for key in ('path', 'sha256', 'shape', 'dtype')):
            raise AssertionError('Prune plan differs from original manifest')
        path = root / row['path']
        if path.is_symlink() or path.parent.is_symlink() or path.resolve().parent != root / 'arrays':
            raise AssertionError('Unsafe prune path')
        checked_array(root, info)  # Rehash immediately before authorizing removal.
        if path.stat().st_size != row['bytes']:
            raise AssertionError('Payload size changed after audit')
        paths.append(path)
    plan = dict(action='prune_only_verified_oracle_final_payloads', audit=result,
                consequence='Full oracle final payloads will no longer be independently re-readable; hashes/statistics and all small observations remain')
    write_exclusive(receipt_path, plan)
    for path in paths:
        path.unlink()
    completion = dict(action='prune_completed', receipt_sha256=digest(receipt_path),
                      removed_files=len(paths), removed_bytes=result['bulk_bytes'],
                      reference_manifest_sha256=digest(root / 'manifest.json'),
                      candidate_manifest_sha256=digest(candidate / 'manifest.json'))
    write_exclusive(completion_path, completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prune-oracle-bulk', action='store_true')
    parser.add_argument('--retention-receipt', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Never overwrite an audit')
    if args.prune_oracle_bulk != (args.retention_receipt is not None):
        parser.error('Pruning requires an explicit new --retention-receipt; omit both for read-only audit')
    result = audit(args.reference, args.candidate)
    write_exclusive(args.output, result)
    if args.prune_oracle_bulk:
        prune_verified_bulk(result, args.retention_receipt)


if __name__ == '__main__':
    main()
