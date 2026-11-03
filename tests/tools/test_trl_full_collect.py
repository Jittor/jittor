"""Negative controls for bounded full-model trace auditing and explicit pruning."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def collector(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('full_collect_test', scripts / 'trl_full_collect.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'CAUSAL_PARAMETERS', 2)
    return module


@pytest.fixture
def pair(collector, tmp_path):
    from trl_rl_probe import array_hash, compare_arrays
    from trl_sft_probe import digest
    left, right = tmp_path / 'oracle', tmp_path / 'candidate'
    for root in (left, right):
        (root / 'arrays').mkdir(parents=True)
        (root / 'initial').mkdir()
    np.savez(left / 'initial/policy-extras.npz')
    common = dict(status='complete', device='cpu', fallback_count=0,
                  protocol=dict(algorithm='orpo', steps=1), versions={}, trl_source={},
                  tool_sources={'trl_full_probe.py': 'fixed'},
                  initial={'policy': dict(extra_artifact_sha256=digest(left / 'initial/policy-extras.npz'),
                                          parameter_count=2, tensors={'parameter/weight': dict(shape=[2], dtype='float32', sha256='fixed')})},
                  trainable_parameters={'weight': [2]}, frozen_parameters=0,
                  case=dict(expected_optimizer_steps=1), optimizer_steps=1,
                  gradient_norms=[1.], parameter_updates=[dict(changed_elements=2)], generation_calls=[])
    oracle = dict(common, runtime='torch', numerical_passed=None, arrays={}, comparison={})
    candidate = dict(common, runtime='jittor', numerical_passed=True, arrays={}, comparison={})
    observations = {
        'step/0/loss/0': np.array(1., np.float32),
        'step/0/embedding_gradient/policy/0/0': np.array([1.], np.float32),
        'step/0/forward/input/input_ids/0': np.array([[1]], np.int64),
        'step/0/forward/output/0/0': np.array([[[1., 2.]]], np.float32),
        'step/0/gradient_sample/weight/0': np.array([1.], np.float32),
        'step/0/gradient_norm/weight/0': np.array(1., np.float64),
        'step/0/updated_parameter_sample/weight/0': np.array([2.], np.float32),
        'step/0/final_gradient/weight/0': np.array([1., 2.], np.float32),
        'step/0/final_parameter/weight/0': np.array([2., 3.], np.float32),
    }
    for i in range(10):
        observations['step/1/final_eval/%d/logits/0' % i] = np.array([[[1., 2.]]], np.float32)
        observations['step/1/final_eval/%d/next_logits/0' % i] = np.array([1., 2.], np.float32)
        observations['step/1/final_eval/%d/input_ids/0' % i] = np.array([[i]], np.int64)
    for key, array in observations.items():
        path = 'arrays/' + hashlib.sha256(key.encode()).hexdigest() + '.npy'
        info = dict(shape=list(array.shape), dtype=str(array.dtype), sha256=array_hash(array), exact=array.dtype.kind == 'i')
        oracle['arrays'][key] = dict(info, path=path, retained=True)
        np.save(left / path, array)
        if collector.BULK.fullmatch(key):
            candidate['arrays'][key] = dict(info, retained=False)
        else:
            candidate['arrays'][key] = dict(info, path=path, retained=True)
            np.save(right / path, array)
        candidate['comparison'][key] = compare_arrays(array, array, exact=info['exact'])
    for root, manifest in ((left, oracle), (right, candidate)):
        (root / 'manifest.json').write_text(json.dumps(manifest))
    return left, right


def alter(root, function):
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    function(manifest)
    path.write_text(json.dumps(manifest))


def test_default_audit_is_readonly_and_includes_logits(collector, pair):
    left, right = pair
    before = set(left.rglob('*'))
    result = collector.audit(left, right)
    assert result['numerical_passed'] and result['next_logits']['top1_matches'] == 10
    assert len(result['verified_oracle_bulk']) == 2
    assert set(left.rglob('*')) == before


def test_pruning_keeps_original_manifests_and_small_payloads(collector, pair, tmp_path):
    left, right = pair
    original = (left / 'manifest.json').read_bytes()
    result = collector.audit(left, right)
    receipt = tmp_path / 'retention.json'
    completion = collector.prune_verified_bulk(result, receipt)
    assert completion['removed_files'] == 2
    assert (left / 'manifest.json').read_bytes() == original
    assert len(list((left / 'arrays').iterdir())) == 37
    assert receipt.exists() and receipt.with_name(receipt.name + '.completed.json').exists()
    with pytest.raises(FileExistsError):
        collector.prune_verified_bulk(result, receipt)
    # A pruned trace cannot silently pass a new complete payload audit.
    with pytest.raises(FileNotFoundError):
        collector.audit(left, right)


@pytest.mark.parametrize('mutation', [
    lambda m: m.update(status='running'),
    lambda m: m.update(fallback_count=1),
    lambda m: m.update(optimizer_steps=0),
    lambda m: m['protocol'].update(algorithm='kto'),
    lambda m: m['trainable_parameters'].clear(),
    lambda m: m['comparison'].pop('step/0/final_gradient/weight/0'),
])
def test_incomplete_or_mismatched_pairs_rejected(collector, pair, mutation):
    left, right = pair
    alter(right, mutation)
    with pytest.raises(AssertionError):
        collector.audit(left, right)


def test_corrupted_oracle_bulk_rejected_before_pruning(collector, pair):
    left, right = pair
    manifest = json.loads((left / 'manifest.json').read_text())
    path = left / manifest['arrays']['step/0/final_gradient/weight/0']['path']
    np.save(path, np.array([3., 4.], np.float32))
    with pytest.raises(AssertionError, match='hash mismatch'):
        collector.audit(left, right)


def test_changed_bulk_between_audit_and_prune_rejected(collector, pair, tmp_path):
    left, right = pair
    result = collector.audit(left, right)
    np.save(left / result['verified_oracle_bulk'][0]['path'], np.array([3., 4.], np.float32))
    receipt = tmp_path / 'receipt.json'
    with pytest.raises(AssertionError, match='hash mismatch'):
        collector.prune_verified_bulk(result, receipt)
    assert not receipt.exists()


def test_nonbulk_prune_plan_rejected(collector, pair, tmp_path):
    result = collector.audit(*pair)
    result['verified_oracle_bulk'][0]['key'] = 'step/0/gradient_sample/weight/0'
    receipt = tmp_path / 'receipt.json'
    with pytest.raises(AssertionError, match='Only final'):
        collector.prune_verified_bulk(result, receipt)
    assert not receipt.exists()


def test_all_initial_policy_parameters_must_be_trained(collector, pair):
    left, right = pair
    for root in pair:
        alter(root, lambda m: m['trainable_parameters'].update(unexpected=[2]))
    with pytest.raises(AssertionError, match='every full policy'):
        collector.audit(left, right)


def test_recorded_retained_comparison_is_recomputed(collector, pair):
    left, right = pair
    def change(manifest):
        manifest['comparison']['step/1/final_eval/0/next_logits/0']['max_abs'] = .01
    alter(right, change)
    with pytest.raises(AssertionError, match='Recorded comparison differs'):
        collector.audit(left, right)


def test_lossless_archive_audit_preserves_full_coverage(collector, pair):
    import shutil
    from trl_trace_archive import archive_trace
    if shutil.which('zstd') is None:
        pytest.skip('Optional zstd executable unavailable')
    left, right = pair
    before = collector.audit(left, right)
    archive_trace(left, archive=True)
    after = collector.audit(left, right)
    assert after['categories'] == before['categories']
    assert after['final_logits_only']['elements'] == 20
    assert after['bulk_bytes'] == before['bulk_bytes']
    assert all(row['storage'] == 'lossless_zstd' for row in after['verified_oracle_bulk'])
    assert set(after['audit_sources']) == {'trl_full_collect.py', 'trl_trace_archive.py'}
    with pytest.raises(AssertionError, match='Cannot prune archived'):
        collector.prune_verified_bulk(after, left.parent / 'must-not-exist.json')
    assert not (left.parent / 'must-not-exist.json').exists()


def test_corrupted_archive_rejected_by_audit(collector, pair):
    import shutil
    from trl_trace_archive import archive_trace
    if shutil.which('zstd') is None:
        pytest.skip('Optional zstd executable unavailable')
    left, right = pair
    archive_trace(left, archive=True)
    path = next((left / 'arrays').glob('*.zst'))
    path.write_bytes(b'corrupted compressed payload')
    with pytest.raises((AssertionError, RuntimeError)):
        collector.audit(left, right)


@pytest.mark.parametrize('key', ['step/0/loss/0', 'step/0/embedding_gradient/policy/0/0',
                                'step/0/forward/input/input_ids/0', 'step/0/forward/output/0/0',
                                'step/1/final_eval/0/input_ids/0'])
def test_required_observations_cannot_be_removed_from_both_sides(collector, pair, key):
    left, right = pair
    alter(left, lambda m: m['arrays'].pop(key))
    alter(right, lambda m: (m['arrays'].pop(key), m['comparison'].pop(key)))
    with pytest.raises(AssertionError, match='Missing'):
        collector.audit(left, right)


def test_candidate_bulk_hash_format_checked(collector, pair):
    left, right = pair
    alter(right, lambda m: m['arrays']['step/0/final_gradient/weight/0'].update(sha256='bad'))
    with pytest.raises(AssertionError, match='Invalid array SHA256'):
        collector.audit(left, right)


def test_rloo_loss_absence_is_explicit(collector, pair):
    left, right = pair
    key = 'step/0/loss/0'
    alter(left, lambda m: (m['protocol'].update(algorithm='rloo'), m['arrays'].pop(key)))
    alter(right, lambda m: (m['protocol'].update(algorithm='rloo'), m['arrays'].pop(key), m['comparison'].pop(key)))
    result = collector.audit(left, right)
    assert result['unmeasured_categories'] == ['scalar_training_loss']
    assert result['categories']['loss'] is None
