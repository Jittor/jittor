"""Reject misleading high-precision controls and corrupt diagnostic artifacts."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('rounding_compare_test', scripts / 'trl_rounding_compare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_run(path, precision='fp32', values=None, **changes):
    from trl_rl_probe import array_hash
    path.mkdir()
    values = np.asarray([1., 2.] if values is None else values,
                        dtype=np.float64 if precision == 'fp64' else np.float32)
    ids = np.array([3, 7], dtype=np.int64)
    arrays = {}
    for index, (name, array) in enumerate([('logits', values), ('input_ids', ids)]):
        filename = str(index) + '.npy'
        np.save(path / filename, array)
        arrays[name] = dict(path=filename, sha256=array_hash(array), shape=list(array.shape), dtype=str(array.dtype))
    manifest = dict(status='complete', fallback_count=0, precision=precision,
                    initial={'weight': 'same'}, batch_manifest_sha256='same', versions={}, parameter_count=2,
                    arrays=arrays)
    manifest.update(changes)
    (path / 'manifest.json').write_text(json.dumps(manifest))
    return path


def test_precision_control_can_compare_different_float_dtypes(probe, tmp_path):
    a = make_run(tmp_path / 'a', 'fp64')
    b = make_run(tmp_path / 'b')
    assert probe.compare(a, b)['fields']['logits']['max_abs'] == 0


@pytest.mark.parametrize('changes', [dict(status='running'), dict(fallback_count=1),
                                     dict(initial={}), dict(batch_manifest_sha256='different'),
                                     dict(algorithm='grpo')])
def test_nonmatching_or_incomplete_control_rejected(probe, tmp_path, changes):
    a = make_run(tmp_path / 'a')
    b = make_run(tmp_path / 'b', **changes)
    with pytest.raises(AssertionError):
        probe.compare(a, b)


def test_corrupted_payload_rejected(probe, tmp_path):
    a = make_run(tmp_path / 'a')
    b = make_run(tmp_path / 'b')
    np.save(b / '0.npy', np.array([2., 3.], dtype=np.float32))
    with pytest.raises(AssertionError, match='hash mismatch'):
        probe.compare(a, b)


def test_claimed_fp64_with_fp32_payload_rejected(probe, tmp_path):
    a = make_run(tmp_path / 'a')
    b = make_run(tmp_path / 'b')
    manifest = json.loads((b / 'manifest.json').read_text())
    manifest['precision'] = 'fp64'
    (b / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match='narrowed'):
        probe.compare(a, b)
