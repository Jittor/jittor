"""Negative evidence-binding contracts; no tensor runtime or GPU is needed."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SOURCE = 'a' * 40
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def evidence_module():
    spec = importlib.util.spec_from_file_location(
        'multinode_evidence',
        ROOT / 'agent/skills/jittor-distributed-verification/recheck_native_fsdp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(tmp_path, files):
    manifest = tmp_path / 'bindings.json'
    manifest.write_text(json.dumps({'source_sha': SOURCE, 'source_status': '', 'files': files}))
    return manifest


def test_changed_measurement_and_unbound_files_refuse(tmp_path, evidence_module):
    raw = tmp_path / 'raw.json'
    raw.write_text('{"rank": 0}')
    manifest = _manifest(tmp_path, {'raw.json': {
        'bytes': raw.stat().st_size, 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest()}})
    bound = evidence_module.BoundFiles(manifest, SOURCE)
    assert bound.json(raw) == {'rank': 0}
    other = tmp_path / 'unbound.json'
    other.write_text('{"rank": 0}')
    with pytest.raises(ValueError, match='no raw binding'):
        bound.require(other)
    raw.write_text('{"rank": 1}')
    with pytest.raises(ValueError, match='raw binding changed'):
        evidence_module.BoundFiles(manifest, SOURCE)


def test_different_framework_source_refuses(tmp_path, evidence_module):
    manifest = _manifest(tmp_path, {})
    with pytest.raises(ValueError, match='clean measurement source'):
        evidence_module.BoundFiles(manifest, 'b' * 40)


def test_binding_cannot_read_outside_artifact_root(tmp_path, evidence_module):
    manifest = _manifest(tmp_path, {'../outside': {'bytes': 0, 'sha256': '0' * 64}})
    with pytest.raises(ValueError, match='escapes'):
        evidence_module.BoundFiles(manifest, SOURCE)


def test_actual_dtype_evidence_cannot_be_replaced_with_policy_label(evidence_module):
    metadata = {'mode': 'parity', 'kind': 'adamw', 'dtype': 'bfloat16', 'rank': 0,
                'world': 2, 'steps': 5, 'shim_loaded': False, 'fallback_delta': 0,
                'actual_device': 'cuda:0', 'float32_precision': ['highest', 'highest'],
                'collectives': {'all_gather': 20, 'reduce_scatter': 20},
                'local_elements': 35, 'global_elements': 66,
                'compute_dtypes': [{'input': 'float32', 'weight': 'float32'}] * 5,
                'collective_dtypes': {'all_gather': ['bfloat16'] * 20,
                                     'reduce_scatter': ['float32'] * 20}}
    with pytest.raises(ValueError, match='actual sharding/compute/reduction'):
        evidence_module._native_metadata(metadata, mode='parity', kind='adamw', dtype='bfloat16', rank=0)
