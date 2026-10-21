"""Explicit protocol dispatch; no registry row may use this validation path."""

import importlib
import json
from pathlib import Path


VALIDATORS = {
    ('accelerate', 'accelerate-checkpoint'): '_accelerate_scorecard',
    ('accelerate', 'accelerate-fsdp2-full'): '_accelerate_scorecard',
    ('accelerate', 'accelerate-checkpoint-fresh-process'): '_accelerate_scorecard',
    ('accelerate', 'accelerate-fsdp2-full-fresh-process'): '_accelerate_scorecard',
    ('deepspeed', 'deepspeed-zero-training'): '_deepspeed_scorecard',
    ('flash-attention', 'flash-attention-training'): '_flash_attention_scorecard',
}


def validator(record, source):
    if (record.get('record_kind') != 'protocol' or record.get('repo') != source.get('repo')
            or record.get('registry_refs') != [] or source.get('axes') != []):
        raise ValueError('Protocol requires matching repo and explicit empty registry axes/references')
    spec = record.get('protocol')
    if not isinstance(spec, dict):
        raise ValueError('Protocol descriptor must be an object')
    module = VALIDATORS.get((record['repo'], spec.get('id')))
    if module is None or not Path(__file__).with_name(module + '.py').is_file():
        raise ValueError('No maintained validator for this repo/protocol scope')
    return importlib.import_module(module)


def validate_spec(record, source):
    return validator(record, source).validate_spec(record, source)


def validate_measurements(record, artifacts, source, *, root=None):
    module = validator(record, source)
    if module.__name__ == '_flash_attention_scorecard':
        return module.validate_measurements(record, artifacts, source, root=root)
    return module.validate_measurements(record, artifacts, source)


def pairing_key(record):
    # Device/provider/allocation belong to actual measurement metadata. Every
    # semantic dimension (world, precision, ZeRO stage, model, accumulation,
    # perturbation formula) must match, without a hand-picked smaller key.
    return record['repo'], json.dumps(record['protocol'], sort_keys=True, allow_nan=False)
