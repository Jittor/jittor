"""Bound complete public exports and refuse unsafe historical migrations."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'


def load(name):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_backend_exports_and_helpers_are_counted():
    names, rows = load('_diffusers_registry').model_exports('''
_import_structure = {}
if is_torch_available():
    _import_structure['unet'] = ['UNet2DModel', 'HelperEnum']
    _import_structure['unet'].extend(['helper_function'])
if is_flax_available():
    _import_structure['flax'] = ['FlaxUNet']
''')
    assert names == ['FlaxUNet', 'HelperEnum', 'UNet2DModel', 'helper_function']
    assert len(rows) == 3 and all(row['guards'] for row in rows)


def test_same_public_alias_in_alternative_branches_counts_once_with_both_origins():
    names, rows = load('_diffusers_registry').model_exports('''
_import_structure = {}
if feature:
    _import_structure['shared'] = ['Alias']
else:
    _import_structure['shared'] = ['Alias']
''')
    assert names == ['Alias'] and len(rows) == 2
    assert rows[0]['guards'] != rows[1]['guards']


@pytest.mark.parametrize('body', [
    "_import_structure['m'] = ['A', 'A']",
    "_import_structure['m'] = ['A']\n_import_structure['m'] = ['B']",
    "_import_structure['m'] = ['A']\n_import_structure['m'].extend(['A'])",
    "_import_structure['m'] = ['A']\n_import_structure['other'] = ['A']",
    "_import_structure[key] = ['A']",
    "_import_structure['m'] = dynamic_names()",
    "_import_structure.update({'m': ['A']})",
    "alias = _import_structure",
    "_import_structure['m'] = alias = ['A']",
    "alias = _import_structure['m']",
    "register(_import_structure)",
    "register(exports=_import_structure)",
    "register(_import_structure['m'])",
    "for name in names:\n    _import_structure[name] = ['A']",
    "def reset():\n    _import_structure = {}",
])
def test_dynamic_or_ambiguous_registry_never_silently_shrinks(body):
    with pytest.raises(ValueError):
        load('_diffusers_registry').model_exports('_import_structure = {}\n' + body)


def scope():
    source = {'repo': 'diffusers', 'upstream': {'commit': 'b' * 40, 'version': '0.35.1'},
              'axes': [{'axis': 'model_api', 'members': ['UNet2DModel', 'FlaxUNet', 'Helper']}]}
    record = {'repo': 'diffusers', 'case': 'diffusers_unet2d', 'status': 'passed',
              'level': None, 'component_evidence_level': 3, 'registry_refs': [],
              'jittor_commit': 'a' * 40, 'upstream': copy.deepcopy(source['upstream']),
              'inputs': {'seed': 0, 'perturb_seed': 7},
              'numerics': {'atol': .002, 'rtol': .01,
                           'parameter_gradients': {'requested': 144, 'checked': 144},
                           'input_gradients': {'requested': 1, 'checked': 1}},
              'training': {'steps': 3, 'learning_rate': .001, 'trainable': 144, 'frozen': 0,
                           'trainable_parameters': ['parameter%d' % i for i in range(144)],
                           'frozen_parameters': []}}
    return source, record


@pytest.mark.parametrize('failure', ['baseline', 'upstream', 'pipeline', 'member', 'gradient', 'threshold', 'steps'])
def test_migration_refuses_stale_source_fake_pipeline_and_weakened_protocol(failure):
    source, record = scope()
    tool = load('collect_diffusers_scorecard')
    assert len(tool.validate_scope(record, source, 'a' * 40)) == 144
    if failure == 'baseline':
        record['jittor_commit'] = 'c' * 40
    elif failure == 'upstream':
        record['upstream']['version'] = 'different'
    elif failure == 'pipeline':
        record['registry_refs'] = [{'repo': 'diffusers', 'axis': 'pipeline_class', 'member': 'DDPMPipeline'}]
    elif failure == 'member':
        source['axes'][0]['members'].remove('UNet2DModel')
    elif failure == 'gradient':
        record['numerics']['parameter_gradients']['checked'] = 143
    elif failure == 'threshold':
        record['numerics']['rtol'] = .1
    else:
        record['training']['steps'] = 1
    with pytest.raises(ValueError):
        tool.validate_scope(record, source, 'a' * 40)


def test_original_hashes_are_verified_before_migration(tmp_path):
    path = tmp_path / 'original.json'
    path.write_text(json.dumps({'measured': True}))
    tool = load('collect_diffusers_scorecard')
    record = {'evidence': {'artifacts': [{'role': 'reference', 'path': str(path),
                                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]}}
    assert tool.original_artifacts(record)['reference'] == path
    path.write_text(json.dumps({'measured': False}))
    with pytest.raises(ValueError, match='Changed/missing original artifact'):
        tool.original_artifacts(record)


def test_archived_training_program_must_match_original_hash(monkeypatch):
    tool = load('collect_diffusers_scorecard')
    program = b'print("original program")\n'
    builder = b'''def _diffusers_unet(torch):
    from diffusers import UNet2DModel
    return UNet2DModel(), {}
CASES = {'diffusers_unet2d': (_diffusers_unet, ('diffusers',))}
'''
    monkeypatch.setattr(tool.subprocess, 'check_output',
                        lambda cmd: builder if cmd[-1].endswith('_ecosystem_cases.py') else program)
    proof = tool.measured_source(ROOT, 'a' * 40, hashlib.sha256(program).hexdigest())
    assert proof['builder_file_sha256'] == hashlib.sha256(builder).hexdigest()
    with pytest.raises(ValueError, match='program hash'):
        tool.measured_source(ROOT, 'a' * 40, '0' * 64)
