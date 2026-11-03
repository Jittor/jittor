"""Offline guardrails for full-checkpoint TRL trace scope and prerequisites."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('full_probe_test', scripts / 'trl_full_probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(tmp_path):
    return ['probe', '--root', str(tmp_path / 'checkout'), '--upstream', str(tmp_path / 'trl'),
            '--output', str(tmp_path / 'result'), '--model', str(tmp_path / 'model'),
            '--initial-buffers', str(tmp_path / 'buffers.npz'),
            '--algorithm', 'orpo', '--runtime', 'torch', '--device', 'cpu']


def test_valid_oracle_defaults_to_ten_updates(probe, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', arguments(tmp_path))
    args = probe.options()
    assert args.steps == 10 and args.reference is None
    assert args.model.is_absolute() and args.initial_buffers.is_absolute()


@pytest.mark.parametrize('extra', [ ['--runtime', 'jittor'], ['--reference', 'unexpected'],
                                    ['--steps', '0'], ['--steps', '11'], ['--threads', '0'],
                                    ['--algorithm', 'sft']])
def test_inconsistent_or_unsupported_protocol_rejected(probe, monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, 'argv', arguments(tmp_path) + extra)
    with pytest.raises(SystemExit):
        probe.options()


def test_output_inside_source_rejected(probe, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', arguments(tmp_path) + ['--output', str(tmp_path / 'checkout/result')])
    with pytest.raises(SystemExit):
        probe.options()


def test_intermediate_sample_cannot_be_classified_as_full_bulk(probe):
    assert probe.observation_is_bulk('final_gradient/model.norm.weight')
    assert probe.observation_is_bulk('final_parameter/model.norm.weight')
    assert not probe.observation_is_bulk('gradient_sample/model.norm.weight')
    assert not probe.observation_is_bulk('updated_parameter_sample/model.norm.weight')
    values = np.arange(16384).reshape(128, 128)
    np.testing.assert_array_equal(probe.sampled(values), [0, 4096, 8192, 12288])


def test_full_cases_use_ten_distinct_real_prompts_with_both_kto_classes(probe):
    import trl_full_cases as cases
    pairs = cases.offline_records('orpo')
    assert len(pairs) == len({row['prompt'] for row in pairs}) == 10
    assert all(row['chosen'] != row['rejected'] for row in pairs)
    kto = cases.offline_records('kto')
    assert sum(row['label'] for row in kto) == 5
    assert all(kto[i]['label'] != kto[i + 1]['label'] for i in range(0, 10, 2))
    assert [row['prompt'] for row in cases.online_records()] == [row['prompt'] for row in pairs]


def test_config_protocol_survives_json_reference_reload(probe):
    """Protect the real config expression against integer id2label key drift."""
    import ast
    import json
    from types import SimpleNamespace

    class Configuration:
        def to_dict(self):
            return {'id2label': {0: 'LABEL_0', 1: 'LABEL_1'}, 'hidden_size': 1024}

        def to_json_string(self, use_diff=True):
            assert not use_diff
            return json.dumps(self.to_dict())

    class AutoConfig:
        @staticmethod
        def from_pretrained(path, local_files_only):
            assert local_files_only
            return Configuration()

    tree = ast.parse(Path(probe.__file__).read_text())
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'config_dict' for target in node.targets))
    expression = ast.Expression(assignment.value)
    config = eval(compile(expression, '<protocol-config>', 'eval'),
                  dict(json=json, AutoConfig=AutoConfig, args=SimpleNamespace(model='local-checkpoint')))
    assert config == json.loads(json.dumps(config))
    assert config['id2label'] == {'0': 'LABEL_0', '1': 'LABEL_1'}
