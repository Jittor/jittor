"""Adversarial controls for the full-Qwen method/configuration trace auditor."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def tools(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    scripts = root/'agent/skills/torch-compat-repo-runbook/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    import trl_lora_probe as probe
    import trl_lora_compare as auditor
    spec = importlib.util.spec_from_file_location('practical_fixtures', Path(__file__).with_name('test_trl_practical_probe.py'))
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    return probe, auditor, fixtures


def make(tools, path, runtime, method='sft', resume=False):
    probe, _, fixtures = tools
    fixtures.make_run(probe, path, runtime, resume=resume)
    file = path/'manifest.json'
    manifest = json.loads(file.read_text())
    protocol = manifest['protocol']
    protocol.update(algorithm=method, gradient_accumulation=1,
                    records=[{'text': text} for text in protocol.pop('texts')])
    manifest['micro_steps'] = {str(step): 1 for step in manifest['optimizer_steps']}
    manifest['rng_before_loss'] = {step+'/0': values for step, values in manifest['rng_before_loss'].items()}
    arrays = manifest['arrays']
    for step in manifest['optimizer_steps']:
        prefix = 'step/%d/' % step
        arrays[prefix+'micro/0/loss'] = arrays.pop(prefix+'loss')
        example = None
        for key in list(arrays):
            if key.startswith(prefix+'input/'):
                example = arrays.pop(key)
                arrays[key.replace(prefix, prefix+'micro/0/')] = example
        from trl_lora_compare import INPUTS
        for name in INPUTS[method]:
            arrays.setdefault(prefix+'micro/0/input/'+name, example)
        if method == 'grpo':
            arrays[prefix+'generation/0/output_ids'] = example
    manifest['generation_calls'] = ([dict(before_global_step=step-1) for step in manifest['optimizer_steps']]
                                    if method == 'grpo' else [])
    file.write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize('method', ['sft', 'dpo', 'grpo'])
def test_method_specific_complete_pair(tools, tmp_path, method):
    a = make(tools, tmp_path/'oracle', 'torch', method)
    b = make(tools, tmp_path/'candidate', 'jittor', method)
    assert tools[1].compare(a, b)['strict_passed']


def test_equally_missing_adapter_gradient_is_rejected(tools, tmp_path):
    a = make(tools, tmp_path/'a', 'torch')
    b = make(tools, tmp_path/'b', 'jittor')
    for path in (a, b):
        file = path/'manifest.json'
        manifest = json.loads(file.read_text())
        del manifest['arrays'][next(key for key in manifest['arrays'] if '/gradient/' in key)]
        file.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match='Missing adapter'):
        tools[1].compare(a, b)


def test_claimed_accumulation_requires_actual_microbatches(tools, tmp_path):
    path = make(tools, tmp_path/'a', 'torch')
    manifest = json.loads((path/'manifest.json').read_text())
    manifest['protocol']['gradient_accumulation'] = 2
    with pytest.raises(AssertionError, match='microbatch'):
        tools[1].validate(manifest)


def test_grpo_cannot_omit_both_rollout_inventories(tools, tmp_path):
    path = make(tools, tmp_path/'a', 'torch', 'grpo')
    manifest = json.loads((path/'manifest.json').read_text())
    manifest['generation_calls'] = []
    with pytest.raises(AssertionError, match='generation inventory'):
        tools[1].validate(manifest)


def test_partial_vocabulary_is_not_a_final_logits_pass(tools, tmp_path):
    path = make(tools, tmp_path/'a', 'torch')
    manifest = json.loads((path/'manifest.json').read_text())
    manifest['arrays']['final/0/next_logits']['shape'] = [10]
    with pytest.raises(AssertionError, match='vocabulary'):
        tools[1].validate(manifest)


def test_restored_state_requires_a_distinct_bound_checkpoint(tools, tmp_path):
    a = make(tools, tmp_path/'a', 'torch')
    b = make(tools, tmp_path/'b', 'torch', resume=True)
    assert tools[1].compare(a, b, resume=True)['strict_passed']
    (a/'trainer/checkpoint-5/optimizer.pt').write_text('changed bytes')
    with pytest.raises(AssertionError, match='checkpoint-five file hash'):
        tools[1].compare(a, b, resume=True)


def test_artifact_guard_runs_before_creating_output(tools, tmp_path):
    from types import SimpleNamespace
    root = tmp_path/'checkout'
    root.mkdir()
    args = SimpleNamespace(root=root, model=tmp_path/'model', upstream=tmp_path/'upstream',
        initial_buffers=tmp_path/'buffers', initial_adapters=tmp_path/'adapters',
        output=root/'large-arrays', resume=None, reference=None, runtime='torch',
        algorithm='sft', device='cpu', precision='fp32')
    with pytest.raises(ValueError, match='outside'):
        tools[0].validate_run_arguments(args)
    assert not args.output.exists()


def test_lora_grpo_requires_independent_rollout_provenance(tools, tmp_path):
    from types import SimpleNamespace
    args = SimpleNamespace(root=tmp_path/'checkout', model=tmp_path/'model', upstream=tmp_path/'upstream',
        initial_buffers=tmp_path/'buffers', initial_adapters=tmp_path/'adapters',
        output=tmp_path/'output', resume=None, reference=None, runtime='jittor',
        algorithm='grpo', device='cuda', precision='fp32')
    with pytest.raises(ValueError, match='rollout provenance'):
        tools[0].validate_run_arguments(args)


def test_unequal_accumulation_must_be_observed_not_just_requested(tools, tmp_path):
    path = make(tools, tmp_path/'a', 'torch')
    manifest = json.loads((path/'manifest.json').read_text())
    manifest['protocol'].update(max_length=128, gradient_accumulation=2)
    for step in manifest['optimizer_steps']:
        manifest['micro_steps'][str(step)] = 2
        for key, value in list(manifest['arrays'].items()):
            if key.startswith('step/%d/micro/0/' % step):
                manifest['arrays'][key.replace('/micro/0/', '/micro/1/')] = value
        manifest['rng_before_loss'][str(step)+'/1'] = manifest['rng_before_loss'][str(step)+'/0']
    with pytest.raises(AssertionError, match='unequal-length'):
        tools[1].validate(manifest)
