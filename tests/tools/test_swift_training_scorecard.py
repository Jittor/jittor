"""Fail closed on incomplete real Swift Trainer observations, without importing Swift."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'
sys.path.insert(0, str(SCRIPTS))
from _swift_training_contract import CASE, compare_pair, fields  # noqa: E402
from component_measure import installed_distribution  # noqa: E402
from swift_training_probe import UPSTREAM_CALLS  # noqa: E402


@pytest.fixture
def protocol(tmp_path):
    source = {'repo': 'ms-swift', 'upstream': {'version': '4.5.2', 'commit': 'a' * 40},
        'package_sources': [{'path': 'swift/__init__.py', 'sha256': 'b' * 64}],
        'axes': [{'axis': key, 'members': [member]} for key, member in
                 [('model_family', 'llama'), ('trainer', 'causal_lm'), ('task', 'causal_lm')]]}
    names = ['parameter%d' % i for i in range(12)]
    identity = {'model_family': 'llama', 'trainer': 'causal_lm', 'task': 'causal_lm',
        'trainer_class': 'swift.trainers.seq2seq_trainer.Seq2SeqTrainer',
        'trainer_registry_target': 'swift.trainers.Seq2SeqTrainer',
        'model_class': 'transformers.models.llama.modeling_llama.LlamaForCausalLM',
        'loader_class': 'swift.model.models.llama.LlamaLoader', 'template_type': 'llama',
        'trainable_parameters': names, 'frozen_parameters': [], 'buffer_names': [], 'optimizer_updates': 3}
    names.sort()
    inputs = {}
    for step in range(3):
        for name in ('attention_mask', 'input_ids', 'labels'):
            data = np.array([[1, 2, 3]], dtype='int64')
            inputs['step/%d/observed_input::%s' % (step, name)] = {
                'dtype': str(data.dtype), 'shape': list(data.shape), 'values': data.tolist(),
                'sha256': hashlib.sha256(data.tobytes()).hexdigest()}
    value = {'case': CASE, 'jittor_commit': 'c' * 40, 'upstream': source['upstream'],
        'steps': 3, 'seed': 0, 'learning_rate': .001, 'mode': 'train',
        'trainable_parameters': names, 'frozen_parameters': [], 'input_gradient_names': [],
        'integer_inputs': ['attention_mask', 'input_ids', 'labels'], 'observed_inputs': inputs,
        'observed_intermediate_gradients': ['embedding_output'],
        'downstream_distribution': {'repo': 'ms-swift', 'version': '4.5.2',
            'source_sha256': {'swift/__init__.py': 'b' * 64},
            'installed_files_sha256': {'__init__.py': 'b' * 64}, 'wheel_sha256': 'd' * 64},
        'registry_refs': [{'repo': 'ms-swift', 'axis': key, 'member': identity[key]}
                          for key in ('model_family', 'trainer', 'task')],
        'training_protocol': {'owner': 'swift.pipelines.train.sft.SwiftSft.main',
            'trainer_owner': 'swift.trainers.Seq2SeqTrainer.train', 'optimizer': 'torch.optim.SGD',
            'scheduler': 'constant', 'gradient_clipping': False, 'identity': identity,
            'upstream_calls': UPSTREAM_CALLS},
        'hf_hub_offline': True, 'device': 'cpu', 'tf32': {'matmul': False},
        'runtime_conditions': {}, 'dependencies': {}, 'model_artifacts': {'weights': 'e' * 64},
        'measurement_source_sha256': {'probe.py': 'f' * 64}}
    value['observed_tensor_devices'] = {name: 'cpu' for name in fields(value) + list(inputs)}
    arrays = {name: np.ones(2, dtype='float32') for name in fields(value)}
    for name in names:
        arrays['step/2/parameter::' + name] = np.full(2, .9, dtype='float32')
    paths = [tmp_path / 'torch.npz', tmp_path / 'jittor.npz']
    values = []
    for runtime in ('torch', 'jittor'):
        item = copy.deepcopy(value)
        item.update(runtime=runtime, interpreter={'is_shim': runtime == 'jittor', 'binary_sha256': '1' * 64},
                    fallback_policy='error' if runtime == 'jittor' else None, fallback_count=0)
        values.append(item)
    return source, paths, values, [copy.deepcopy(arrays), copy.deepcopy(arrays)]


def evaluate(protocol):
    source, paths, values, arrays = protocol
    for path, value, data in zip(paths, values, arrays):
        np.savez(path, **data)
        path.with_suffix('.json').write_text(json.dumps(value))
    return compare_pair(*paths, source, 'c' * 40, seed=0)


def test_complete_three_step_observation_passes(protocol):
    result = evaluate(protocol)
    assert result['status'] == 'passed'
    assert len(result['rows']) == 93


@pytest.mark.parametrize('failure', ['missing-grad', 'extra', 'shape', 'nonfinite', 'numerical',
    'stale-source', 'source-files', 'wheel', 'frozen', 'input-grad', 'missing-integer',
    'integer-mismatch', 'integer-hash', 'wrong-loader', 'wrong-trainer', 'calls',
    'updates', 'device', 'fallback', 'shim-oracle', 'no-update', 'integer-array'])
def test_rejects_incomplete_or_relabelled_training(protocol, failure):
    _source, _paths, values, arrays = protocol
    candidate = values[1]
    key = 'step/0/grad::parameter0'
    if failure == 'missing-grad':
        arrays[1].pop(key)
    elif failure == 'extra':
        arrays[1]['extra'] = np.ones(2)
    elif failure == 'shape':
        arrays[1][key] = np.ones(1, dtype='float32')
    elif failure == 'nonfinite':
        arrays[1][key][0] = np.nan
    elif failure == 'numerical':
        arrays[1][key][0] = 2.
    elif failure == 'stale-source':
        candidate['jittor_commit'] = '0' * 40
    elif failure == 'source-files':
        candidate['downstream_distribution']['source_sha256'] = {}
    elif failure == 'wheel':
        candidate['downstream_distribution'].pop('wheel_sha256')
    elif failure == 'frozen':
        candidate['frozen_parameters'] = ['parameter0']
    elif failure == 'input-grad':
        candidate['input_gradient_names'] = ['input_ids']
    elif failure == 'missing-integer':
        candidate['observed_inputs'].pop('step/0/observed_input::labels')
    elif failure in ('integer-mismatch', 'integer-hash'):
        item = candidate['observed_inputs']['step/0/observed_input::input_ids']
        item['values'][0][0] = 10
        if failure == 'integer-mismatch':
            item['sha256'] = hashlib.sha256(np.array(item['values'], dtype='int64').tobytes()).hexdigest()
    elif failure == 'wrong-loader':
        candidate['training_protocol']['identity']['loader_class'] = 'fake.LlamaLoader'
    elif failure == 'wrong-trainer':
        candidate['training_protocol']['identity']['trainer_registry_target'] = 'fake.Trainer'
    elif failure == 'calls':
        candidate['training_protocol']['upstream_calls']['swift.model.register.ModelLoader.load'] = 0
    elif failure == 'updates':
        candidate['training_protocol']['identity']['optimizer_updates'] = 0
    elif failure == 'device':
        candidate['observed_tensor_devices'][key] = 'cuda'
    elif failure == 'fallback':
        candidate['fallback_count'] = 1
    elif failure == 'shim-oracle':
        values[0]['interpreter']['is_shim'] = True
    elif failure == 'no-update':
        for side in arrays:
            for name in values[0]['trainable_parameters']:
                side['step/2/parameter::' + name] = side['initial::' + name].copy()
    elif failure == 'integer-array':
        for side in arrays:
            side[key] = np.ones(2, dtype='int64')
    with pytest.raises(ValueError):
        evaluate(protocol)


def test_integer_tokens_never_expand_floating_error_budget(protocol):
    for value in protocol[2]:
        for entry in value['observed_inputs'].values():
            data = np.full((1, 3), 10**9, dtype='int64')
            entry.update(values=data.tolist(), sha256=hashlib.sha256(data.tobytes()).hexdigest())
    protocol[3][1]['step/0/grad::parameter0'][0] += .1
    with pytest.raises(ValueError, match='Numerical mismatch'):
        evaluate(protocol)


def test_ms_swift_distribution_import_alias_preserves_real_hash_checks(tmp_path):
    package = tmp_path / 'swift'
    package.mkdir()
    path = package / '__init__.py'
    path.write_text("__version__ = '4.5.2'\n")
    source = {'repo': 'ms-swift', 'upstream': {'version': '4.5.2'}, 'package_sources': [
        {'path': 'swift/__init__.py', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]}
    module = SimpleNamespace(__file__=str(path), __version__='4.5.2')
    assert installed_distribution(source, module)['source_sha256'] == {
        row['path']: row['sha256'] for row in source['package_sources']}
    path.write_text('changed actual installed source\n')
    with pytest.raises(ValueError, match='differs'):
        installed_distribution(source, module)


def test_swift_maintained_gate_executes_actual_protocol_and_fails_closed():
    tree = ast.parse((ROOT / 'noxfile.py').read_text())
    session = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'ecosystem_swift')
    text = ast.get_source_segment((ROOT / 'noxfile.py').read_text(), session)
    for required in ('JITTOR_REQUIRE_REAL_TORCH', 'JITTOR_REQUIRE_SWIFT', 'JITTOR_TEST_REQUIRE_EXECUTION',
                     'CUDA_VISIBLE_DEVICES', 'JITTOR_ECOSYSTEM_SWIFT', 'test_ecosystem_swift.py'):
        assert required in text
    harness = (ROOT / 'compat/tests/torch/_ecosystem_swift_harness.py').read_text()
    assert "for seed in (0, 7, 19)" in harness and 'compare_pair' in harness
    probe = (SCRIPTS / 'swift_training_probe.py').read_text()
    assert 'super().train(trainer)' in probe and 'ObservedSft(sft).main()' in probe
    assert 'torch.optim.SGD(' not in probe and 'loss.backward(' not in probe
