"""Offline guards against false two-rank TRL/DDP acceptance."""
import copy
import importlib.util
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'
PROBE = None


@pytest.fixture(autouse=True)
def load_probe_after_collection(monkeypatch):
    global PROBE
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location('trl_ddp_probe_test_owner', SCRIPTS / 'trl_ddp_probe.py')
    PROBE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PROBE)


def manifests(runtime='jittor', resumed=False):
    names = {'adapter_%03d' % index: [1] for index in range(112)}
    steps = list(range(6 if resumed else 1, 11))
    arrays = {}
    for step in steps:
        prefix = 'step/%d/' % step
        for key in ('input_ids', 'attention_mask', 'labels'):
            arrays[prefix + 'input/' + key] = dict(sha256='rank0-' + str(step), shape=[1, 8], dtype='int64')
        arrays[prefix + 'loss'] = dict(sha256='loss', shape=[], dtype='float32')
        for name in names:
            for category in ('gradient', 'parameter'):
                arrays[prefix + category + '/' + name] = dict(sha256='same', shape=[1], dtype='float32')
            for field in ('step', 'exp_avg', 'exp_avg_sq'):
                arrays[prefix + 'adam/' + name + '/' + field] = dict(sha256='same', shape=[1], dtype='float32')
    for index in range(3):
        for key in ('input_ids', 'next_logits'):
            arrays['final/%d/%s' % (index, key)] = dict(sha256='same', shape=[1], dtype='float32')
    left = dict(status='complete', runtime=runtime, rank=0, world_size=2, backend='nccl',
        warm_only=False, ddp_wrapped=True, fallback_count=0, optimizer_steps=steps,
        updates=[1] * len(steps), driver_device_checks=112, pid=101, cache_root='rank0-cache',
        gradient_norms=[1.] * len(steps), rng_before_loss={str(step): {} for step in steps},
        scheduler={str(step): {} for step in steps},
        checkpoint_files={'checkpoint-%d/%s' % (10 if resumed else 5, name): 'bound'
                          for name in PROBE.CHECKPOINT_FILES},
        cuda_visible_devices='0' if runtime == 'jittor' else '0,1', logical_device='cuda:0',
        trainable_parameters=names, protocol={}, initial={}, initial_adapters_sha256='initial',
        versions={}, source_files={}, runtime_sources={}, resumed_from_step=5 if resumed else None,
        arrays=arrays)
    right = copy.deepcopy(left)
    right.update(rank=1, pid=102, cache_root='rank1-cache',
                 cuda_visible_devices='1' if runtime == 'jittor' else '0,1',
                 logical_device='cuda:0' if runtime == 'jittor' else 'cuda:1')
    for step in steps:
        right['arrays']['step/%d/input/input_ids' % step]['sha256'] = 'rank1-' + str(step)
    return [left, right]


@pytest.mark.parametrize('runtime', ['torch', 'jittor'])
@pytest.mark.parametrize('resumed', [False, True])
def test_complete_two_rank_inventory(runtime, resumed):
    assert PROBE.validate_pair_manifests(manifests(runtime, resumed)) == list(range(6 if resumed else 1, 11))


@pytest.mark.parametrize('kind', ['missing_rank', 'world_one', 'warm_only', 'no_ddp', 'fallback',
                                'wrong_steps', 'missing_updates', 'same_process', 'same_cache',
                                'same_device', 'no_pointer_check', 'same_inputs', 'missing_gradient',
                                'missing_final', 'different_parameter', 'different_adam', 'extra_field',
                                'zero_gradient', 'missing_rng', 'missing_scheduler', 'missing_rank_rng'])
def test_fake_or_incomplete_rank_agreement_is_rejected(kind):
    pair = manifests()
    left, right = pair
    if kind == 'missing_rank':
        pair.pop()
    elif kind == 'world_one':
        right['world_size'] = 1
    elif kind == 'warm_only':
        right['warm_only'] = True
    elif kind == 'no_ddp':
        right['ddp_wrapped'] = False
    elif kind == 'fallback':
        right['fallback_count'] = 1
    elif kind == 'wrong_steps':
        right['optimizer_steps'] = [1]
    elif kind == 'missing_updates':
        right['updates'] = []
    elif kind == 'same_process':
        right['pid'] = left['pid']
    elif kind == 'same_cache':
        right['cache_root'] = left['cache_root']
    elif kind == 'same_device':
        right['cuda_visible_devices'] = left['cuda_visible_devices']
    elif kind == 'no_pointer_check':
        right['driver_device_checks'] = 0
    elif kind == 'same_inputs':
        right['arrays']['step/1/input/input_ids'] = copy.deepcopy(left['arrays']['step/1/input/input_ids'])
    elif kind == 'missing_gradient':
        del right['arrays']['step/1/gradient/adapter_000']
    elif kind == 'missing_final':
        for manifest in pair:
            del manifest['arrays']['final/0/next_logits']
    elif kind == 'different_parameter':
        right['arrays']['step/1/parameter/adapter_000']['sha256'] = 'changed'
    elif kind == 'different_adam':
        right['arrays']['step/1/adam/adapter_000/exp_avg']['sha256'] = 'changed'
    elif kind == 'zero_gradient':
        right['gradient_norms'][0] = 0.
    elif kind == 'missing_rng':
        del right['rng_before_loss']['1']
    elif kind == 'missing_scheduler':
        del right['scheduler']['1']
    elif kind == 'missing_rank_rng':
        del right['checkpoint_files']['checkpoint-5/rng_state_1.pth']
    else:
        for manifest in pair:
            manifest['arrays']['extra'] = dict(sha256='same', shape=[], dtype='float32')
    with pytest.raises(AssertionError):
        PROBE.validate_pair_manifests(pair)


def test_loss_may_differ_between_ranks():
    pair = manifests()
    pair[1]['arrays']['step/1/loss']['sha256'] = 'different local loss'
    PROBE.validate_pair_manifests(pair)


def test_oracle_ignores_unused_candidate_runtime_context():
    pair = manifests('torch')
    pair[1]['runtime_sources'] = {'unused-candidate-source': 'changed'}
    PROBE.validate_pair_manifests(pair)
    pair = manifests('jittor')
    pair[1]['runtime_sources'] = {'actual-candidate-source': 'changed'}
    with pytest.raises(AssertionError, match='runtime sources'):
        PROBE.validate_pair_manifests(pair)


def test_launchers_are_public_and_rank_counts_are_explicit(tmp_path):
    for runtime, module in [('torch', 'torch.distributed.run'), ('jittor', 'jittor.distributed.launch')]:
        command = PROBE.launch_command('python', runtime, 'probe.py', ['3', '7'], tmp_path, ['worker'])
        assert command[:3] == ['python', '-m', module]
        assert any(value.endswith('=2') and 'nproc' in value for value in command)
        if runtime == 'jittor':
            assert '--device-ids=3,7' in command
            assert '--backend=nccl' in command


def test_existing_rank_caches_keep_identity_without_copy_or_rename(tmp_path):
    homes = [tmp_path / 'published0', tmp_path / 'published1']
    for rank in range(2):
        env = {}
        PROBE.rank_environment(env, tmp_path / 'run', 'jittor', rank, homes)
        assert env['JITTOR_HOME'] == str(homes[rank])
        assert 'JT_BUILD_CACHE_NAME' not in env
        assert 'cache_name' not in env
        assert 'HOME' not in env


def test_probe_source_bindings_exist():
    root = SCRIPTS.parents[3]
    assert len(PROBE._owner_sources(root)) >= 10
    assert len(PROBE._sources(root)) == 6
