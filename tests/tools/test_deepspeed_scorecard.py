"""Host evidence contracts; synthetic packaging fixtures never establish a level."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'
sys.path.insert(0, str(SCRIPTS))
validator = importlib.import_module('_deepspeed_scorecard')
collector = importlib.import_module('collect_deepspeed_scorecard')


def _git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE).decode().strip()


@pytest.fixture
def package(tmp_path):
    if os.name != 'posix':
        pytest.skip('POSIX directory-symlink packaging fixture')
    root = tmp_path / 'upstream'
    root.mkdir()
    files = {
        'deepspeed/__init__.py': '# packaging fixture, no runtime\n',
        'deepspeed/ops/__init__.py': '# ops package\n',
        'accelerator/__init__.py': '# actual linked accelerator source\n',
        'op_builder/__init__.py': '# actual linked builder source\n',
        'op_builder/all_ops.py': 'ALL_OPS = {}\n',
        'csrc/actual.cpp': 'int actual_source = 7;\n',
        'csrc/native.h': '#define ACTUAL_SOURCE 7\n',
        'csrc/fixture_test.py': '# non-package developer test, not a wheel module\n',
        'csrc/device.metal': '// MPS source is outside this CPU/CUDA wheel closure\n',
        'deepspeed/README.md': '# developer documentation\n',
        'setup.py': '# packaging fixture; no execution\n',
        'MANIFEST.in': 'recursive-include deepspeed *.cpp *.h\n',
        'version.txt': '0.19.7\n',
    }
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    (root / 'deepspeed/accelerator').symlink_to('../accelerator', target_is_directory=True)
    (root / 'deepspeed/ops/csrc').symlink_to('../../csrc', target_is_directory=True)
    (root / 'deepspeed/ops/op_builder').symlink_to('../../op_builder', target_is_directory=True)
    _git(root, 'init', '-q')
    _git(root, 'remote', 'add', 'origin', 'https://github.com/deepspeedai/DeepSpeed.git')
    _git(root, 'add', 'deepspeed', 'accelerator', 'op_builder', 'csrc', 'setup.py', 'MANIFEST.in', 'version.txt')
    _git(root, '-c', 'user.name=Packaging fixture', '-c', 'user.email=fixture@example.invalid',
         'commit', '-qm', 'Create isolated packaging fixture')
    commit = _git(root, 'rev-parse', 'HEAD')
    generated = {'version': '0.19.7+' + commit[:7], 'git_hash': commit[:7], 'git_branch': 'HEAD',
                 'installed_ops': {'cpu_adam': False}, 'accelerator_name': 'cpu',
                 'torch_info': {'version': '2.11', 'cuda_version': '12.8'}}
    members = {
        'deepspeed/__init__.py': files['deepspeed/__init__.py'],
        'deepspeed/ops/__init__.py': files['deepspeed/ops/__init__.py'],
        'deepspeed/accelerator/__init__.py': files['accelerator/__init__.py'],
        'deepspeed/ops/op_builder/__init__.py': files['op_builder/__init__.py'],
        'deepspeed/ops/op_builder/all_ops.py': files['op_builder/all_ops.py'],
        'deepspeed/ops/csrc/actual.cpp': files['csrc/actual.cpp'],
        'deepspeed/ops/csrc/native.h': files['csrc/native.h'],
        validator.GENERATED: ''.join('%s=%r\n' % item for item in generated.items()),
    }
    return root, members, tmp_path / 'fixture.whl'


def _wheel(path, members):
    with zipfile.ZipFile(path, 'w') as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return path


def test_source_closure_follows_actual_git_symlinks_and_includes_cpp(package):
    root, members, path = package
    source = validator.source_descriptor(root, _wheel(path, members))
    rows = {row['path']: row for row in source['package_sources']}
    assert set(rows) == set(members) - {validator.GENERATED}
    assert rows['deepspeed/ops/csrc/actual.cpp']['upstream_path'] == 'csrc/actual.cpp'
    assert rows['deepspeed/ops/op_builder/all_ops.py']['upstream_path'] == 'op_builder/all_ops.py'
    excluded = {row['path'] for row in source['excluded_package_sources']}
    assert 'deepspeed/ops/csrc/fixture_test.py' in excluded
    assert 'deepspeed/ops/csrc/device.metal' in excluded
    assert source['axes'] == []
    # A packaging-only fixture cannot impersonate the supported upstream SHA.
    with pytest.raises(ValueError, match='upstream source'):
        validator.validate_spec(_record(), source)


@pytest.mark.parametrize('tamper', ('cpp', 'missing-csrc', 'extra-python', 'generated-version', 'executable-generated'))
def test_full_source_wheel_proof_rejects_payload_or_generation_tampering(package, tamper):
    root, members, path = package
    if tamper == 'cpp':
        members['deepspeed/ops/csrc/actual.cpp'] = 'int changed = 9;\n'
    elif tamper == 'missing-csrc':
        del members['deepspeed/ops/csrc/native.h']
    elif tamper == 'extra-python':
        members['deepspeed/untracked_module.py'] = 'injected = True\n'
    elif tamper == 'generated-version':
        members[validator.GENERATED] = members[validator.GENERATED].replace('0.19.7', '0.18.2')
    else:
        marker = path.parent / 'must-not-execute'
        members[validator.GENERATED] += "trojan=__import__('pathlib').Path(%r).touch()\n" % str(marker)
    with pytest.raises(ValueError):
        validator.source_descriptor(root, _wheel(path, members))
    assert not (path.parent / 'must-not-execute').exists()


def test_dirty_upstream_and_unofficial_remote_are_rejected(package):
    root, members, path = package
    _wheel(path, members)
    original = (root / 'csrc/actual.cpp').read_text()
    (root / 'csrc/actual.cpp').write_text('int changed = 9;\n')
    with pytest.raises(ValueError, match='clean'):
        validator.source_descriptor(root, path)
    (root / 'csrc/actual.cpp').write_text(original)
    _git(root, 'remote', 'set-url', 'origin', 'https://example.invalid/fork')
    with pytest.raises(ValueError, match='official'):
        validator.source_descriptor(root, path)


@pytest.mark.parametrize('target', ('../../../escape', '/absolute', 'loop'))
def test_package_symlink_escape_and_cycles_are_rejected(target):
    path = 'deepspeed/loop'
    tree = {path: ('120000', 'link')}
    with pytest.raises(ValueError, match='symlink'):
        validator._package_closure(tree, {'link': target.encode()})


def _record():
    return {'repo': 'deepspeed', 'record_kind': 'protocol', 'registry_refs': [],
            'protocol': validator.protocol(1), 'case': 'zero1-world2-bf16-adamw', 'device': 'cpu'}


@pytest.mark.parametrize('tamper', ('stage3', 'optimizer', 'world', 'rank-bool', 'seed-bool', 'extra-axis', 'registry'))
def test_protocol_rejects_unmeasured_semantic_axes_and_fake_registry_breadth(tamper):
    record = _record()
    if tamper == 'registry':
        record['registry_refs'] = [{'repo': 'deepspeed', 'member': 'invented'}]
    else:
        name, value = {'stage3': ('zero_stage', 3), 'optimizer': ('optimizer', 'Adam'),
                       'world': ('world_size', 1), 'rank-bool': ('ranks', [False, True]),
                       'seed-bool': ('seed', False), 'extra-axis': ('offload', 'cpu')}[tamper]
        record['protocol'][name] = value
    with pytest.raises(ValueError):
        validator.validate_spec(record, {'repo': 'deepspeed', 'axes': [],
                                         'upstream': {'commit': validator.UPSTREAM_COMMIT}})


def _perturbation():
    before, after = {}, {}
    for name in validator.PARAMETERS:
        before['initial/parameter/' + name] = np.array([1.], dtype='float32')
        after['initial/parameter/' + name] = np.array([1.], dtype='float32')
    for index in range(6):
        before['step%d/output' % index] = np.array([1.], dtype='float32')
        after['step%d/output' % index] = np.array([1. if index == 0 else 1.25], dtype='float32')
        before['step%d/loss' % index] = np.array(.5, dtype='float32')
        after['step%d/loss' % index] = np.array(.75, dtype='float32')
    before['step0/input_grad/inputs'] = np.array([.01], dtype='float32')
    after['step0/input_grad/inputs'] = np.array([.02], dtype='float32')
    return before, after


def test_perturbation_can_be_forward_invariant_initially_but_must_change_training():
    before, after = _perturbation()
    identity = {'initial_weights_sha256': 'same-fixture'}
    validator.verify_perturbation(before, after, identity, identity)
    for category in ('initial/parameter/up.weight', 'step0/input_grad/inputs', 'all-outputs', 'step3/loss'):
        changed = copy.deepcopy(after)
        if category.startswith('initial/'):
            changed[category][0] += .125
        elif category == 'all-outputs':
            for index in range(6):
                changed['step%d/output' % index] = before['step%d/output' % index].copy()
        else:
            changed[category] = before[category].copy()
        with pytest.raises(ValueError):
            validator.verify_perturbation(before, changed, identity, identity)


def test_collector_refuses_to_overwrite_previous_evidence(tmp_path):
    marker = tmp_path / 'previous.json'
    marker.write_text(json.dumps({'old': 'immutable'}))
    with pytest.raises(ValueError, match='overwrite'):
        collector.collect({}, {}, None, None, tmp_path, '', '', {})
    assert json.loads(marker.read_text()) == {'old': 'immutable'}
