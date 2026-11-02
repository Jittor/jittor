"""Real Git history guards for migrated component source-evidence closures."""
import hashlib
import importlib
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = 'agent/skills/torch-compat-repo-runbook/scripts/'
LEGACY = 'tools/run_transformers_gate.py'
ENVIRONMENT = 'tools/ecosystem_environment.py'
RUNTIME = SCRIPTS + '_component_runtime.py'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def write(root, name, content='VALUE = 1\n'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def commit(root, *names):
    git(root, 'add', '--', *names)
    git(root, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-qm', 'source fixture')
    return git(root, 'rev-parse', 'HEAD')


@pytest.fixture
def history(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / SCRIPTS))
    metric = importlib.import_module('_metric_td_installation')
    diffusers = importlib.import_module('_diffusers_installation')
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    names = ['python/jittor/__init__.py', 'compat/__init__.py', 'src/op.cc', 'backends/cpu/op.cc',
             'noxfile.py', 'tests/_helpers/artifacts.py', LEGACY, 'tools/run_metric_td_gate.py']
    names += [SCRIPTS + name for name in ('_vllm_installation.py', '_vllm_component_contract.py',
        'component_measure.py', '_installed_source_proof.py', 'registry_inventory.py',
        'collect_vllm_scorecard.py', '_metric_td_installation.py', 'collect_component_scorecard.py',
        'training_probe.py', 'verify_repo.py', '_runtime_platform.py', '_diffusers_installation.py')]
    for name in names:
        write(root, name)
    write(root, 'tools/run_diffusers_gate.py', "common = load('environment', ROOT / 'tools/run_transformers_gate.py')\n")
    names.append('tools/run_diffusers_gate.py')
    old = commit(root, *names)
    return root, old, metric, diffusers


def migrate(root):
    git(root, 'rm', '--', LEGACY, 'tools/run_metric_td_gate.py')
    write(root, ENVIRONMENT)
    write(root, RUNTIME)
    write(root, SCRIPTS + '_metric_td_installation.py', 'from _component_runtime import validate_runtime_metadata\n')
    write(root, 'tools/run_diffusers_gate.py', "common = load('environment', ROOT / 'tools/ecosystem_environment.py')\n")
    return commit(root, ENVIRONMENT, RUNTIME, SCRIPTS + '_metric_td_installation.py', 'tools/run_diffusers_gate.py')


def test_historical_closure_survives_new_environment_extraction(history):
    root, old, metric, diffusers = history
    previous = diffusers.framework_snapshot(root, baseline=old)
    assert LEGACY in previous and ENVIRONMENT not in previous and RUNTIME not in previous
    new = migrate(root)
    current = diffusers.framework_snapshot(root, baseline=new)
    assert ENVIRONMENT in current and RUNTIME in current and LEGACY not in current
    assert diffusers.framework_snapshot(root, baseline=old) == previous
    write(root, ENVIRONMENT, 'DIRTY = True\n')
    write(root, LEGACY, 'UNTRACKED_OLD_NAME = True\n')
    assert diffusers.framework_snapshot(root, baseline=new) == current
    assert diffusers.framework_snapshot(root, baseline=old) == previous
    assert metric.framework_snapshot(root, baseline=old)[LEGACY] == previous[LEGACY]


def test_unrelated_new_helper_does_not_replace_the_historical_consumers_dependency(history):
    root, old, _, diffusers = history
    previous = diffusers.framework_snapshot(root, baseline=old)
    write(root, ENVIRONMENT)
    both = commit(root, ENVIRONMENT)
    assert diffusers.framework_snapshot(root, baseline=both) == previous


@pytest.mark.parametrize('missing', [ENVIRONMENT, RUNTIME])
def test_new_commit_cannot_omit_a_dependency_still_imported_by_its_consumer(history, missing):
    root, _, _, diffusers = history
    migrate(root)
    git(root, 'rm', '--', missing)
    broken = commit(root, 'tools/run_diffusers_gate.py')
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        diffusers.framework_snapshot(root, baseline=broken)


def test_current_snapshot_requires_tracked_runtime_helper(history):
    root, _, metric, _ = history
    migrate(root)
    git(root, 'rm', '--cached', '--', RUNTIME)
    with pytest.raises(ValueError):
        metric.framework_snapshot(root)


def test_symlink_identity_uses_link_bytes_and_not_the_targets_current_content(history):
    root, _, metric, _ = history
    link = root/'python/jittor/compat'
    link.symlink_to('../../compat')
    baseline=commit(root, 'python/jittor/compat')
    digest=hashlib.sha256(b'../../compat').hexdigest()
    assert metric.framework_snapshot(root, baseline=baseline)['python/jittor/compat']==digest
    write(root,'compat/__init__.py','CHANGED = True\n')
    assert metric.framework_snapshot(root, baseline=baseline)['python/jittor/compat']==digest
