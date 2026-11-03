"""Tool relocation must not rewrite the identity of historical evidence."""

import importlib
from pathlib import Path
import subprocess

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / 'tools/ecosystem'


def test_selected_commit_controls_tool_paths(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    layout = importlib.import_module('_source_layout')

    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True).strip()

    git('init', '-q')
    git('config', 'user.name', 'Layout fixture')
    git('config', 'user.email', 'layout@example.invalid')
    old = 'agent/skills/torch-compat-repo-runbook/scripts/training_probe.py'
    current = 'tools/ecosystem/training_probe.py'
    (tmp_path / old).parent.mkdir(parents=True)
    (tmp_path / old).write_text('historical bytes\n')
    git('add', old)
    git('-c', 'commit.gpgsign=false', 'commit', '-qm', 'Historical tool')
    before = git('rev-parse', 'HEAD')
    (tmp_path / current).parent.mkdir(parents=True)
    (tmp_path / old).rename(tmp_path / current)
    (tmp_path / current).write_text('current bytes\n')
    git('add', old, current)
    git('-c', 'commit.gpgsign=false', 'commit', '-qm', 'Relocated tool')
    after = git('rev-parse', 'HEAD')
    assert layout.source_path(tmp_path, current) == current
    assert layout.source_path(tmp_path, current, before) == old
    assert layout.source_path(tmp_path, current, after) == current
    assert git('show', before + ':' + layout.source_path(tmp_path, current, before)) == 'historical bytes'
    # Today's presence of an old path cannot influence immutable past selection.
    (tmp_path / old).write_text('untracked stale duplicate\n')
    assert layout.source_path(tmp_path, current, after) == current
    git('add', old)
    git('-c', 'commit.gpgsign=false', 'commit', '-qm', 'Ambiguous duplicate')
    with pytest.raises(ValueError, match='Ambiguous'):
        layout.source_path(tmp_path, current, git('rev-parse', 'HEAD'))


def test_audit_does_not_require_agent_docs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    audit = importlib.import_module('audit_coverage')
    monkeypatch.setattr(audit, 'ecosystem_cases', lambda root: {})
    monkeypatch.setattr(audit, 'dedicated_gates', lambda root: {})
    monkeypatch.setattr(audit, 'git', lambda *args: 'a' * 40)
    for name in ('parity', 'speed', 'verl', 'longformer', 'openmmlab', 'openmmlab_installed', 'vllm'):
        path = tmp_path / ('compat/tests/torch/test_ecosystem_' + name + '.py')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')
    assert not (tmp_path / 'agent').exists()
    result = audit.audit(tmp_path)
    assert {row['repo'] for row in result['repos']} == set(audit.INTEGRATIONS)
    assert all(row['verified_level'] is None and row['registered_cases'] == [] for row in result['repos'])
    assert 'Integration repository' in audit.render_audit(result)


def test_flash_frozen_checkout_keeps_original_source_map(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    flash = importlib.import_module('_flash_attention_scorecard')
    files = ['noxfile.py', 'tests/_helpers/artifacts.py',
             'agent/skills/torch-compat-repo-runbook/scripts/registry_inventory.py',
             'agent/skills/torch-compat-repo-runbook/scripts/_flash_attention_scorecard.py',
             'agent/skills/torch-compat-repo-runbook/scripts/_protocol_scorecards.py',
             'agent/skills/flash-attention-torch-compat/scripts/flash_training_probe.py']
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original source\n')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True).strip()
    git('init', '-q')
    git('add', '--', *files)
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Frozen measured checkout')
    captured = flash.source_files(tmp_path, baseline=git('rev-parse', 'HEAD'))
    assert set(captured) == set(files)
    assert not any(name.startswith('tools/ecosystem') for name in captured)
