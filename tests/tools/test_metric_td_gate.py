"""Host-only complete-protocol/source guards; synthetic files grant no L grade."""

import hashlib
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'


@pytest.fixture
def gate(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('metric_td_gate', ROOT / 'tools/run_metric_td_gate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / 'repo'
    (root / 'docs/results').mkdir(parents=True)
    (root / 'docs/results/result.md').write_text('Host fixture; no real model run.')
    monkeypatch.setattr(module, 'ROOT', root)
    argv = ['--candidate-python', str(tmp_path / 'candidate/bin/python'),
            '--oracle-python', str(tmp_path / 'oracle/bin/python'), '--device', 'cpu',
            '--output', str(tmp_path / 'raw-output'), '--cache-root', str(tmp_path / 'benchmark-cache'),
            '--report', 'docs/results/result.md']
    for repo in module.REPOS:
        argv += ['--' + repo + '-checkout', str(tmp_path / repo),
                 '--' + repo + '-wheel', str(tmp_path / (repo + '.whl'))]
    return module, root, argv


def test_both_original_complete_components_are_mandatory(gate):
    module, _, argv = gate
    value = module.options(argv)
    assert value.torchmetrics_checkout != value.tensordict_checkout
    assert value.candidate_python != value.oracle_python
    assert module.REPOS == ('torchmetrics', 'tensordict')
    assert module.EXPECTED == {'torchmetrics': (4, 6, 2), 'tensordict': (6, 29, 3)}


@pytest.mark.parametrize('change', ('same-python', 'inside-source', 'output-file', 'stale-output',
                                    'overlapping-cache', 'one-repo-only', 'fewer-steps',
                                    'larger-budget', 'missing-wheel', 'outside-report'))
def test_invalid_or_reduced_original_protocol_is_refused(gate, tmp_path, change):
    module, root, argv = gate
    if change == 'same-python':
        argv[argv.index('--candidate-python') + 1] = argv[argv.index('--oracle-python') + 1]
    elif change == 'inside-source':
        argv[argv.index('--cache-root') + 1] = str(root / 'cache')
    elif change in ('output-file', 'stale-output'):
        path = Path(argv[argv.index('--output') + 1])
        if change == 'output-file':
            path.write_text('do not overwrite')
        else:
            path.mkdir()
            (path / 'old').write_text('preserve old evidence')
    elif change == 'overlapping-cache':
        argv[argv.index('--cache-root') + 1] = argv[argv.index('--output') + 1] + '/cache'
    elif change == 'one-repo-only':
        argv += ['--repo', 'torchmetrics']
    elif change == 'fewer-steps':
        argv += ['--steps', '1']
    elif change == 'larger-budget':
        argv += ['--atol', '1']
    elif change == 'missing-wheel':
        index = argv.index('--tensordict-wheel')
        del argv[index:index + 2]
    else:
        outside = tmp_path / 'outside.md'
        outside.write_text('outside')
        argv[argv.index('--report') + 1] = str(outside)
    with pytest.raises(SystemExit):
        module.options(argv)


def test_runtime_paths_cannot_be_replaced_by_stale_sites_or_unit_cache(gate):
    module, _, _ = gate
    stale = {name: '/stale' for name in ('JITTOR_ECOSYSTEM_PACKAGE_SITE',
        'JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE', 'JITTOR_ECOSYSTEM_CACHE_ROOT')}
    common = SimpleNamespace(environment=lambda *_: dict(stale, JITTOR_HOME='/owned/benchmark'))
    value = module.runtime_environment(common, Path('/owned'), 'jittor', 'cpu')
    assert not stale.keys() & value.keys()
    assert value['JITTOR_HOME'] == '/owned/benchmark'
    for name in ('JITTOR_REQUIRE_TORCHMETRICS', 'JITTOR_REQUIRE_TENSORDICT',
                 'JITTOR_REQUIRE_REAL_TORCH', 'JITTOR_TEST_REQUIRE_EXECUTION'):
        assert value[name] == '1'
    assert value['JITTOR_ECOSYSTEM_TF32'] == '0'
    assert value['OMP_NUM_THREADS'] == value['MKL_NUM_THREADS'] == value['OPENBLAS_NUM_THREADS'] == '2'


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import _metric_td_installation as component
    import _vllm_installation as vllm

    names = ['python/jittor/__init__.py',
             *['agent/skills/torch-compat-repo-runbook/scripts/' + name + '.py'
               for name in ('_metric_td_installation', 'collect_component_scorecard',
                            'training_probe', 'verify_repo', '_runtime_platform')],
             'tools/run_transformers_gate.py']
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('committed fixture ' + name)
    subprocess.run(['git', '-C', str(tmp_path), 'init', '-q'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'add', *names], check=True)

    def commit():
        subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Fixture', '-c',
                        'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
        return subprocess.check_output(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'], text=True).strip()

    old = commit()
    monkeypatch.setattr(vllm, 'framework_sources', lambda *_args, **_kwargs: names[:1])
    return component, tmp_path, names, old, commit


def test_historical_closure_uses_git_presence_and_does_not_relabel_old_rows(source_tree):
    component, root, names, old, commit = source_tree
    legacy = component.framework_snapshot(root, baseline=old)
    assert names[-1] not in legacy  # It existed, but was not part of the old protocol.
    runner = root / 'tools/run_metric_td_gate.py'
    runner.write_text('new complete runner')
    assert component.framework_snapshot(root) == legacy  # Untracked disk presence grants nothing.
    subprocess.run(['git', '-C', str(root), 'add', 'tools/run_metric_td_gate.py'], check=True)
    with pytest.raises(ValueError, match='must be committed'):
        component.framework_snapshot(root)
    current = commit()
    measured = component.framework_snapshot(root, baseline=current)
    assert measured['tools/run_metric_td_gate.py'] == hashlib.sha256(runner.read_bytes()).hexdigest()
    assert measured[names[-1]] == hashlib.sha256((root / names[-1]).read_bytes()).hexdigest()
    assert component.framework_snapshot(root) == measured
    runner.write_text('changed current runner')
    with pytest.raises(ValueError, match='changed committed'):
        component.framework_snapshot(root)
    assert component.framework_snapshot(root, baseline=old) == legacy
    assert component.framework_snapshot(root, baseline=current) == measured
    measured['tools/run_metric_td_gate.py'] = '0' * 64
    assert component.framework_snapshot(root, baseline=current)['tools/run_metric_td_gate.py'] != '0' * 64


@pytest.mark.parametrize('missing', ('tools/run_metric_td_gate.py', 'tools/run_transformers_gate.py'))
def test_current_tracked_missing_tool_is_refused(source_tree, missing):
    component, root, _, _, commit = source_tree
    runner = root / 'tools/run_metric_td_gate.py'
    runner.write_text('new complete runner')
    subprocess.run(['git', '-C', str(root), 'add', 'tools/run_metric_td_gate.py'], check=True)
    commit()
    (root / missing).unlink()
    with pytest.raises(ValueError, match='Missing or changed committed'):
        component.framework_snapshot(root)
