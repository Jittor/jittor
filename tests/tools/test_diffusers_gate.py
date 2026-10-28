"""Host-only launch/source boundaries; numerical claims require real workers."""

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
    spec = importlib.util.spec_from_file_location('diffusers_gate', ROOT / 'tools/run_diffusers_gate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / 'repo'
    (root / 'docs/results').mkdir(parents=True)
    report = root / 'docs/results/result.md'
    report.write_text('Synthetic host fixture; no numerical conclusion.')
    monkeypatch.setattr(module, 'ROOT', root)
    argv = ['--candidate-python', str(tmp_path / 'candidate/bin/python'),
            '--oracle-python', str(tmp_path / 'oracle/bin/python'), '--device', 'cpu',
            '--checkout', str(tmp_path / 'official-source'), '--wheel', str(tmp_path / 'fixed.whl'),
            '--output', str(tmp_path / 'new-output'), '--cache-root', str(tmp_path / 'benchmark-cache'),
            '--report', 'docs/results/result.md']
    return module, root, argv


def test_distinct_interpreters_and_explicit_external_benchmark_output_are_retained(gate):
    module, _, argv = gate
    args = module.options(argv)
    assert args.candidate_python != args.oracle_python
    assert args.output.name == 'new-output' and args.cache_root.name == 'benchmark-cache'


def test_stale_site_and_unit_cache_overlays_cannot_replace_explicit_runtime_roots(gate):
    module, _, _ = gate
    inherited = {name: '/stale/other-source' for name in (
        'JITTOR_ECOSYSTEM_PACKAGE_SITE', 'JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE',
        'JITTOR_ECOSYSTEM_CACHE_ROOT')}
    common = SimpleNamespace(environment=lambda *_: dict(inherited, JITTOR_HOME='/owned/benchmark'))
    value = module.runtime_environment(common, Path('/owned'), 'jittor', 'cpu')
    assert not inherited.keys() & value.keys()
    assert value['JITTOR_HOME'] == '/owned/benchmark'
    assert all(value[name] == '1' for name in ('JITTOR_REQUIRE_DIFFUSERS',
               'JITTOR_REQUIRE_REAL_TORCH', 'JITTOR_TEST_REQUIRE_EXECUTION'))
    assert value['JITTOR_ECOSYSTEM_TF32'] == '0'


@pytest.mark.parametrize('change', ('same-interpreter', 'inside-source', 'nonempty-output',
                                    'missing-report', 'outside-report', 'partial-cases', 'short-training'))
def test_invalid_or_reduced_launch_protocol_is_refused(gate, tmp_path, change):
    module, root, argv = gate
    if change == 'same-interpreter':
        argv[argv.index('--candidate-python') + 1] = argv[argv.index('--oracle-python') + 1]
    elif change == 'inside-source':
        argv[argv.index('--cache-root') + 1] = str(root / 'jit')
    elif change == 'nonempty-output':
        output = Path(argv[argv.index('--output') + 1])
        output.mkdir()
        (output / 'old-evidence').write_text('retain existing evidence')
    elif change == 'missing-report':
        argv[-1] = 'docs/results/missing.md'
    elif change == 'outside-report':
        path = tmp_path / 'outside.md'
        path.write_text('outside source')
        argv[-1] = str(path)
    else:
        argv += ['--cases', 'one-forward-only'] if change == 'partial-cases' else ['--steps', '1']
    with pytest.raises(SystemExit):
        module.options(argv)


def test_diffusers_callback_always_uses_its_fixed_source_validator(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import _diffusers_installation as diffusers
    import _metric_td_installation as component

    calls = []
    monkeypatch.setattr(component, 'validate_runtime_metadata', lambda *args, **kwargs: calls.append((args, kwargs)))
    value = {'source_validator': 'untrusted metadata cannot choose a validator'}
    diffusers.validate_runtime_metadata(value, 'a' * 40, {'actual': 'platform'})
    assert calls == [((value, 'a' * 40, {'actual': 'platform'}),
                      {'source_validator': diffusers.validate_measurement_source})]


def test_historical_snapshot_reads_immutable_git_tools_and_preserves_existing_closure(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import _diffusers_installation as diffusers
    import _metric_td_installation as component

    monkeypatch.setattr(component, 'framework_snapshot', lambda *args, **kwargs: {'existing-source': 'a' * 64})
    names = ('agent/skills/torch-compat-repo-runbook/scripts/_diffusers_installation.py',
             'tools/run_diffusers_gate.py', 'tools/run_transformers_gate.py')
    subprocess.run(['git', '-C', str(tmp_path), 'init', '-q'], check=True)
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('committed ' + name)
    subprocess.run(['git', '-C', str(tmp_path), 'add', *names], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Fixture', '-c',
                    'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
    baseline = subprocess.check_output(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'], text=True).strip()
    expected = {name: hashlib.sha256(('committed ' + name).encode()).hexdigest() for name in names}
    for name in names:
        (tmp_path / name).write_text('dirty current file')
    value = diffusers.framework_snapshot(tmp_path, baseline=baseline)
    assert value == dict(expected, **{'existing-source': 'a' * 64})
    value[names[0]] = 'b' * 64
    assert diffusers.framework_snapshot(tmp_path, baseline=baseline)[names[0]] == expected[names[0]]
