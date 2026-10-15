#!/usr/bin/env python3
"""Run both complete maintained Metrics/TensorDict component protocols."""

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'
REPOS = ('torchmetrics', 'tensordict')
EXPECTED = {'torchmetrics': (4, 6, 2), 'tensordict': (6, 29, 3)}
RUNTIME_KEYS = ('HOME', 'REAL_HOME', 'JITTOR_HOME', 'TMPDIR', 'XDG_CACHE_HOME',
                'JITTOR_TORCH_CACHE_ROOT', 'JITTOR_TORCH_RUNTIME_ROOT', 'JITTOR_TORCH_HOME',
                'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'PYTHONPATH', 'JITTOR_SOURCE_ROOT',
                'JITTOR_TORCH_SHIM', 'JT_BUILD_NVCC_PATH', 'nvcc_path', 'cache_name',
                'LD_LIBRARY_PATH', 'LD_PRELOAD', 'JITTOR_REQUIRE_TORCHMETRICS',
                'JITTOR_REQUIRE_TENSORDICT', 'JITTOR_REQUIRE_REAL_TORCH',
                'JITTOR_TEST_REQUIRE_EXECUTION', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_environment(common, cache, runtime, device):
    env = common.environment(cache, runtime, device)
    for name in ('JITTOR_ECOSYSTEM_PACKAGE_SITE', 'JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE',
                 'JITTOR_ECOSYSTEM_CACHE_ROOT'):
        env.pop(name, None)
    env.update(JITTOR_ECOSYSTEM_TF32='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', cache_name='metric-tensordict-source-' + device,
               JITTOR_REQUIRE_TORCHMETRICS='1', JITTOR_REQUIRE_TENSORDICT='1',
               JITTOR_REQUIRE_REAL_TORCH='1', JITTOR_TEST_REQUIRE_EXECUTION='1')
    if device == 'cuda':
        env['JITTOR_CUDA_WHEEL_STRICT'] = '1'
    return env


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-python', type=Path, required=True)
    parser.add_argument('--oracle-python', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    for repo in REPOS:
        parser.add_argument('--' + repo + '-checkout', type=Path, required=True)
        parser.add_argument('--' + repo + '-wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--report', default='docs/results/2026-10-16-metrics-tensordict-maintained-cpu-nightly.md')
    args = parser.parse_args(argv)
    names = ['candidate_python', 'oracle_python', 'output', 'cache_root']
    names += [repo + '_' + kind for repo in REPOS for kind in ('checkout', 'wheel')]
    for name in names:
        setattr(args, name, getattr(args, name).absolute())
    if args.candidate_python == args.oracle_python:
        parser.error('Candidate and genuine oracle require independent environment interpreters')
    for path in (args.output, args.cache_root):
        if path.resolve() == ROOT or ROOT in path.resolve().parents:
            parser.error('Outputs and benchmark caches must be outside the source checkout')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Use a new empty output directory')
    if (args.output.resolve() == args.cache_root.resolve()
            or args.output.resolve() in args.cache_root.resolve().parents
            or args.cache_root.resolve() in args.output.resolve().parents):
        parser.error('Raw output and benchmark cache roots must be disjoint')
    report = (ROOT / args.report).resolve()
    if not args.report.startswith('docs/results/') or ROOT not in report.parents or not report.is_file():
        parser.error('An existing versioned results report is required')
    return args


def main(argv=None):
    args = options(argv)
    sys.path.insert(0, str(SCRIPTS))
    from _metric_td_installation import digest, framework_snapshot, source_descriptor
    from audit_coverage import scorecard
    from collect_component_scorecard import SPECS, collect, compare_arrays, parameter_names, require
    import numpy as np

    require(not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain']),
            'Maintained component execution requires clean committed source')
    baseline = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    frozen = framework_snapshot(ROOT)
    require(frozen.get('tools/run_metric_td_gate.py') == digest(__file__),
            'The complete maintained runner must be committed and captured in the source closure')
    sources = {repo: source_descriptor(repo, getattr(args, repo + '_checkout'),
                                      getattr(args, repo + '_wheel')) for repo in REPOS}
    args.output.mkdir(parents=True, exist_ok=True)
    common = load('metric_td_runtime_environment', ROOT / 'tools/run_transformers_gate.py')
    verify = load('metric_td_reference_comparison', SCRIPTS / 'verify_repo.py')
    environments = {runtime: runtime_environment(common, args.cache_root, runtime, args.device)
                    for runtime in ('torch', 'jittor')}
    commands, records, worker_pids = [], [], set()

    def run(command, env, log):
        require(framework_snapshot(ROOT) == frozen, 'Measured framework/runner bytes changed')
        command = list(map(str, command))
        row = {'argv': command, 'log': str(log.relative_to(args.output)), 'baseline': baseline,
               'environment': {key: env.get(key) for key in (*RUNTIME_KEYS, 'CUDA_VISIBLE_DEVICES',
                    'JT_BACKEND', 'JT_USE_CUDA', 'JITTOR_ECOSYSTEM_TF32', 'JITTOR_CUDA_WHEEL_STRICT',
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}
        commands.append(row)
        print('METRIC_TD_PHASE', str(log.relative_to(args.output)), flush=True)
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=3600)
        row['returncode'] = result.returncode
        (args.output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        require(result.returncode == 0, 'Actual maintained worker failed; see ' + str(log))

    for repo in REPOS:
        spec, folder = SPECS[repo], args.output / repo
        folder.mkdir()
        case, wrappers = spec['case'], {}
        for runtime, python in (('torch', args.oracle_python), ('jittor', args.candidate_python)):
            path = folder / ('launch-' + runtime)
            assignments = [key + '=' + environments[runtime][key]
                           for key in RUNTIME_KEYS if key in environments[runtime]]
            command = ['env', *assignments, str(python), str(SCRIPTS / 'component_measure.py'),
                       '--root', str(ROOT), '--repo', repo,
                       '--checkout', str(getattr(args, repo + '_checkout')),
                       '--wheel', str(getattr(args, repo + '_wheel')), '--']
            path.write_text('#!/bin/sh\nexec ' + ' '.join(shlex.quote(x) for x in command) + ' "$@"\n')
            path.chmod(0o700)
            wrappers[runtime] = path
        measurement = folder / 'measurements'
        controller_env = dict(environments['torch'], JITTOR_TORCH_SHIM='0', PYTHONPATH='')
        run([args.oracle_python, SCRIPTS / 'verify_repo.py', '--repo', repo, '--cases', case,
             '--device', args.device, '--oracle-python', wrappers['torch'], '--shim-python', wrappers['jittor'],
             '--seed', '0', '--repeats', '3', '--no-memory', '--out', measurement],
            controller_env, folder / 'base.log')
        weights = measurement / (case + '.ref.weights.npz')
        names = parameter_names(spec, weights)
        require(names == spec['parameters'], 'Original component parameter scope changed')
        with np.load(weights) as initial:
            require(sorted(initial.files) == sorted(names), 'Incomplete actual component initial weights')
        expected = ['__output__', *['grad::' + name for name in names],
                    *['ingrad::' + name for name in spec['inputs']]]
        values, paths = [], []
        for runtime, label in (('torch', 'ref'), ('jittor', 'shim')):
            output = measurement / (case + '.perturb.' + label + '.npz')
            run([wrappers[runtime], ROOT / 'compat/tests/torch/_ecosystem_runner.py', case, output,
                 '--runtime', runtime, '--device', args.device, '--seed', '7', '--repeats', '3',
                 '--weights', weights], environments[runtime], folder / ('perturb-' + runtime + '.log'))
            paths.append(output)
            values.append(json.loads(output.with_suffix('.json').read_text()))
        rows, absolute, relative, _worst, _abskey, missing = verify._compare(*paths)
        require(not missing and not verify._pair_problems(*values, args.device),
                'Invalid fixed-weight input perturbation pair')
        compare_arrays(*paths, expected, .002, .01)
        for label in ('ref', 'shim'):
            with np.load(measurement / (case + '.' + label + '.npz')) as base, np.load(
                    measurement / (case + '.perturb.' + label + '.npz')) as changed:
                require(not np.array_equal(base['__output__'], changed['__output__']),
                        'No actual input perturbation: ' + repo + '/' + label)
        (measurement / 'perturb-comparison.json').write_text(json.dumps({'case': case, 'status': 'passed',
            'seed': 7, 'input_perturbation_verified': True, 'reference_result': values[0],
            'candidate_result': values[1], 'rows': rows, 'max_abs': absolute,
            'max_scaled_relative': relative}, indent=2) + '\n')
        if repo == 'tensordict':
            for runtime, label in (('torch', 'ref'), ('jittor', 'shim')):
                run([wrappers[runtime], SCRIPTS / 'training_probe.py', '--root', ROOT, '--case', case,
                     '--runtime', runtime, '--device', args.device, '--seed', '0', '--steps', '3',
                     '--learning-rate', '.001', '--weights', weights,
                     '--output', measurement / (case + '.train.' + label + '.npz')],
                    environments[runtime], folder / ('train-' + runtime + '.log'))
        packets = [json.loads(path.read_text()) for path in measurement.glob(case + '.*.json')]
        count, pairs, level = EXPECTED[repo]
        pids = {value['interpreter']['pid'] for value in packets}
        require(len(packets) == len(pids) == count and not pids & worker_pids,
                'Missing independent base/perturb/training worker processes')
        worker_pids.update(pids)
        base = {runtime: json.loads((measurement / (case + '.' + label + '.json')).read_text())
                for runtime, label in (('torch', 'ref'), ('jittor', 'shim'))}
        platform = dict(base['jittor']['platform'],
                        oracle=dict(base['torch']['interpreter'], torch_npu_version='not-applicable'))
        platform_path = folder / 'platform.json'
        platform_path.write_text(json.dumps(platform, indent=2) + '\n')
        record = collect(sources[repo], measurement, platform_path, baseline, args.report,
                         [row['argv'] for row in commands])
        require(record['numerics']['checked_tensors'] == pairs and record['level'] == level,
                'The complete original component protocol changed')
        records.append(record)
    require(len(worker_pids) == 10, 'Both complete original components require ten actual workers')
    result = scorecard(sources, records, baseline, ROOT)
    (args.output / 'scorecard.json').write_text(json.dumps(result, indent=2) + '\n')
    require(len(result['accepted']) == 2 and not result['rejected'] and not result.get('blocked'),
            'Shared auditor refused the complete actual component evidence')
    require(framework_snapshot(ROOT) == frozen
            and not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'])
            and subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip() == baseline,
            'Source changed during maintained execution')
    (args.output / 'records.json').write_text(json.dumps({'schema_version': 2, 'inventories': sources,
                                                        'records': records}, indent=2) + '\n')
    print('METRIC_TD_GATE_PASSED ' + json.dumps({'baseline': baseline, 'device': args.device,
          'workers': 10, 'pairs': 35, 'torchmetrics_level': 2, 'tensordict_level': 3}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
