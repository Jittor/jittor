#!/usr/bin/env python3
"""Replay the complete maintained UNet component with actual installation proof."""

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'
CASE = 'diffusers_unet2d'
RUNTIME_KEYS = ('HOME', 'REAL_HOME', 'JITTOR_HOME', 'TMPDIR', 'XDG_CACHE_HOME',
                'JITTOR_TORCH_CACHE_ROOT', 'JITTOR_TORCH_RUNTIME_ROOT', 'JITTOR_TORCH_HOME',
                'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'PYTHONPATH', 'JITTOR_SOURCE_ROOT',
                'JITTOR_TORCH_SHIM', 'JT_BUILD_NVCC_PATH', 'nvcc_path', 'cache_name',
                'LD_LIBRARY_PATH', 'LD_PRELOAD', 'JITTOR_REQUIRE_DIFFUSERS',
                'JITTOR_REQUIRE_REAL_TORCH', 'JITTOR_TEST_REQUIRE_EXECUTION',
                'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_environment(common, cache, runtime, device):
    env = common.environment(cache, runtime, device)
    # These legacy overlays must not substitute another package or unit cache
    # for the explicitly supplied private runtime environments/benchmark roots.
    for name in ('JITTOR_ECOSYSTEM_PACKAGE_SITE', 'JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE',
                 'JITTOR_ECOSYSTEM_CACHE_ROOT'):
        env.pop(name, None)
    env.update(JITTOR_ECOSYSTEM_TF32='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', cache_name='diffusers-installed-bound-' + device,
               JITTOR_REQUIRE_DIFFUSERS='1', JITTOR_REQUIRE_REAL_TORCH='1',
               JITTOR_TEST_REQUIRE_EXECUTION='1')
    if device == 'cuda':
        env['JITTOR_CUDA_WHEEL_STRICT'] = '1'
    return env


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-python', type=Path, required=True)
    parser.add_argument('--oracle-python', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--report', default='docs/results/2026-10-16-diffusers-official-installation.md')
    args = parser.parse_args(argv)
    for name in ('candidate_python', 'oracle_python', 'checkout', 'wheel', 'output', 'cache_root'):
        setattr(args, name, getattr(args, name).absolute())
    if args.candidate_python == args.oracle_python:
        parser.error('Candidate and genuine oracle need separate environment interpreters')
    for path in (args.output, args.cache_root):
        if path.resolve() == ROOT or ROOT in path.resolve().parents:
            parser.error('Output and benchmark caches must be outside the source checkout')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new empty output directory')
    report = (ROOT / args.report).resolve()
    if not args.report.startswith('docs/results/') or ROOT not in report.parents or not report.is_file():
        parser.error('An existing versioned results report is required')
    return args


def main(argv=None):
    args = options(argv)
    sys.path.insert(0, str(SCRIPTS))
    from _diffusers_installation import framework_snapshot, source_descriptor
    from collect_component_scorecard import SPECS, collect, compare_arrays, parameter_names, require
    from audit_coverage import scorecard

    require(not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain']),
            'Diffusers measurements require a clean committed source')
    baseline = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    frozen = framework_snapshot(ROOT)
    source = source_descriptor(args.checkout, args.wheel)
    args.output.mkdir(parents=True, exist_ok=True)
    common = load('diffusers_runtime_environment', ROOT / 'tools/run_transformers_gate.py')
    verify = load('diffusers_reference_comparison', SCRIPTS / 'verify_repo.py')
    environments, wrappers = {}, {}
    for runtime, python in (('torch', args.oracle_python), ('jittor', args.candidate_python)):
        env = runtime_environment(common, args.cache_root, runtime, args.device)
        environments[runtime] = env
        path = args.output / ('launch-' + runtime)
        assignments = [key + '=' + env[key] for key in RUNTIME_KEYS if key in env]
        command = ['env', *assignments, str(python), str(SCRIPTS / 'component_measure.py'),
                   '--root', str(ROOT), '--repo', 'diffusers', '--checkout', str(args.checkout),
                   '--wheel', str(args.wheel), '--']
        path.write_text('#!/bin/sh\nexec ' + ' '.join(shlex.quote(x) for x in command) + ' "$@"\n')
        path.chmod(0o700)
        wrappers[runtime] = path
    commands = []

    def run(command, env, log):
        require(framework_snapshot(ROOT) == frozen, 'Framework/probe bytes changed during Diffusers gate')
        command = list(map(str, command))
        row = {'argv': command, 'log': str(log.relative_to(args.output)), 'baseline': baseline,
               'environment': {key: env.get(key) for key in (*RUNTIME_KEYS, 'CUDA_VISIBLE_DEVICES',
                    'JT_BACKEND', 'JT_USE_CUDA', 'JITTOR_ECOSYSTEM_TF32', 'JITTOR_CUDA_WHEEL_STRICT',
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}
        commands.append(row)
        print('DIFFUSERS_PHASE', log.name, flush=True)
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=3600)
        row['returncode'] = result.returncode
        (args.output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        require(result.returncode == 0, 'Actual Diffusers worker failed; see ' + str(log))

    folder = args.output / 'measurements'
    controller_env = dict(environments['torch'], JITTOR_TORCH_SHIM='0', PYTHONPATH='')
    run([args.oracle_python, SCRIPTS / 'verify_repo.py', '--repo', 'diffusers', '--cases', CASE,
         '--device', args.device, '--oracle-python', wrappers['torch'], '--shim-python', wrappers['jittor'],
         '--seed', '0', '--repeats', '3', '--no-memory', '--out', folder],
        controller_env, args.output / 'base.log')
    weights = folder / (CASE + '.ref.weights.npz')
    names = parameter_names(SPECS['diffusers'], weights)
    require(len(names) == 144, 'Original full UNet parameter scope changed')
    expected = ['__output__', *['grad::' + name for name in names], 'ingrad::sample']
    values, paths = [], []
    for runtime, label in (('torch', 'ref'), ('jittor', 'shim')):
        output = folder / (CASE + '.perturb.' + label + '.npz')
        run([wrappers[runtime], ROOT / 'compat/tests/torch/_ecosystem_runner.py', CASE, output,
             '--runtime', runtime, '--device', args.device, '--seed', '7', '--repeats', '3', '--weights', weights],
            environments[runtime], args.output / ('perturb-' + runtime + '.log'))
        paths.append(output)
        values.append(json.loads(output.with_suffix('.json').read_text()))
    rows, absolute, relative, _worst, _abskey, missing = verify._compare(*paths)
    require(not missing and not verify._pair_problems(*values, args.device), 'Invalid original perturbation pair')
    compare_arrays(*paths, expected, .002, .01)
    import numpy as np

    for runtime, label in (('torch', 'ref'), ('jittor', 'shim')):
        with np.load(folder / (CASE + '.' + label + '.npz')) as base, np.load(
                folder / (CASE + '.perturb.' + label + '.npz')) as perturbed:
            require(not np.array_equal(base['__output__'], perturbed['__output__']), 'No real input perturbation')
    (folder / 'perturb-comparison.json').write_text(json.dumps({'case': CASE, 'status': 'passed', 'seed': 7,
        'input_perturbation_verified': True, 'reference_result': values[0], 'candidate_result': values[1],
        'rows': rows, 'max_abs': absolute, 'max_scaled_relative': relative}, indent=2) + '\n')
    for runtime, label in (('torch', 'ref'), ('jittor', 'shim')):
        run([wrappers[runtime], SCRIPTS / 'training_probe.py', '--root', ROOT, '--case', CASE,
             '--runtime', runtime, '--device', args.device, '--seed', '0', '--steps', '3',
             '--learning-rate', '.001', '--weights', weights,
             '--output', folder / (CASE + '.train.' + label + '.npz')],
            environments[runtime], args.output / ('train-' + runtime + '.log'))
    packets = [json.loads(path.read_text()) for path in folder.glob('*.json')
               if path.name.startswith(CASE + '.')]
    require(len(packets) == 6 and len({v['interpreter']['pid'] for v in packets}) == 6,
            'Missing six independent actual base/perturb/training processes')
    base = {runtime: json.loads((folder / (CASE + '.' + label + '.json')).read_text())
            for runtime, label in (('torch', 'ref'), ('jittor', 'shim'))}
    platform = dict(base['jittor']['platform'],
                    oracle=dict(base['torch']['interpreter'], torch_npu_version='not-applicable'))
    platform_path = args.output / 'platform.json'
    platform_path.write_text(json.dumps(platform, indent=2) + '\n')
    record = collect(source, folder, platform_path, baseline, args.report,
                     [row['argv'] for row in commands])
    require(record['numerics']['checked_tensors'] == 1165 and record['training']['trainable'] == 144,
            'Original UNet numerical protocol was reduced')
    result = scorecard({'diffusers': source}, [record], baseline, ROOT)
    (args.output / 'scorecard.json').write_text(json.dumps(result, indent=2) + '\n')
    require(len(result['accepted']) == 1 and not result['rejected'], 'Shared auditor refused actual Diffusers evidence')
    require(framework_snapshot(ROOT) == frozen
            and not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'])
            and subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip() == baseline,
            'Source changed during Diffusers execution')
    (args.output / 'records.json').write_text(json.dumps({'schema_version': 2, 'inventories': {'diffusers': source},
                                                        'records': [record]}, indent=2) + '\n')
    print('DIFFUSERS_GATE_PASSED ' + json.dumps({'baseline': baseline, 'device': args.device,
          'workers': 6, 'array_pairs': 1165, 'level': 3}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
