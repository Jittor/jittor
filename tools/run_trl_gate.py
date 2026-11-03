#!/usr/bin/env python3
"""Offline, independent-interpreter gate for 16/18 pinned official TRL Trainers.

Reuses the measured tiny producers and their original comparison budgets. No
packages are installed, no model is downloaded and no Trainer math is replaced.
All oracle processes run before the frozen candidate batch. An optional warm
JITTOR_HOME is exclusively leased for the entire run; never share it with tests.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'
METHODS = ('sft', 'dpo', 'kto', 'orpo', 'cpo', 'reward', 'ppo', 'rloo', 'gkd',
           'online_dpo', 'grpo', 'prm', 'bco', 'iterative_sft', 'nash_md', 'xpo')
TAIL = frozenset(METHODS[11:])
TRAINERS = ('SFTTrainer', 'DPOTrainer', 'KTOTrainer', 'ORPOTrainer', 'CPOTrainer',
            'RewardTrainer', 'PPOTrainer', 'RLOOTrainer', 'GKDTrainer', 'OnlineDPOTrainer',
            'GRPOTrainer', 'PRMTrainer', 'BCOTrainer', 'IterativeSFTTrainer', 'NashMDTrainer', 'XPOTrainer')
PINS = {'trl': '0.21.0', 'transformers': '4.56.2', 'accelerate': '1.15.0',
        'datasets': '3.6.0', 'numpy': '1.26.4', 'peft': '0.17.1'}
REVISION = '46d09bd2408f17605409fb3ee8ba12705add7faa'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def external(path):
    path = Path(path).expanduser().resolve()
    require(path != ROOT and ROOT not in path.parents, 'Output/cache/upstream must be outside the source checkout')
    return path


def fingerprint():
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard',
                                     'src', 'backends', 'python/jittor', 'compat'], cwd=ROOT, text=True).splitlines()
    return {name: digest(ROOT/name) for name in sorted(set(names))
            if '/tests/' not in name and (ROOT/name).is_file()}


def common_sources():
    paths = [Path(__file__), ROOT/'compat/tests/torch/_ecosystem_runner.py', *SCRIPTS.glob('trl_*.py')]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def expected_tools(method):
    require(method in METHODS, 'Unknown Trainer selector')
    if method in TAIL:
        producer, case = 'trl_tail_probe.py', 'trl_tail_cases.py'
    else:
        producer = 'trl_mainstream_probe.py'
        case = 'trl_offline_cases.py' if method in METHODS[:6] else 'trl_mainstream_online.py'
    return {producer, case, '_ecosystem_runner.py', 'trl_rl_probe.py', 'trl_sft_probe.py'}


def environment(output, runtime, device, home, cuda_device=None):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('JT_', 'JITTOR_', 'TRL_', 'PYTEST_')) or key in (
                'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'REAL_TORCH_SITE', 'REAL_TORCH_PYTHON',
                'cache_name', 'use_cuda', 'nvcc_path', 'cc_path', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK',
                'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key, None)
    private = output/'contexts'/runtime
    for key, folder in (('HOME', 'home'), ('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'xdg'),
                        ('HF_HOME', 'hf'), ('TORCH_HOME', 'torch'),
                        ('JITTOR_TORCH_CACHE_ROOT', 'shim-cache'), ('JITTOR_TORCH_RUNTIME_ROOT', 'shim-runtime')):
        directory = private/folder
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    env.update(JITTOR_HOME=str(home), PYTHONPATH=str(ROOT/'python') if runtime == 'jittor' else '',
               JITTOR_TORCH_SHIM='1' if runtime == 'jittor' else '0', PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', JITTOR_TORCH_KEEP_HOME='1', JITTOR_TORCH_KEEP_TMPDIR='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', USE_TF='0', USE_FLAX='0',
               JT_USE_CUDA=str(int(device == 'cuda')), use_cuda=str(int(device == 'cuda')),
               JT_BUILD_USE_MKL='0', JT_BUILD_USE_MPI='0', JT_BUILD_USE_NCCL='0', JT_BUILD_USE_CUTT='0',
               DISABLE_MULTIPROCESSING='1', JT_USE_PARALLEL_OP_COMPILER='0', JT_CUDA_KERNEL_MATH='strict',
               JITTOR_ECOSYSTEM_TF32='0', JITTOR_ECOSYSTEM_CUDNN_BENCHMARK='0',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', gdb_path='')
    if device == 'cpu':
        env.update(CUDA_VISIBLE_DEVICES='', nvcc_path='', JT_BUILD_NVCC_PATH='')
    else:
        require(cuda_device and ',' not in cuda_device, 'CUDA requires exactly one explicit --cuda-device')
        nvcc = os.environ.get('JT_BUILD_NVCC_PATH') or shutil.which('nvcc')
        require(nvcc, 'CUDA requires NVCC')
        env.update(CUDA_VISIBLE_DEVICES=cuda_device, nvcc_path=nvcc, JT_BUILD_NVCC_PATH=nvcc)
    return env


PREFLIGHT = r'''
import importlib.metadata,json,sys
from pathlib import Path
runtime,output=sys.argv[1:]
versions={name:importlib.metadata.version(name) for name in ('trl','transformers','accelerate','datasets','numpy','peft')}
result=dict(runtime=runtime,versions=versions,python=list(sys.version_info[:2]),prefix=sys.prefix,
            base_prefix=sys.base_prefix,executable=sys.executable)
if runtime=='torch':
 import torch
 result.update(shim=hasattr(torch,'_torch_compat_install_context'),torch_version=str(torch.__version__),
               torch_origin=str(torch.__file__),binary=getattr(getattr(torch,'_C',None),'__file__',None))
Path(output).write_text(json.dumps(result,indent=2)+'\n')
'''
WARMUP = r'''
import json,sys
from pathlib import Path
import jittor as jt
import torch
assert Path(jt.__file__).resolve()==Path(sys.argv[1])/'python/jittor/__init__.py'
assert hasattr(torch,'_torch_compat_install_context')
Path(sys.argv[2]).write_text(json.dumps(dict(status='complete',shim=True,source=str(jt.__file__)))+'\n')
'''


def validate_environment(info, runtime):
    require(info['runtime'] == runtime and info.get('python') == [3, 11], 'Pinned gate requires Python 3.11')
    require(info.get('versions') == PINS, 'Downstream dependency pin mismatch')
    require(info['prefix'] != info['base_prefix'], 'Use a private virtual environment')
    require(ROOT not in Path(info['prefix']).resolve().parents, 'Environments must be outside the source checkout')
    if runtime == 'torch':
        require(info.get('shim') is False, 'Oracle resolved the Jittor shim')
        require(info.get('torch_version', '').split('+')[0] == '2.8.0', 'Oracle requires genuine Torch 2.8.0')
        binary = info.get('binary')
        require(binary and Path(binary).is_file() and Path(binary).suffix in ('.so', '.pyd'), 'Oracle has no genuine Torch binary')
        prefix = Path(info['prefix']).resolve()
        require(prefix in Path(binary).resolve().parents and prefix in Path(info['torch_origin']).resolve().parents,
                'Oracle Torch escaped its private environment')
        info['binary_sha256'] = digest(binary)


@contextmanager
def lease(home):
    import fcntl
    home.mkdir(parents=True, exist_ok=True)
    with (home/'.trl-gate.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('JITTOR_HOME is leased by another TRL gate') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def command(argv, output, label, env, *, allow_restart=False):
    log = output/'logs'/(label+'.log')
    row = dict(command=[str(value) for value in argv], start=time.time(), log=str(log.relative_to(output)))
    try:
        with log.open('x') as stream:
            result = subprocess.run(row['command'], cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=1800)
        row['exit_code'] = result.returncode
    except subprocess.TimeoutExpired:
        row['exit_code'] = 'timeout'
        raise
    finally:
        row['end'] = time.time()
        write(output/(label+'.command.json'), row)
    if allow_restart and result.returncode == 3:
        require('jit_utils was rebuilt' in log.read_text(), 'Unexpected warm-up exit 3')
        return False
    require(result.returncode == 0, 'Command failed: '+label+'; see '+str(log))
    return True


def audit_pair(reference_path, candidate_path, method, device, steps, oracle_proof=None):
    """Portable version of the current-16 full-payload audit; no tensor runtime import."""
    import numpy as np
    sys.path.insert(0, str(SCRIPTS))
    from trl_rl_probe import array_hash, compare_arrays
    from trl_final_logits_probe import checked_array
    from trl_full_collect import aggregate, compare_recorded
    paths = [Path(reference_path), Path(candidate_path)]
    a, b = [json.loads((path/'manifest.json').read_text()) for path in paths]
    require(a['runtime'] == 'torch' and b['runtime'] == 'jittor', 'Require independent oracle/candidate')
    require(a.get('oracle_binary') and Path(a['oracle_binary']).is_file(), 'Oracle binary proof missing')
    require(a.get('torch_version', '').split('+')[0] == '2.8.0', 'Wrong actual oracle Torch')
    if oracle_proof is not None:
        validate_environment(oracle_proof.copy(), 'torch')
        require(a['oracle_binary'] == oracle_proof['binary'] and a['torch_version'] == oracle_proof['torch_version']
                and digest(a['oracle_binary']) == oracle_proof['binary_sha256'], 'Oracle binary changed after preflight')
    for path, manifest in zip(paths, (a, b)):
        source = manifest['trl_source']
        require(source.get('source_verified') is True and source['version'] == PINS['trl']
                and source['upstream']['revision'] == REVISION, 'Unverified official TRL source')
        require(all(manifest['versions'].get(name) == PINS[name] for name in ('trl', 'transformers', 'accelerate')),
                'Wrong actual Trainer dependencies')
        require(manifest['status'] == 'complete' and manifest['fallback_count'] == 0, 'Incomplete or fallback training')
        require(manifest['device'] == device and manifest['protocol']['algorithm'] == method, 'Wrong actual method/device')
        require(manifest['protocol']['steps'] == steps and type(manifest['optimizer_steps']) is int
                and manifest['optimizer_steps'] == steps and manifest['case']['expected_optimizer_steps'] == steps,
                'Missing actual optimizer calls')
        norms, updates = manifest['gradient_norms'], manifest['parameter_updates']
        require(len(norms) == len(updates) == steps and all(math.isfinite(x) and x >= 0 for x in norms)
                and all(math.isfinite(x['l2']) and x['l2'] >= 0 and x['changed_elements'] >= 0 for x in updates)
                and any(norms) and any(x['changed_elements'] for x in updates), 'Missing finite nonzero training signal')
        names = {name[len('parameter/'):]: value['shape']
                 for name, value in manifest['initial']['policy']['tensors'].items() if name.startswith('parameter/')}
        if method == 'ppo':
            names = {'policy.'+name: value for name, value in names.items()}
            names.update({'value_model.'+name[len('parameter/'):]: value['shape']
                          for name, value in manifest['initial']['value']['tensors'].items() if name.startswith('parameter/')})
        require(names and names == manifest['trainable_parameters'], 'Incomplete full trainable inventory')
        for role, info in manifest['initial'].items():
            artifact = path/'initial'/(role+'.npz')
            require(digest(artifact) == info['artifact_sha256'], 'Initial archive hash mismatch')
            with np.load(artifact, allow_pickle=False) as payload:
                require(set(payload.files) == set(info['tensors']), 'Initial tensor inventory missing')
                for key in payload.files:
                    value, spec = payload[key], info['tensors'][key]
                    require(np.isfinite(value).all() and list(value.shape) == spec['shape']
                            and str(value.dtype) == spec['dtype'] and array_hash(value) == spec['sha256'], 'Initial tensor mismatch')
        for step in range(steps):
            prefix = 'step/%d/' % step
            for kind in ('gradient', 'updated_parameter'):
                expected = {prefix+kind+'/'+name+'/0' for name in names}
                require(expected == {key for key in manifest['arrays'] if key.startswith(prefix+kind+'/')}, 'Missing full '+kind+' inventory')
                for name, shape in names.items():
                    require(manifest['arrays'][prefix+kind+'/'+name+'/0']['shape'] == shape, 'Trained tensor shape changed')
            for role in ('policy', 'value') if method == 'ppo' else ('policy',):
                require(any(key.startswith(prefix+'embedding_gradient/'+role+'/') for key in manifest['arrays']),
                        'Missing input embedding gradient: '+role)
            require(any(key.startswith((prefix+'forward/input/', prefix+'forward/positional/'))
                        for key in manifest['arrays']), 'Missing training forward inputs')
            require(any(key.startswith(prefix+'forward/output/') for key in manifest['arrays']), 'Missing training forward outputs')
            require(prefix+'loss/0' in manifest['arrays'], 'Missing scalar loss')
            require(manifest['arrays'][prefix+'loss/0']['shape'] == [], 'Expected scalar loss, not a vector')
        for index in range(10):
            for name in ('input_ids', 'logits') + (() if method in ('reward', 'prm') else ('next_logits',)):
                require('step/%d/final_eval/%d/%s/0' % (steps, index, name) in manifest['arrays'], 'Missing final evaluation inventory')
            require(manifest['arrays']['step/%d/final_eval/%d/input_ids/0' % (steps, index)]['exact'] is True,
                    'Final evaluation inputs must be exact')
        require(set(manifest.get('tool_sources', {})) == expected_tools(method), 'Incomplete producer source inventory')
        for name, sha in manifest['tool_sources'].items():
            source = ROOT/'compat/tests/torch/_ecosystem_runner.py' if name == '_ecosystem_runner.py' else SCRIPTS/name
            require(digest(source) == sha, 'Changed producer source')
    for key in ('protocol', 'versions', 'trl_source', 'tool_sources', 'trainable_parameters', 'case', 'generation_calls'):
        require(a[key] == b[key], 'Pair protocol/source mismatch: '+key)
    require(set(a['initial']) == set(b['initial']) and all(a['initial'][role]['tensors'] == b['initial'][role]['tensors']
                                                        for role in a['initial']), 'Initial model roles differ')
    require(set(a['arrays']) == set(b['arrays']) == set(b['comparison']), 'Missing observation/comparison')
    if method in ('nash_md', 'xpo'):
        require(len(b['generation_calls']) == 2*steps, 'Incomplete online generation calls')
    if method == 'nash_md':
        require(sum(call['role'] == 'geometric_mixture' for call in b['generation_calls']) == steps, 'Missing mixture generation')
    rows = {}
    for key, info in a['arrays'].items():
        require(info['exact'] == b['arrays'][key]['exact'], 'Changed exact comparison policy')
        row = compare_arrays(checked_array(paths[0], info), checked_array(paths[1], b['arrays'][key]), exact=info['exact'])
        compare_recorded(row, b['comparison'][key])
        if info['exact']:
            require(row['passed'], 'Controlled inputs or generation differ')
        rows[key] = row
    passed = all(row['passed'] for row in rows.values())
    require(passed == b['numerical_passed'], 'Completion flag disagrees with recomputed comparisons')
    return dict(method=method, device=device, steps=steps, numerical_passed=passed, observations=len(rows),
                trained_tensors=len(b['trainable_parameters']), fallback_count=0,
                gradients=aggregate(row for key, row in rows.items() if '/gradient/' in key),
                parameters=aggregate(row for key, row in rows.items() if '/updated_parameter/' in key),
                losses=aggregate(row for key, row in rows.items() if '/loss/' in key),
                positive_gradient_steps={side: sum(value > 0 for value in manifest['gradient_norms'])
                                         for side, manifest in (('torch', a), ('jittor', b))},
                nonzero_parameter_update_steps={side: sum(value['changed_elements'] > 0 for value in manifest['parameter_updates'])
                                                for side, manifest in (('torch', a), ('jittor', b))},
                reference_manifest=digest(paths[0]/'manifest.json'), candidate_manifest=digest(paths[1]/'manifest.json'))


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('jittor-python', 'oracle-python', 'upstream', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--steps', type=int, choices=range(1, 11), default=3)
    parser.add_argument('--jittor-home', type=Path, help='Exclusively leased prepared cache; one process at a time')
    parser.add_argument('--cuda-device', help='One explicit physical CUDA ordinal or UUID')
    return parser.parse_args()


def run(args):
    output, upstream = external(args.output), external(args.upstream)
    require(not output.exists() and upstream != output and upstream not in output.parents, 'Use a fresh external output directory')
    interpreters = {runtime: path.expanduser().absolute() for runtime, path in
                    (('torch', args.oracle_python), ('jittor', args.jittor_python))}
    require(all(path.is_file() for path in interpreters.values()), 'Interpreter does not exist')
    require(interpreters['torch'] != interpreters['jittor'], 'Use two private interpreters')
    require(args.device != 'cuda' or args.cuda_device and ',' not in args.cuda_device, 'CUDA requires --cuda-device')
    home = external(args.jittor_home) if args.jittor_home else output/'jittor-home'
    require(home != upstream and upstream not in home.parents, 'Cache must not pollute upstream')
    sys.path.insert(0, str(SCRIPTS))
    from trl_inventory import inventory
    inv = inventory(upstream, REVISION, selected=frozenset(TRAINERS))
    require(inv['denominator']['count'] == 18 and inv['evaluation_subset']['count'] == 16, 'Pinned Trainer inventory changed')
    output.mkdir(parents=True, exist_ok=False)
    (output/'logs').mkdir()
    (output/'audits').mkdir()
    write(output/'inventory.json', inv)
    shared = common_sources()
    write(output/'producer-sources.json', shared)
    environments = {runtime: environment(output, runtime, args.device, home, args.cuda_device) for runtime in interpreters}
    summary = dict(status='running', selected=16, denominator=18, revision=REVISION, device=args.device, steps=args.steps,
                   scope='Tiny random FP32 Qwen3; unchanged official text Trainer math; no diffusion/full-checkpoint/quality/speed acceptance', rows=[])
    write(output/'summary.json', summary)
    try:
        with lease(home):
            proofs = {}
            for runtime, python in interpreters.items():
                proof = output/(runtime+'-environment.json')
                command([python, '-c', PREFLIGHT, runtime, proof], output, runtime+'-preflight', environments[runtime])
                proofs[runtime] = json.loads(proof.read_text())
                validate_environment(proofs[runtime], runtime)
                write(proof, proofs[runtime])
            require(proofs['torch']['prefix'] != proofs['jittor']['prefix'], 'Interpreters share one environment')
            warmed = command([interpreters['jittor'], '-c', WARMUP, ROOT, output/'warmup.json'],
                             output, 'warmup', environments['jittor'], allow_restart=True)
            if not warmed:
                command([interpreters['jittor'], '-c', WARMUP, ROOT, output/'warmup.json'], output, 'warmup-restart', environments['jittor'])
            runtime_sources = None
            for runtime in ('torch', 'jittor'):
                if runtime == 'jittor':
                    runtime_sources = fingerprint()
                    write(output/'runtime-sources.json', runtime_sources)
                for method in METHODS:
                    require(common_sources() == shared, 'Producer sources changed during gate')
                    if runtime == 'jittor':
                        require(fingerprint() == runtime_sources, 'Candidate runtime sources changed')
                    trace = output/'measurements'/method/runtime/'trace'
                    trace.parent.mkdir(parents=True)
                    producer = SCRIPTS/('trl_tail_probe.py' if method in TAIL else 'trl_mainstream_probe.py')
                    argv = [interpreters[runtime], '-u', producer, '--root', ROOT, '--upstream', upstream, '--output', trace,
                            '--runtime', runtime, '--device', args.device, '--algorithm', method, '--steps', str(args.steps)]
                    if runtime == 'jittor':
                        argv += ['--reference', output/'measurements'/method/'torch/trace']
                    command(argv, output, method+'-'+runtime, environments[runtime])
                    require(common_sources() == shared, 'Producer sources changed during measurement')
                    if runtime == 'jittor':
                        require(fingerprint() == runtime_sources, 'Runtime source changed during measurement')
                        request = dict(reference_path=str(output/'measurements'/method/'torch/trace'), candidate_path=str(trace),
                                       method=method, device=args.device, steps=args.steps, oracle_proof=proofs['torch'])
                        request_path = output/'audits'/(method+'-request.json')
                        write(request_path, request)
                        command([interpreters['torch'], Path(__file__), '--_audit', request_path], output,
                                method+'-audit', environments['torch'])
                        row = json.loads(request_path.with_name(method+'.json').read_text())
                        summary['rows'].append(row)
                        write(output/'summary.json', summary)
                        require(row['numerical_passed'], 'Strict numerical comparison failed: '+method)
            require(len(summary['rows']) == 16, 'Missing required Trainer rows')
            summary['status'] = 'passed'
            write(output/'summary.json', summary)
    except Exception as error:
        summary.update(status='failed', error=str(error))
        write(output/'summary.json', summary)
        raise
    return 0


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--_audit':
        request_path = Path(sys.argv[2])
        request = json.loads(request_path.read_text())
        write(request_path.with_name(request['method']+'.json'), audit_pair(**request))
        return 0
    return run(options())


if __name__ == '__main__':
    raise SystemExit(main())
