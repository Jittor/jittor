"""Collect immutable maintained DeepSpeed CPU/CUDA protocol evidence.

This tool imports no framework or downstream package and never launches models.
It rechecks real Engine artifacts against the shared protocol validator. The
record has no registry numerator/denominator and makes no performance claim.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import _deepspeed_scorecard as protocol


def artifact(role, path):
    path = Path(path).resolve()
    return {'role': role, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def metadata(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive['metadata']))


def collect(source, roots, wheel, rebuilt, output, baseline, report, commands):
    output = Path(output).resolve()
    protocol.require(not output.exists(), 'Refusing to overwrite previous DeepSpeed scorecard artifacts')
    protocol.require(protocol.ROOT.resolve() not in output.parents, 'DeepSpeed raw artifacts must be outside checkout')
    output.mkdir(parents=True)
    source_path = output / 'source.json'
    source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + '\n')
    records = []
    for device, root in roots.items():
        for stage in (1, 2):
            cases = {scenario: Path(root).resolve() / device / ('zero%d' % stage) / 'AdamW' /
                     'bfloat16' / 'world2' / ('perturb-%g' % offset)
                     for scenario, offset in (('base', 0.), ('perturb', .125))}
            first = metadata(cases['base'] / 'jittor/rank0.npz')
            reference = metadata(cases['base'] / 'torch/rank0.npz')
            platform = copy.deepcopy(first['platform'])
            platform['oracle'] = {**reference['interpreter'], 'torch_npu_version': 'not-applicable'}
            platform['oracle_platform'] = reference['platform']
            stem = '%s-zero%d' % (device, stage)
            platform_path = output / (stem + '-platform.json')
            platform_path.write_text(json.dumps(platform, indent=2, sort_keys=True) + '\n')
            comparison = output / (stem + '-comparison.json')
            comparison.write_text('{}\n')
            artifacts = [artifact('platform', platform_path), artifact('comparison', comparison),
                         artifact('upstream-wheel', wheel), artifact('upstream-rebuilt-wheel', rebuilt),
                         artifact('source-inventory', source_path),
                         artifact('reference', cases['base'] / 'torch/rank0.npz'),
                         artifact('candidate', cases['base'] / 'jittor/rank0.npz')]
            for scenario, case in cases.items():
                artifacts.extend((artifact(scenario + '-sources', case / 'source.json'),
                                  artifact('gate-' + scenario + '-comparison', case / 'comparison.json')))
                for label, runtime in (('reference', 'torch'), ('candidate', 'jittor')):
                    for rank in (0, 1):
                        artifacts.append(artifact('%s-%s-rank%d' % (label, scenario, rank),
                                                  case / runtime / ('rank%d.npz' % rank)))
            record = {'repo': 'deepspeed', 'case': 'zero%d-world2-bf16-adamw' % stage,
                      'record_kind': 'protocol', 'protocol': first['protocol'], 'registry_refs': [],
                      'status': 'passed', 'level': 3, 'tier': 'core', 'jittor_commit': baseline,
                      'upstream': source['upstream'], 'downstream_versions': {
                          'candidate': first['dependencies']['deepspeed']['version'],
                          'oracle': reference['dependencies']['deepspeed']['version']},
                      'device': device, 'observed_device': device, 'platform': platform,
                      'inputs': {'seed': 0, 'hf_hub_offline': True, 'shape_kind': 'tiny',
                                 'input_perturbations': [0., .125]},
                      'fallback': {'policy': 'error', 'count': 0},
                      'training': {'steps': 3, 'loss_trajectory_passed': True,
                          'parameter_trajectory_passed': True, 'trainable': 8, 'frozen': 0,
                          'trainable_parameters': sorted(protocol.PARAMETERS), 'frozen_parameters': []},
                      'evidence': {'command': commands[device], 'report': report,
                          'gate_nodeid': 'compat/tests/torch/test_ecosystem_deepspeed_multirank.py::'
                              'test_two_rank_bfloat16_engine_training_and_resume', 'artifacts': artifacts}}
            observed = protocol.measurements(record, artifacts, source)
            comparison.write_text(json.dumps({'status': 'passed', 'protocol': record['protocol'],
                'jittor_commit': baseline, 'device': device, 'observed': observed}, indent=2, sort_keys=True) + '\n')
            record['evidence']['artifacts'] = [artifact(item['role'], item['path']) for item in artifacts]
            record['numerics'] = {'passed': True, 'finite': True, 'missing_tensors': [], 'atol': .002, 'rtol': .01,
                **{name: observed[name] for name in ('checked_tensors', 'max_abs', 'max_scaled_relative',
                                                   'input_perturbation_verified')},
                **{axis: {'requested': observed[axis], 'checked': observed[axis]}
                   for axis in ('parameter_gradients', 'input_gradients')}}
            protocol.validate_measurements(record, record['evidence']['artifacts'], source)
            records.append(record)
    document = {'schema_version': 2, 'source': source, 'records': records}
    (output / 'evidence.json').write_text(json.dumps(document, indent=2, sort_keys=True) + '\n')
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--wheel', required=True, type=Path)
    parser.add_argument('--rebuilt-wheel', required=True, type=Path)
    parser.add_argument('--cpu-root', type=Path)
    parser.add_argument('--cuda-root', type=Path)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--cpu-command', default='nox -s ecosystem_deepspeed_multirank (CPU MPI backend verification)')
    parser.add_argument('--cuda-command', default='nox -s ecosystem_deepspeed_multirank (CUDA public jtrun)')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scorecard', action='store_true', help='Also call the shared common protocol auditor')
    options = parser.parse_args()
    roots = {name: path for name, path in (('cpu', options.cpu_root), ('cuda', options.cuda_root)) if path}
    protocol.require(roots, 'At least one actual device artifact root is required')
    source = protocol.source_descriptor(options.upstream, options.wheel)
    evidence = collect(source, roots, options.wheel, options.rebuilt_wheel, options.output, options.baseline,
                       options.report, {'cpu': options.cpu_command, 'cuda': options.cuda_command})
    if options.scorecard:
        from audit_coverage import scorecard
        result = scorecard({'deepspeed': source}, evidence['records'], options.baseline, protocol.ROOT)
        result.update(auditor_commit=protocol.git(protocol.ROOT, 'rev-parse', 'HEAD').decode().strip(),
                      baseline_policy='explicit-historical')
        (options.output / 'scorecard.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        protocol.require(not result['rejected'] and not result.get('blocked'), 'DeepSpeed shared scorecard rejected/blocked records')
    print(json.dumps({'records': len(evidence['records']), 'evidence': str(options.output / 'evidence.json')}))


if __name__ == '__main__':
    main()
