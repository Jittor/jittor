"""Collect maintained Accelerate protocol evidence without registry numerators.

Every row is recomputed from newly source-bound rank JSON/NPZ observations.
Failed independent agreement remains failed even when own-runtime resume was
exact. CPU world-one is never used as a CPU world-two counterpart.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from registry_inventory import ROOT, git, write_json
from audit_coverage import scorecard
from _accelerate_scorecard import measurements

sys.path.insert(0, str(ROOT / 'compat/tests/torch'))
from _accelerate_protocol import source_descriptor  # noqa: E402


def artifact(role, path):
    path = Path(path).resolve()
    return {'role': role, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def collect(directory, source, baseline, report, command, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    execution = json.loads((directory / 'execution.json').read_text())
    groups = sorted({(row['world_size'], row['precision']) for row in execution['results']})
    records = []
    for world, precision in groups:
        executions = [row for row in execution['results'] if (row['world_size'], row['precision']) == (world, precision)]
        expected = {(runtime, seed) for runtime in ('torch', 'jittor') for seed in (0, 7)}
        if (len(executions) != 4 or {(row['runtime'], row['seed']) for row in executions} != expected
                or any(row['returncode'] != 0 or row['status'] != 'passed' for row in executions)):
            records.append({'record_kind': 'protocol', 'repo': 'accelerate', 'status': 'blocked', 'level': None,
                            'case': 'world%d-%s-checkpoint' % (world, precision), 'registry_refs': [],
                            'jittor_commit': baseline, 'upstream': source['upstream'],
                            'protocol': {'id': 'accelerate-checkpoint' if world == 1 else 'accelerate-fsdp2-full',
                                         'world_size': world, 'precision': precision, 'ranks': list(range(world))},
                            'reason': 'Missing or failed fresh required launcher; see immutable execution logs',
                            'evidence': {'artifacts': [artifact('execution', directory / 'execution.json')]}})
            continue
        artifacts = []
        values = {}
        for runtime, label in (('torch', 'reference'), ('jittor', 'candidate')):
            for seed, scenario in ((0, 'base'), (7, 'perturb')):
                folder = directory / 'measurements' / runtime / ('world%d-%s-seed%d' % (world, precision, seed))
                case = 'single_checkpoint' if world == 1 else 'fsdp2_full'
                for rank in range(world):
                    stem = '%s_rank%d' % (case, rank)
                    path = folder / (stem + '.json')
                    value = json.loads(path.read_text())
                    values[(runtime, seed, rank)] = value
                    prefix = '%s-%s-rank%d' % (label, scenario, rank)
                    artifacts.extend((artifact(prefix + '-metadata', path), artifact(prefix + '-arrays', folder / (stem + '.npz'))))
        reference, candidate = values[('torch', 0, 0)], values[('jittor', 0, 0)]
        platform = dict(candidate['platform'], oracle={**reference['interpreter'], 'torch_npu_version': 'not-applicable'})
        scope = 'world%d-%s' % (world, precision)
        platform_path = output / (scope + '-platform.json')
        write_json(platform_path, platform)
        artifacts.extend((artifact('reference', directory / 'measurements/torch' / ('world%d-%s-seed0' % (world, precision)) / (case + '_rank0.json')),
                          artifact('candidate', directory / 'measurements/jittor' / ('world%d-%s-seed0' % (world, precision)) / (case + '_rank0.json')),
                          artifact('platform', platform_path)))
        row = {'record_kind': 'protocol', 'repo': 'accelerate', 'case': scope + '-checkpoint',
               'protocol': {'id': 'accelerate-checkpoint' if world == 1 else 'accelerate-fsdp2-full',
                            'world_size': world, 'precision': precision, 'ranks': list(range(world))},
               'status': 'passed', 'level': None, 'tier': 'core', 'registry_refs': [],
               'jittor_commit': baseline, 'upstream': source['upstream'],
               'downstream_versions': {'candidate': candidate['downstream_distribution']['version'],
                                       'oracle': reference['downstream_distribution']['version']},
               'device': candidate['device'], 'observed_device': candidate['device'], 'platform': platform,
               'inputs': {'seed': 0, 'perturb_seed': 7, 'hf_hub_offline': True, 'shape_kind': 'tiny'},
               'fallback': {'policy': candidate['fallback_policy'], 'count': candidate['fallback_count']},
               'evidence': {'command': command, 'report': report,
                            'gate_nodeid': ('compat/tests/torch/test_optimizer_checkpoint_state.py::TestOptimizerCheckpoint::test_accelerate_save_load_continues_model_optimizer_scheduler'
                                           if world == 1 else 'compat/tests/torch/test_accelerate_distributed_integration.py::test_fsdp2_full_model_optimizer_checkpoint_resume[' + precision + ']'),
                            'artifacts': artifacts}}
        actual = measurements(row, artifacts, source)
        comparison = output / (scope + '-comparison.json')
        write_json(comparison, actual)
        artifacts.append(artifact('comparison', comparison))
        row.update(actual)
        row['status'] = 'passed' if actual['numerics']['passed'] else 'failed'
        row['level'] = 3 if row['status'] == 'passed' else None
        records.append(row)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--command', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if git(ROOT, 'rev-parse', args.baseline + '^{commit}') != args.baseline:
        parser.error('Expected an existing full measurement source commit')
    source = source_descriptor(args.upstream)
    rows = []
    for index, directory in enumerate(args.run):
        rows.extend(collect(directory, source, args.baseline, args.report, args.command, args.output / ('run%d' % index)))
    write_json(args.output / 'records.json', {'schema_version': 2, 'records': rows})
    value = scorecard({'accelerate': source}, rows, args.baseline)
    write_json(args.output / 'scorecard.json', value)
    print(json.dumps({'accepted': len(value['accepted']), 'rejected': value['rejected'], 'protocols': value.get('protocols', [])}))
    return int(bool(value['rejected']))


if __name__ == '__main__':
    raise SystemExit(main())
