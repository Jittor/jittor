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
        fresh = any(row.get('checkpoint_mode') == 'fresh-process' for row in executions)
        scope = 'world%d-%s' % (world, precision)
        artifact_scope = scope + ('-fresh-process' if fresh else '')
        case_label = scope + ('-fresh-process-checkpoint' if fresh else '-checkpoint')
        protocol = {'id': 'accelerate-checkpoint' if world == 1 else 'accelerate-fsdp2-full',
                    'world_size': world, 'precision': precision, 'ranks': list(range(world))}
        if fresh:
            from _accelerate_fresh_scorecard import protocol_spec
            protocol = protocol_spec(world, precision)
        expected = {(runtime, seed) for runtime in ('torch', 'jittor') for seed in (0, 7)}
        if (len(executions) != 4 or {(row['runtime'], row['seed']) for row in executions} != expected
                or any(row['returncode'] != 0 or row['status'] != 'passed' for row in executions)):
            records.append({'record_kind': 'protocol', 'repo': 'accelerate', 'status': 'blocked', 'level': None,
                            'case': case_label, 'registry_refs': [],
                            'jittor_commit': baseline, 'upstream': source['upstream'],
                            'protocol': protocol,
                            'reason': 'Missing or failed fresh required launcher; see immutable execution logs',
                            'evidence': {'artifacts': [artifact('execution', directory / 'execution.json')]}})
            continue
        artifacts = [artifact('execution', directory / 'execution.json')] if fresh else []
        values = {}
        metadata_paths = {}
        for runtime, label in (('torch', 'reference'), ('jittor', 'candidate')):
            for seed, scenario in ((0, 'base'), (7, 'perturb')):
                folder = directory / 'measurements' / runtime / ('world%d-%s-seed%d' % (world, precision, seed))
                case = 'single_checkpoint' if world == 1 else 'fsdp2_full'
                for rank in range(world):
                    stem = '%s_rank%d' % (case, rank)
                    path = (folder / 'produce' if fresh else folder) / (stem + '.json')
                    value = json.loads(path.read_text())
                    values[(runtime, seed, rank)] = value
                    metadata_paths[(runtime, seed, rank)] = path
                    prefix = '%s-%s-rank%d' % (label, scenario, rank)
                    for phase in (('produce', 'consume') if fresh else (None,)):
                        phase_folder = folder / phase if phase else folder
                        phase_prefix = prefix + '-' + phase if phase else prefix
                        artifacts.extend((artifact(phase_prefix + '-metadata', phase_folder / (stem + '.json')),
                                          artifact(phase_prefix + '-arrays', phase_folder / (stem + '.npz'))))
                    if fresh:
                        checkpoint = Path(value['result']['checkpoint_directory'])
                        artifacts.extend(artifact(prefix + '-checkpoint-' + name, checkpoint / name)
                                         for name in value['result']['checkpoint_sha256'])
        reference, candidate = values[('torch', 0, 0)], values[('jittor', 0, 0)]
        platform = dict(candidate['platform'], oracle={**reference['interpreter'], 'torch_npu_version': 'not-applicable'})
        platform_path = output / (artifact_scope + '-platform.json')
        write_json(platform_path, platform)
        artifacts.extend((artifact('reference', metadata_paths[('torch', 0, 0)]),
                          artifact('candidate', metadata_paths[('jittor', 0, 0)]),
                          artifact('platform', platform_path)))
        row = {'record_kind': 'protocol', 'repo': 'accelerate', 'case': case_label,
               'protocol': protocol,
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
        comparison = output / (artifact_scope + '-comparison.json')
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
