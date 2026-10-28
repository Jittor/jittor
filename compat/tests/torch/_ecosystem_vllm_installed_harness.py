"""Execute the original vLLM seeds with complete installed-source metadata."""

import json
import os
from pathlib import Path
import subprocess
import sys

from _helpers.child_process import PYTHON
from _ecosystem_harness import REAL_TORCH_PYTHON, _run_command

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'


def compare_primitives(test, checkout, wheel):
    from _helpers.artifacts import artifact_directory

    sys.path.insert(0, str(SCRIPTS))
    from _vllm_component_contract import CASE, compare_pair, validate_perturbation
    from registry_inventory import inventory

    source = inventory('vllm', checkout)
    baseline = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    output = os.environ.get('JITTOR_VLLM_EVIDENCE_DIR', '').strip()
    output = Path(output) / test.device / CASE if output else None
    with artifact_directory(ROOT, output=output, prefix='jittor-vllm-installed-') as directory:
        directory = Path(directory)

        def run(runtime, seed, weights=None):
            python = REAL_TORCH_PYTHON if runtime == 'torch' else PYTHON
            target = directory / ('seed%d-%s.npz' % (seed, runtime))
            arguments = [CASE, str(target), '--runtime', runtime, '--device', test.device,
                         '--seed', str(seed), '--repeats', '1']
            if weights is not None:
                arguments += ['--weights', str(weights)]
            command = [python, str(SCRIPTS / 'component_measure.py'), '--root', str(ROOT),
                       '--repo', 'vllm', '--checkout', str(checkout), '--',
                       str(ROOT / 'compat/tests/torch/_ecosystem_runner.py'), *arguments]
            _report, log = _run_command(python, CASE, command)
            target.with_suffix('.log').write_text(log)
            return target

        reference = run('torch', 0)
        weights = reference.with_name(reference.stem + '.weights.npz')
        candidate = run('jittor', 0, weights)
        base = compare_pair(reference, candidate, source, baseline, test.device, 0, wheel, ROOT)
        pairs = [base]
        for seed in (7, 19):
            paths = [run(runtime, seed, weights) for runtime in ('torch', 'jittor')]
            pair = compare_pair(*paths, source, baseline, test.device, seed, wheel, ROOT)
            for side, original, changed in zip(('reference_result', 'candidate_result'),
                                               (reference, candidate), paths):
                test.assertEqual(pair[side]['weights_sha256'], base[side]['weights_sha256'])
                validate_perturbation(original, changed)
            pairs.append(pair)
        result = {'case': CASE, 'status': 'passed', 'baseline': baseline, 'device': test.device,
                  'seeds': [0, 7, 19], 'checked_tensors': sum(len(pair['rows']) for pair in pairs),
                  'pairs': pairs}
        # Only complete, bound and numerically checked phases receive this marker.
        with (directory / 'complete-comparison.json').open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print('[vllm-installed/%s] seeds0/7/19: all four input gradients, 15 array pairs' % test.device)
