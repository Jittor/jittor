"""Run the original VERL component scopes with strict provenance and trajectories."""

from pathlib import Path
import subprocess
import tempfile

from _helpers.child_process import PYTHON
from _ecosystem_harness import REAL_TORCH_PYTHON, _run_command

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'


def compare_training(test, case, checkout, wheel):
    import sys
    import numpy as np

    sys.path.insert(0, str(SCRIPTS))
    from _verl_component_contract import compare_pair
    from registry_inventory import inventory

    source = inventory('verl', checkout)
    test.assertEqual(source['upstream']['commit'], 'e5a627cf55c94d85406556854c2c1a22db4ff5e7')
    baseline = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    with tempfile.TemporaryDirectory(prefix='jittor-verl-training-') as temporary:
        directory = Path(temporary)

        def run(runtime, seed, training=False, weights=None):
            python = REAL_TORCH_PYTHON if runtime == 'torch' else PYTHON
            output = directory / ('%s-seed%d-%s.npz' % ('training' if training else 'parity', seed, runtime))
            script = SCRIPTS / 'training_probe.py' if training else ROOT / 'compat/tests/torch/_ecosystem_runner.py'
            arguments = ['--root', str(ROOT), '--case', case, '--output', str(output), '--steps', '3',
                         '--learning-rate', '.001'] if training else [case, str(output), '--repeats', '1']
            arguments += ['--runtime', runtime, '--device', test.device, '--seed', str(seed)]
            if weights is not None:
                arguments += ['--weights', str(weights)]
            command = [python, str(SCRIPTS / 'component_measure.py'), '--root', str(ROOT), '--repo', 'verl',
                       '--checkout', str(checkout), '--', str(script), *arguments]
            _run_command(python, case, command, result_marker='TRAINING_RESULT ' if training else 'ECOSYSTEM_RESULT ')
            return output

        reference = run('torch', 0)
        weights = reference.with_name(reference.stem + '.weights.npz')
        candidate = run('jittor', 0, weights=weights)
        base = compare_pair(reference, candidate, source, baseline, case, 0, wheel, ROOT)
        for seed in (7, 19):
            paths = [run(runtime, seed, weights=weights) for runtime in ('torch', 'jittor')]
            pair = compare_pair(*paths, source, baseline, case, seed, wheel, ROOT)
            test.assertEqual(pair['reference_result']['weights_sha256'], base['reference_result']['weights_sha256'])
            for original, changed in zip((reference, candidate), paths):
                with np.load(original) as a, np.load(changed) as b:
                    test.assertFalse(np.array_equal(a['__output__'], b['__output__']), 'Unchanged VERL perturbation output')
        paths = [run(runtime, 0, training=True, weights=weights) for runtime in ('torch', 'jittor')]
        pair = compare_pair(*paths, source, baseline, case, 0, wheel, ROOT, training=True)
        test.assertEqual(pair['reference_result']['weights_sha256'], base['reference_result']['weights_sha256'])
        print('[verl-training/%s] %s: original complete gradients, seeds 0/7/19, three updates, %d trajectory fields'
              % (test.device, case, len(pair['rows'])))
