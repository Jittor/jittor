"""Execute every original MM phase through the shared installed-source producer."""

from pathlib import Path
import subprocess
import sys
import tempfile

from _helpers.child_process import PYTHON
from _ecosystem_harness import REAL_TORCH_PYTHON, _run_command

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'


def compare_training(test, case, checkouts, wheels):
    import numpy as np

    sys.path.insert(0, str(SCRIPTS))
    from _openmmlab_component_contract import SPECS, compare_pair
    from registry_inventory import inventory

    sources = {repo: inventory(repo, path) for repo, path in checkouts.items()}
    baseline = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    repo = SPECS[case]['repo']
    with tempfile.TemporaryDirectory(prefix='jittor-openmmlab-installed-') as temporary:
        directory = Path(temporary)

        def run(runtime, seed, training=False, weights=None):
            python = REAL_TORCH_PYTHON if runtime == 'torch' else PYTHON
            output = directory / ('%s-seed%d-%s.npz' % ('training' if training else 'parity', seed, runtime))
            script = (SCRIPTS / 'openmmlab_training_probe.py' if training
                      else ROOT / 'compat/tests/torch/_ecosystem_runner.py')
            arguments = ['--root', str(ROOT), '--case', case, '--output', str(output), '--steps', '3',
                         '--learning-rate', '.01'] if training else [case, str(output), '--repeats', '1']
            arguments += ['--runtime', runtime, '--device', test.device, '--seed', str(seed)]
            if weights is not None:
                arguments += ['--weights', str(weights)]
            command = [python, str(SCRIPTS / 'component_measure.py'), '--root', str(ROOT), '--repo', repo,
                       '--checkout', str(checkouts[repo])]
            if repo == 'mmcv':
                command += ['--mmengine-checkout', str(checkouts['mmengine'])]
            command += ['--', str(script), *arguments]
            _run_command(python, case, command, result_marker='TRAINING_RESULT ' if training else 'ECOSYSTEM_RESULT ')
            return output

        reference = run('torch', 0)
        weights = reference.with_name(reference.stem + '.weights.npz')
        candidate = run('jittor', 0, weights=weights)
        base = compare_pair(reference, candidate, sources, baseline, case, 0, wheels, ROOT)
        for seed in (7, 19):
            paths = [run(runtime, seed, weights=weights) for runtime in ('torch', 'jittor')]
            pair = compare_pair(*paths, sources, baseline, case, seed, wheels, ROOT)
            test.assertEqual(pair['reference_result']['weights_sha256'], base['reference_result']['weights_sha256'])
            for original, changed in zip((reference, candidate), paths):
                with np.load(original) as a, np.load(changed) as b:
                    test.assertFalse(np.array_equal(a['__output__'], b['__output__']), 'Unchanged OpenMMLab perturbation output')
        paths = [run(runtime, 0, training=True, weights=weights) for runtime in ('torch', 'jittor')]
        pair = compare_pair(*paths, sources, baseline, case, 0, wheels, ROOT, training=True)
        test.assertEqual(pair['reference_result']['weights_sha256'], base['reference_result']['weights_sha256'])
        print('[openmmlab-installed/%s] %s: all original gradients, seeds 0/7/19, real train_step, %d trajectory fields'
              % (test.device, case, len(pair['rows'])))
