"""Use the shared child-process contract for the real Swift Trainer protocol."""

from pathlib import Path
import sys
import tempfile

from _helpers.child_process import PYTHON
from _ecosystem_harness import REAL_TORCH_PYTHON, _run_command

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'
PROBE = SCRIPTS / 'swift_training_probe.py'
CASE = 'ms_swift_trainer_full_llama'


def compare_training(test, upstream, wheel):
    sys.path.insert(0, str(SCRIPTS))
    from registry_inventory import git, inventory
    from _swift_training_contract import compare_pair, read
    source = inventory('ms-swift', upstream)
    baseline = git(ROOT, 'rev-parse', 'HEAD')
    with tempfile.TemporaryDirectory(prefix='jittor-swift-trainer-') as directory:
        root = Path(directory)
        model = root / 'model'
        prepared, _ = _run_command(REAL_TORCH_PYTHON, CASE,
            [REAL_TORCH_PYTHON, str(PROBE), 'prepare', '--root', str(ROOT), '--model', str(model)])
        test.assertEqual(prepared.get('action'), 'prepared')
        paths = []
        for seed in (0, 7, 19):
            pair = []
            for python, runtime in ((REAL_TORCH_PYTHON, 'torch'), (PYTHON, 'jittor')):
                output = root / ('seed%d-%s.npz' % (seed, runtime))
                command = [python, str(PROBE), 'run', '--root', str(ROOT), '--model', str(model),
                    '--runtime', runtime, '--device', test.device, '--upstream', str(upstream),
                    '--wheel', str(wheel), '--output', str(output), '--seed', str(seed)]
                report, _ = _run_command(python, CASE, command)
                test.assertEqual(read(output.with_suffix('.json')), report)
                pair.append(output)
            comparison = compare_pair(*pair, source, baseline, seed=seed)
            test.assertEqual(comparison['status'], 'passed')
            paths.append(pair)
        import numpy as np
        for side in (0, 1):
            for alternative in paths[1:]:
                with np.load(paths[0][side]) as original, np.load(alternative[side]) as changed:
                    key = 'step/0/observed_input::input_ids'
                    test.assertNotEqual(read(paths[0][side].with_suffix('.json'))['observed_inputs'][key],
                                        read(alternative[side].with_suffix('.json'))['observed_inputs'][key])
                    test.assertFalse(np.array_equal(original['step/0/output'], changed['step/0/output']))
                    for name in original.files:
                        if name.startswith('initial::'):
                            np.testing.assert_array_equal(original[name], changed[name])
