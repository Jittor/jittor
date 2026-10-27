"""Separate-process checkpoint phases for the existing Accelerate fixtures.

No model or optimizer mathematics lives here. A consumer reads only the public
Accelerate checkpoint; producer observations are checked by the outside auditor.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid


def fresh_phase():
    phase = os.environ.get('JITTOR_ACCELERATE_CHECKPOINT_PHASE')
    if phase not in (None, 'produce', 'consume'):
        raise ValueError('Unknown checkpoint process phase')
    return phase


def checkpoint_files(directory):
    directory = Path(directory)
    files = sorted(path for path in directory.rglob('*') if path.is_file())
    if not files or any(path.is_symlink() for path in directory.rglob('*')):
        raise ValueError('Checkpoint must contain real persisted files')
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def rng_snapshot(torch, device):
    import numpy as np

    def values(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): values(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [values(item) for item in value]
        return value

    return values({'python': random.getstate(), 'numpy': np.random.get_state(),
                   'torch_cpu': torch.get_rng_state(),
                   **({'torch_cuda': torch.cuda.get_rng_state_all()} if device == 'cuda' else {})})


def run_fresh_phase(capture, accelerator, update, state, sample, extra=None):
    phase = fresh_phase()
    if phase is None or not capture.enabled:
        raise ValueError('Separate-process phases require actual protocol capture')
    directory = Path(os.environ['JITTOR_ACCELERATE_CHECKPOINT_DIRECTORY']).resolve()
    output = Path(os.environ['JITTOR_ACCELERATE_PROTOCOL_OUTPUT']).resolve()
    if directory == output or output in directory.parents:
        raise ValueError('Checkpoint and phase observation directories must be separate')
    torch = capture.torch
    def sample_checkpoint_streams():
        values = sample()
        if capture.device == 'cuda':
            # torchrun exposes both allocated cards to each process; jtrun
            # exposes that rank's card. Exercise every visible generator.
            values = dict(values, torch_cuda_all=[
                torch.rand(3, device='cuda:%d' % index).detach().cpu().numpy().tolist()
                for index in range(torch.cuda.device_count())])
        return values

    result = {'phase': phase, 'process_nonce': str(uuid.uuid4()), 'pid': os.getpid(),
              'parent_pid': os.getppid(), 'observed_monotonic_ns': time.monotonic_ns(),
              'checkpoint_directory': str(directory),
              'process_environment': {key: os.environ.get(key) for key in
                  ('HOME', 'JITTOR_HOME', 'TMPDIR', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK',
                   'JITTOR_DIST_RENDEZVOUS_DIR', 'MASTER_ADDR', 'MASTER_PORT')}}
    if phase == 'produce':
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('Refusing to overwrite a persisted checkpoint')
        result['rng_initial'] = rng_snapshot(torch, capture.device)
        result['advanced_random'] = sample_checkpoint_streams()
        result['before_save'] = [update() for _ in range(3)]
        accelerator.save_state(str(directory), safe_serialization=False)
        accelerator.wait_for_everyone()
        result['checkpoint_sha256'] = checkpoint_files(directory)
        result['saved'] = state()
        result['rng_saved'] = rng_snapshot(torch, capture.device)
        result['expected_random'] = sample_checkpoint_streams()
        result['expected_resume'] = [update() for _ in range(2)]
    else:
        # A distinct stream makes failure to restore observable even when the
        # training fixture itself deliberately overwrites all initial weights.
        import numpy as np
        seed = 1831 + capture.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if capture.device == 'cuda':
            torch.cuda.manual_seed_all(seed)
        sample_checkpoint_streams()
        result['before_restore'] = state()
        result['rng_before'] = rng_snapshot(torch, capture.device)
        result['checkpoint_sha256'] = checkpoint_files(directory)
        accelerator.load_state(str(directory))
        result['after_restore'] = state()
        result['rng_restored'] = rng_snapshot(torch, capture.device)
        result['resumed_random'] = sample_checkpoint_streams()
        result['actual_resume'] = [update() for _ in range(2)]
    accelerator.wait_for_everyone()
    if checkpoint_files(directory) != result['checkpoint_sha256']:
        raise ValueError('Training or restoration modified the persisted checkpoint')
    if extra:
        result.update(extra())
    capture.finish(result, backend='none' if capture.world == 1 else torch.distributed.get_backend())
    print(json.dumps({'checkpoint_phase': phase, 'rank': capture.rank, 'pid': os.getpid()}), flush=True)
