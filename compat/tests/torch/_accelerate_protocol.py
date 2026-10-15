"""Actual-process observations for the maintained Accelerate checkpoint gates.

Inactive in ordinary unit runs. No model/optimizer implementation lives here.
"""

import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(*args):
    return subprocess.check_output(list(args), text=True).strip()


def cpu_description():
    info = Path('/proc/cpuinfo')
    if info.is_file():
        names = [line.split(':', 1)[1].strip() for line in info.read_text().splitlines()
                 if line.startswith('model name')]
        if names:
            return names[0]
    return platform.processor() or platform.machine()


def selected_nvidia_wheels():
    """Report the resolved distribution, including when sites shadow versions."""
    names = {dist.metadata['Name'] for dist in importlib.metadata.distributions()
             if dist.metadata.get('Name', '').lower().startswith('nvidia-')}
    return {name: importlib.metadata.version(name) for name in sorted(names)}


def source_descriptor(checkout):
    """Bind all tracked package sources, with no invented registry denominator."""
    checkout = Path(checkout).resolve()
    if command('git', '-C', str(checkout), 'status', '--porcelain'):
        raise ValueError('Accelerate upstream checkout must be clean')
    remote = command('git', '-C', str(checkout), 'remote', 'get-url', 'origin')
    if remote not in ('https://github.com/huggingface/accelerate',
                       'https://github.com/huggingface/accelerate.git',
                       'git@github.com:huggingface/accelerate.git'):
        raise ValueError('Expected official Accelerate upstream')
    paths = command('git', '-C', str(checkout), 'ls-files', '--', 'src/accelerate').splitlines()
    files = [{'path': path, 'sha256': digest(checkout / path)} for path in paths]
    if not files:
        raise ValueError('Empty upstream Accelerate package')
    import ast
    tree = ast.parse((checkout / 'src/accelerate/__init__.py').read_text())
    versions = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == '__version__' for target in node.targets)]
    if len(versions) != 1:
        raise ValueError('Unresolved upstream Accelerate version')
    return {'repo': 'accelerate', 'upstream': {'remote': remote,
            'commit': command('git', '-C', str(checkout), 'rev-parse', 'HEAD'),
            'version': versions[0]}, 'axes': [], 'package_sources': files}


def hyperparameters(optimizer):
    """Observe declared AdamW settings without optimizer implementation flags."""
    return {str(index): {name: group[name] for name in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize')}
            for index, group in enumerate(optimizer.param_groups)}


class Capture:
    def __init__(self, case, torch, device, world=1, rank=0, precision='no'):
        self.enabled = bool(os.environ.get('JITTOR_ACCELERATE_PROTOCOL_OUTPUT'))
        self.torch, self.device = torch, device
        self.case, self.world, self.rank, self.precision = case, world, rank, precision
        self.arrays, self.manifest = {}, {}
        if self.enabled:
            if command('git', '-C', str(ROOT), 'status', '--porcelain'):
                raise ValueError('Protocol measurements require clean Jittor sources')
            self.source = source_descriptor(os.environ['JITTOR_ACCELERATE_CHECKOUT'])

    def add(self, key, value):
        if not self.enabled:
            return
        torch = self.torch
        if isinstance(value, torch.Tensor):
            if callable(getattr(value, 'full_tensor', None)):
                value = value.full_tensor()
            original = str(value.dtype)
            observed_device = value.device.type
            pointer_device = None
            if observed_device == 'cuda':
                ordinal = ctypes.c_int(-1)
                driver = ctypes.CDLL('libcuda.so.1')
                rc = driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ctypes.c_void_p(value.data_ptr()))
                if rc or ordinal.value != value.device.index:
                    raise ValueError('Observed tensor is not an allocation on its declared CUDA ordinal')
                pointer_device = ordinal.value
            if value.dtype == torch.bfloat16:
                value = value.float()
            array = value.detach().clone().cpu().numpy().copy()
            if key in self.arrays:
                raise ValueError('Duplicate protocol observation: ' + key)
            self.arrays[key] = array
            observed = {'shape': list(array.shape), 'dtype': str(array.dtype),
                        'torch_dtype': original, 'device': observed_device}
            if pointer_device is not None:
                observed['pointer_device'] = pointer_device
            self.manifest[key] = observed
        elif isinstance(value, dict):
            for name, item in value.items():
                self.add(key + '/' + str(name), item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self.add(key + '/' + str(index), item)
        elif isinstance(value, (bool, int, float)):
            import numpy as np
            array = np.asarray(value)
            if key in self.arrays:
                raise ValueError('Duplicate protocol observation: ' + key)
            self.arrays[key] = array
            self.manifest[key] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                                  'torch_dtype': 'metadata.' + str(array.dtype), 'device': 'host-metadata'}

    def finish(self, result, backend='none'):
        if not self.enabled:
            return
        import numpy as np
        from _ecosystem_runner import _dependency_report, _runtime_conditions
        from _helpers.child_process import PYTHON

        torch = self.torch
        shim = hasattr(torch, '_torch_compat_install_context')
        binary = getattr(getattr(torch, '_C', None), '__file__', None) if not shim else None
        if not shim and (not binary or not Path(binary).is_file()):
            raise ValueError('Oracle must load a real binary PyTorch extension')
        dependencies = _dependency_report(('accelerate',))
        installed = dependencies['accelerate']
        package = Path(installed['origin']).parent
        expected = {row['path']: row['sha256'] for row in self.source['package_sources']}
        actual = {path: digest(package.joinpath(*Path(path).parts[2:])) for path in expected}
        if actual != expected or installed['version'] != self.source['upstream']['version']:
            raise ValueError('Actually imported Accelerate differs from clean pinned upstream')
        jt = sys.modules.get('jittor') if shim else None
        if jt is not None and Path(jt.__file__).resolve() != ROOT / 'python/jittor/__init__.py':
            raise ValueError('Measured Jittor is from a different worktree')
        precision = {'matmul': bool(torch.backends.cuda.matmul.allow_tf32) if self.device == 'cuda' else False,
                     'cudnn': bool(torch.backends.cudnn.allow_tf32) if self.device == 'cuda' else False,
                     'cudnn_benchmark': bool(torch.backends.cudnn.benchmark) if self.device == 'cuda' else False,
                     'matmul_precision': torch.get_float32_matmul_precision()}
        gpu_rows = []
        if self.device == 'cuda':
            gpu_rows = [line.split(', ') for line in command('nvidia-smi',
                '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader').splitlines()]
        allocation = os.environ.get('JITTOR_ACCELERATE_DEVICES', '').split(',') if self.device == 'cuda' else []
        selected = [row for row in gpu_rows if row[0] in allocation or row[1] in allocation]
        if len(selected) != len(allocation):
            raise ValueError('Cannot resolve physical GPU allocation')
        rank_uuid = None
        logical_index = None
        if self.device == 'cuda':
            logical_index = torch.cuda.current_device()
            driver = ctypes.CDLL('libcuda.so.1')
            cu_device, identifier = ctypes.c_int(), ctypes.create_string_buffer(16)
            if driver.cuDeviceGet(ctypes.byref(cu_device), logical_index) or driver.cuDeviceGetUuid(identifier, cu_device):
                raise ValueError('Cannot bind actual CUDA driver device UUID')
            rank_uuid = 'GPU-' + str(uuid.UUID(bytes=identifier.raw))
        native_flags = {name: int(getattr(jt.flags, name)) for name in
                        ('use_cuda', 'auto_graph_replay', 'auto_graph_replay_retain_bytes')} if jt else {}
        output = Path(os.environ['JITTOR_ACCELERATE_PROTOCOL_OUTPUT'])
        output.mkdir(parents=True, exist_ok=True)
        stem = self.case + '_rank%d' % self.rank
        npz = output / (stem + '.npz')
        if npz.exists() or (output / (stem + '.json')).exists():
            raise ValueError('Refusing to overwrite historical protocol measurements')
        np.savez(npz, **self.arrays)
        source_files = ('compat/tests/torch/_accelerate_protocol.py',
                        'compat/tests/torch/test_accelerate_distributed_integration.py',
                        'compat/tests/torch/test_optimizer_checkpoint_state.py',
                        'tools/run_accelerate_gate.py')
        if os.environ.get('JITTOR_ACCELERATE_CHECKPOINT_PHASE'):
            source_files += ('compat/tests/torch/_accelerate_fresh_process.py',)
        metadata = {'schema_version': 1, 'case': self.case, 'runtime': 'jittor' if shim else 'torch',
                    'jittor_commit': command('git', '-C', str(ROOT), 'rev-parse', 'HEAD'),
                    'upstream': self.source['upstream'], 'source_descriptor': self.source,
                    'measurement_source_sha256': {path: digest(ROOT / path) for path in source_files},
                    'downstream_distribution': {**installed, 'source_sha256': actual},
                    'interpreter': {'executable': PYTHON, 'pid': os.getpid(), 'is_shim': shim,
                                    'torch_version': str(torch.__version__), 'binary_extension': binary,
                                    'binary_sha256': digest(binary) if binary else None},
                    'device': self.device, 'rank': self.rank, 'world_size': self.world,
                    'mixed_precision': self.precision, 'backend': backend,
                    'seed': int(os.environ.get('JITTOR_ACCELERATE_INPUT_SEED', '0')),
                    'hf_hub_offline': os.environ.get('HF_HUB_OFFLINE') == '1',
                    'tf32': precision, 'runtime_conditions': _runtime_conditions(torch, precision),
                    'native_runtime_flags': native_flags,
                    'fallback_policy': 'error' if shim else None,
                    'fallback_count': jt.core.backend_fallback_count() if jt else None,
                    'array_manifest': self.manifest, 'arrays': str(npz), 'arrays_sha256': digest(npz),
                    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'rank_device_uuid': rank_uuid, 'logical_device_index': logical_index,
                    'platform': {'os': platform.platform(), 'cpu': cpu_description(),
                        'accelerator_model': [row[2] for row in selected] or 'not-applicable',
                        'card_count': len(selected), 'card_indices': allocation,
                        'physical_devices': [{'index': row[0], 'uuid': row[1]} for row in selected],
                        'driver': selected[0][3] if selected else 'not-applicable',
                        'cuda': str(torch.version.cuda) if self.device == 'cuda' else 'not-applicable',
                        'cann': 'not-applicable',
                        'nvcc': command(os.environ['JT_BUILD_NVCC_PATH'], '--version') if self.device == 'cuda' else 'not-applicable',
                        'gcc': command(shutil.which('g++') or 'g++', '--version'),
                        'nvidia_wheels': selected_nvidia_wheels(),
                        'jittor_flags': {'use_cuda': native_flags.get('use_cuda', int(self.device == 'cuda')),
                            'use_acl': 0, 'graph_replay': 'default' if native_flags.get('auto_graph_replay') else 'eager',
                            'tf32': precision['matmul'], 'hf32': False},
                        'precision': precision}, 'result': result}
        (output / (stem + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
