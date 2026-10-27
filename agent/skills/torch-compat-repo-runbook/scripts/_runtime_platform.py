"""Actual-process platform fields shared by maintained downstream captures."""

import importlib.metadata
from pathlib import Path
import platform
import subprocess


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
    """Use resolved distributions, retaining actual sys.path precedence."""
    names = {dist.metadata['Name'] for dist in importlib.metadata.distributions()
             if dist.metadata.get('Name', '').lower().startswith('nvidia-')}
    return {name: importlib.metadata.version(name) for name in sorted(names)}


def capture_platform(torch, device, *, physical_devices, card_indices, precision,
                     jittor_flags, nvcc, cxx):
    """Assemble the existing Accelerate header from explicit runtime inputs.

    physical_devices are actual queried [index, UUID, model, driver] rows. The
    caller separately observes each tensor's device/pointer; inventory is never
    a substitute for that evidence. Compiler failures remain failures.
    """
    if device not in ('cpu', 'cuda'):
        raise ValueError('Platform capture supports only actual CPU/CUDA execution')
    if (not isinstance(physical_devices, list) or not isinstance(card_indices, list)
            or len(physical_devices) != len(card_indices)):
        raise ValueError('Incomplete actual physical allocation')
    if device == 'cpu':
        if physical_devices or card_indices:
            raise ValueError('CPU platform cannot claim allocated CUDA devices')
    elif (not physical_devices or any(len(row) != 4 or not all(isinstance(v, str) and v for v in row)
                                      for row in physical_devices)
          or len({row[1] for row in physical_devices}) != len(physical_devices)
          or len(set(card_indices)) != len(card_indices)
          or any(not any(token in row[:2] for token in card_indices) for row in physical_devices)
          or any(not any(token in row[:2] for row in physical_devices) for token in card_indices)):
        raise ValueError('Unobserved or mismatched physical CUDA allocation')
    required_precision = {'matmul', 'cudnn', 'cudnn_benchmark', 'matmul_precision'}
    if (not isinstance(precision, dict) or set(precision) != required_precision
            or any(type(precision[key]) is not bool for key in required_precision - {'matmul_precision'})
            or precision['matmul_precision'] not in ('highest', 'high', 'medium')):
        raise ValueError('Missing actual precision controls')
    required_flags = {'use_cuda', 'use_acl', 'graph_replay', 'tf32', 'hf32'}
    if not isinstance(jittor_flags, dict) or set(jittor_flags) != required_flags:
        raise ValueError('Platform capture requires explicit backend flags')
    if (type(jittor_flags['use_cuda']) is not int or jittor_flags['use_cuda'] != int(device == 'cuda')
            or jittor_flags['use_acl'] != 0 or jittor_flags['graph_replay'] not in ('default', 'eager')
            or jittor_flags['tf32'] is not precision['matmul'] or jittor_flags['hf32'] is not False):
        raise ValueError('Backend flags disagree with observed device/precision')
    return {'os': platform.platform(), 'cpu': cpu_description(),
            'accelerator_model': [row[2] for row in physical_devices] or 'not-applicable',
            'card_count': len(physical_devices), 'card_indices': card_indices,
            'physical_devices': [{'index': row[0], 'uuid': row[1]} for row in physical_devices],
            'driver': physical_devices[0][3] if physical_devices else 'not-applicable',
            'cuda': str(torch.version.cuda) if device == 'cuda' else 'not-applicable',
            'cann': 'not-applicable',
            'nvcc': command(nvcc, '--version') if device == 'cuda' else 'not-applicable',
            'gcc': command(cxx, '--version'), 'nvidia_wheels': selected_nvidia_wheels(),
            'jittor_flags': jittor_flags, 'precision': precision}
