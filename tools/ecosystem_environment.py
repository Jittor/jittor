"""Isolated runtime environments shared by source-bound ecosystem gates."""
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def external_path(path):
    resolved = Path(path).resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError('Measurement/cache paths must be outside the source checkout')
    return resolved


def environment(cache, runtime, device):
    env = dict(os.environ)
    folder = external_path(cache) / ('candidate' if runtime == 'jittor' else 'oracle') / device
    for key, name in {'HOME': 'home', 'REAL_HOME': 'home', 'JITTOR_HOME': 'jittor-home',
                      'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'xdg-cache', 'TORCH_HOME': 'torch-home',
                      'JITTOR_TORCH_CACHE_ROOT': 'shim-cache', 'JITTOR_TORCH_RUNTIME_ROOT': 'shim-runtime',
                      'JITTOR_TORCH_HOME': 'home', 'TORCH_EXTENSIONS_DIR': 'torch-extensions'}.items():
        path = folder / name
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONPATH=str(ROOT / 'python') if runtime == 'jittor' else '',
               JITTOR_SOURCE_ROOT=str(ROOT) if runtime == 'jittor' else '',
               JITTOR_TORCH_SHIM='1' if runtime == 'jittor' else '0',
               JITTOR_TORCH_KEEP_HOME='1', JITTOR_TORCH_KEEP_TMPDIR='1',
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               JT_BACKEND=device, JT_USE_CUDA=str(int(device == 'cuda')), use_cuda=str(int(device == 'cuda')),
               JT_BUILD_USE_MKL='0', JT_BUILD_USE_NCCL='0', JT_BUILD_USE_MPI='0', JT_BUILD_USE_CUTT='0',
               JT_USE_PARALLEL_OP_COMPILER='0', DISABLE_MULTIPROCESSING='1', JT_CUDA_KERNEL_MATH='strict',
               JITTOR_ECOSYSTEM_TF32='1', JITTOR_ECOSYSTEM_CUDNN_BENCHMARK='0',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               cache_name='ecosystem-source-bound-' + device)
    if device == 'cpu':
        env.update(CUDA_VISIBLE_DEVICES='', nvcc_path='', JT_BUILD_NVCC_PATH='')
    else:
        if not env.get('CUDA_VISIBLE_DEVICES', '').strip() or ',' in env['CUDA_VISIBLE_DEVICES']:
            raise ValueError('Allocate exactly one explicit CUDA device')
        nvcc = env.get('JT_BUILD_NVCC_PATH') or env.get('nvcc_path') or shutil.which('nvcc')
        if not nvcc:
            raise ValueError('CUDA requires a real NVCC toolchain')
        env.update(nvcc_path=nvcc, JT_BUILD_NVCC_PATH=nvcc)
    return env
