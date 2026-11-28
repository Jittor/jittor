"""Runtime identity, pinned source checks and explicit CPU execution."""
import hashlib
import importlib
import importlib.metadata
import platform
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def lock():
    return json.loads((ROOT/'upstream/lock.json').read_text())


def verify_sources():
    config = lock()
    for name, digest in config['files'].items():
        path = ROOT/'upstream/snapshot'/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed official source: '+name)
    return config


def payload_digest(config):
    return hashlib.sha256(json.dumps(config['runtime_payload_sha256'],sort_keys=True).encode()).hexdigest()


def measurement_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['tools/runtime.py','tools/run_upstream_tests.py','examples/offline_training.py']}


def initialize(backend, device="cpu"):
    if device not in ("cpu", "cuda"):
        raise ValueError("integration runner supports CPU/CUDA")
    config = verify_sources()
    os.environ.update(USE_TORCH_XLA='0', ACCELERATE_USE_CPU=str(device=='cpu').lower(),
                      DS_ACCELERATOR=device, JT_BACKEND=device, JT_USE_CUDA=str(int(device=='cuda')),
                      use_cuda=str(int(device=='cuda')), JT_BUILD_USE_MPI='0',
                      JT_BUILD_USE_MKL='0', JT_BUILD_USE_CUTT='0', DISABLE_MULTIPROCESSING='1',
                      JT_USE_PARALLEL_OP_COMPILER='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    if device == 'cpu':
        os.environ.update(CUDA_VISIBLE_DEVICES='', JT_BUILD_NVCC_PATH='', nvcc_path='', JT_BUILD_USE_NCCL='0')
    if backend == 'jittor':
        os.environ['JITTOR_TORCH_SHIM'] = '1'
        import jittor
        jittor.flags.use_cuda = int(device == 'cuda')
    import torch
    is_candidate = hasattr(torch, '_torch_compat_install_context')
    if is_candidate != (backend == 'jittor'):
        raise RuntimeError('Wrong Torch implementation for requested backend')
    if backend == 'torch' and not str(torch._C.__file__).endswith(('.so','.pyd','.dylib')):
        raise RuntimeError('Oracle requires native Torch binary')
    library = importlib.import_module(config['library'])
    if library.__version__ != config['version']:
        raise RuntimeError('Installed upstream version differs from pin')
    if config['library']=='ray' and library.__commit__ != config['commit']:
        raise RuntimeError('Ray wheel commit differs from pin')
    package = Path(library.__file__).parent
    for name, digest in config['runtime_payload_sha256'].items():
        if hashlib.sha256((package/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Installed upstream source differs from pin: '+name)
    probe = torch.ones(2, device=device)
    if probe.device.type != device:
        raise RuntimeError('probe did not execute on the requested device')
    cpuinfo = Path('/proc/cpuinfo')
    hardware = next((line.partition(':')[2].strip() for line in cpuinfo.read_text().splitlines() if line.startswith('model name')),platform.machine()) if cpuinfo.exists() else platform.machine()
    if device == 'cuda':
        hardware = torch.cuda.get_device_name(torch.cuda.current_device())
    identity = dict(hardware=hardware, python_version=platform.python_version(), numpy_version=importlib.metadata.version('numpy'), measurement_sha256=measurement_hashes(), runtime_payload_digest=payload_digest(config), placement_scope='Explicit device probe; runtime test assertions check workload placement', backend=backend, device=device, torch_version=torch.__version__, torch_file=torch.__file__, library_version=library.__version__, library_file=library.__file__, upstream_commit=config['commit'])
    if device == 'cuda':
        identity['visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    if backend == 'jittor':
        source_root = Path(os.environ['JITTOR_SOURCE_ROOT']).resolve()
        commit = subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip()
        if commit != config['core_commit'] or subprocess.check_output(['git','-C',str(source_root),'status','--porcelain','--untracked-files=no'],text=True).strip():
            raise RuntimeError('Candidate requires the clean pinned Jittor source')
        distribution = importlib.metadata.distribution('jittor')
        if Path(jittor.__file__).resolve()!=Path(distribution.locate_file('jittor/__init__.py')).resolve():
            raise RuntimeError('Loaded core differs from installed distribution')
        payload = {}
        for package_name in ('jittor','jittor_utils'):
            package_root = Path(distribution.locate_file(package_name)).resolve()
            for path in package_root.rglob('*'):
                if not path.is_file() or path.suffix not in ('.py','.pyi','.h','.cc','.c','.cu','.hpp','.cpp','.cuh','.inl'):
                    continue
                rel = path.relative_to(package_root)
                source = source_root/rel if package_name=='jittor' and rel.parts[0] in ('compat','src','backends') else source_root/'python'/package_name/rel
                if not source.exists():
                    source = source_root/'python'/package_name/rel
                if not source.is_file() or source.read_bytes()!=path.read_bytes():
                    raise RuntimeError('Installed framework payload differs: '+str(rel))
                payload[package_name+'/'+rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        identity.update(core_commit=commit, framework_payload_digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(), framework_payload_files=len(payload), jittor_version=jittor.__version__, jittor_file=jittor.__file__)
    return torch, identity


def external_output(value):
    path = Path(value).expanduser().resolve()
    if path.is_relative_to(ROOT) or path.exists():
        raise ValueError('Choose a new output path outside the repository')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
