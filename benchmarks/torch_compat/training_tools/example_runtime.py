"""Pinned official-example execution: identity, environment and evidence capture."""
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import signal
import subprocess
import time

from runtime import ROOT, lock

REPORT_SOURCE = '''\
import atexit
import json
import os
import sys


def _report():
    directory = os.environ.get("JT_EXAMPLE_REPORT_DIR")
    if not directory:
        return
    payload = {"pid": os.getpid(), "fallback": None, "jittor_version": None,
               "jittor_file": None, "torch_version": None, "torch_file": None,
               "cuda_available": None, "cuda_memory": None, "accelerator_device": None,
               "jittor_has_acl": None, "probe_device": None}
    jittor = sys.modules.get("jittor")
    if jittor is not None:
        try:
            payload["fallback"] = int(jittor.core.backend_fallback_count())
            payload["jittor_version"] = jittor.__version__
            payload["jittor_file"] = jittor.__file__
            payload["jittor_has_acl"] = bool(getattr(getattr(jittor, "compiler", None), "has_acl", 0))
        except Exception:
            pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            payload["torch_version"] = torch.__version__
            payload["torch_file"] = torch.__file__
            payload["cuda_available"] = bool(torch.cuda.is_available())
            if payload["cuda_available"]:
                payload["cuda_memory"] = int(torch.cuda.max_memory_allocated())
            probe = "cuda" if payload.get("jittor_has_acl") else None
            payload["probe_device"] = str(torch.zeros(1, device=probe).device if probe else torch.zeros(1).device)
        except Exception:
            pass
    jittor_state = sys.modules.get("jittor")
    if jittor_state is not None:
        try:
            payload["jittor_use_cuda"] = int(jittor_state.flags.use_cuda)
        except Exception:
            pass
    accelerate = sys.modules.get("accelerate")
    if accelerate is not None:
        try:
            state_module = getattr(accelerate, "state", None) or sys.modules.get("accelerate.state")
            state_class = getattr(state_module, "AcceleratorState", None)
            shared = getattr(state_class, "_shared_state", None)
            storage = getattr(shared, "_storage", shared)
            device = storage.get("device") if isinstance(storage, dict) else None
            payload["accelerator_device"] = str(device) if device is not None else None
            payload["accelerator_state_keys"] = sorted(storage) if isinstance(storage, dict) else None
        except Exception as error:
            payload["accelerator_error"] = type(error).__name__ + ": " + str(error)
    with open(os.path.join(directory, "%d.json" % os.getpid()), "w") as handle:
        json.dump(payload, handle)


atexit.register(_report)
'''

DEVICES = ('cpu', 'cuda', 'rocm', 'ascend')
DEVICE_LINE = re.compile(r'^Device: (\w+)$', re.M)
# Fix PRs live either in the upstream repository or in the campaign fork while
# the integration branch is being assembled.
FIX_SOURCE = re.compile(r'^https://github\.com/(?:Jittor|uyzhang)/jittor/pull/\d+$')


def checkout_identity(checkout):
    """Require a clean checkout of the pinned upstream commit."""
    checkout = pathlib.Path(checkout).resolve()
    config = lock()
    if not (checkout / 'examples').is_dir():
        raise ValueError('Checkout does not contain official examples: ' + str(checkout))
    head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if head != config['commit']:
        raise ValueError('Checkout commit differs from the upstream pin')
    dirty = subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if dirty:
        raise ValueError('Official checkout must be clean')
    return checkout, config


def script_digest(checkout, script):
    path = (checkout / script).resolve()
    if path != checkout and checkout not in path.parents:
        raise ValueError('Example escapes the pinned checkout')
    if not path.is_file():
        raise ValueError('Missing official example: ' + script)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def core_commit():
    source = pathlib.Path(os.environ.get('JITTOR_SOURCE_ROOT', pathlib.Path.home() / '.local/state/jittor-accelerate/jittor-source')).resolve()
    if not (source / '.git').is_dir():
        raise ValueError('JITTOR_SOURCE_ROOT must name the pinned Jittor source')
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if dirty:
        raise ValueError('Candidate requires the clean pinned Jittor source')
    return commit


def allowed_core_commits(config):
    """Current framework pin plus the previously verified pin.

    Evidence recorded before a re-pin stays bound to that earlier commit; each
    run names its own framework source so two pins are never silently mixed.
    """
    commits = [config['core_commit'], *config.get('previous_core_commits', [])]
    return tuple(dict.fromkeys(commits))


def require_allowed_core(config, commit):
    if commit not in allowed_core_commits(config):
        raise ValueError('Jittor source is not an approved framework pin: ' + str(commit))
    return commit


def source_root():
    source = pathlib.Path(os.environ.get('JITTOR_SOURCE_ROOT', pathlib.Path.home() / '.local/state/jittor-accelerate/jittor-source')).resolve()
    if not (source / '.git').is_dir():
        raise ValueError('JITTOR_SOURCE_ROOT must name the pinned Jittor source')
    return source


def patch_manifest(path, config):
    """Validate a declared compat patch against the base source and the runtime.

    A patch manifest states the upstream fix PR and the exact base/patched
    digests of every framework file. The base source must still be the pinned
    commit and its file bytes must match ``base_sha256``; the installed
    framework must match the patched digest. The returned record makes the
    deviation machine-checkable instead of claiming the pinned build.
    """
    manifest_path = pathlib.Path(path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    data = json.loads(raw)
    for key in ('base_commit', 'files'):
        if key not in data:
            raise ValueError('Patch manifest is missing ' + key)
    if data['base_commit'] != config['core_commit']:
        raise ValueError('Patch base commit differs from the pinned core commit')
    sources = data.get('sources')
    if sources is None:
        sources = [data['source']] if data.get('source') else []
    if not isinstance(sources, list) or not sources or not all(
            isinstance(item, str) and FIX_SOURCE.match(item)
            for item in sources):
        raise ValueError('Patch manifest must name the upstream fix pull requests')
    patched_from = data.get('patched_from')
    if patched_from is not None and not re.match(r'^[0-9a-f]{40}$', str(patched_from)):
        raise ValueError('Patch manifest records an invalid source commit')
    if not isinstance(data['files'], list) or not data['files']:
        raise ValueError('Patch manifest must list the patched files')
    source = source_root()
    distribution = importlib.metadata.distribution('jittor')
    seen = set()
    files = []
    for item in data['files']:
        path_value = item.get('path')
        if not path_value or path_value in seen:
            raise ValueError('Patch manifest paths must be unique and non-empty')
        seen.add(path_value)
        installed = pathlib.Path(distribution.locate_file('jittor/' + path_value)).resolve()
        original = (source / path_value).resolve()
        if not installed.is_file():
            raise ValueError('Patch manifest names a missing framework file: ' + path_value)
        base_sha = item.get('base_sha256')
        if base_sha is None:
            # A file added by the patch has no bytes at the base commit. The
            # absence is checked so an omission cannot masquerade as an add.
            if original.exists():
                raise ValueError('Patch manifest declares an added file that exists at the base commit: ' + path_value)
        else:
            if not original.is_file():
                raise ValueError('Patch manifest names a missing framework file: ' + path_value)
            if sha256(original) != base_sha:
                raise ValueError('Pinned framework file differs from the manifest base digest: ' + path_value)
        if sha256(installed) != item.get('sha256'):
            raise ValueError('Installed framework file differs from the declared patch: ' + path_value)
        entry = dict(path=path_value, sha256=item['sha256'], base_sha256=base_sha)
        if 'from' in item:
            if not re.match(r'^[0-9a-f]{40}$', str(item['from'])):
                raise ValueError('Patched file records an invalid source commit: ' + path_value)
            entry['from'] = item['from']
        files.append(entry)
    result = dict(base_commit=data['base_commit'], sources=list(sources), files=files,
                  manifest_sha256=hashlib.sha256(raw).hexdigest())
    if patched_from is not None:
        result['patched_from'] = patched_from
    return result


def installed_version(python, library):
    code = 'import importlib.metadata as m; print(m.version(%r))' % library
    return subprocess.check_output([python, '-c', code], text=True).strip()


def environment(backend, device, cache_root, report_dir, extra_paths=()):
    """Build the child environment for one official example execution."""
    if device not in DEVICES:
        raise ValueError('Unknown device: ' + device)
    report_dir = pathlib.Path(report_dir).resolve()
    reports = report_dir / 'reports'
    site = report_dir / 'site'
    reports.mkdir(parents=True, exist_ok=True)
    site.mkdir(parents=True, exist_ok=True)
    (site / 'sitecustomize.py').write_text(REPORT_SOURCE)
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    extra = os.environ.get('JT_EXAMPLE_EXTRA_PATH', '')
    extra_paths_from_env = [item for item in extra.split(os.pathsep) if item] if extra else []
    paths = [str(site), *[str(path) for path in extra_paths], *extra_paths_from_env,
             os.environ.get('PYTHONPATH', '')]
    env['PYTHONPATH'] = os.pathsep.join(path for path in paths if path)
    env['JT_EXAMPLE_REPORT_DIR'] = str(reports)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               HF_EVALUATE_OFFLINE='1',
               HF_HOME=str(pathlib.Path.home() / '.cache/huggingface'),
               OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', PYTHONUNBUFFERED='1')
    if backend == 'jittor':
        env.update(JITTOR_TORCH_SHIM='1', JITTOR_TORCH_KEEP_HOME='1',
                   JITTOR_TORCH_SKIP_EXT_BUILD='1', JITTOR_HOME=str(cache_root))
        if device == 'cpu':
            env.update(JT_BACKEND='cpu', JT_USE_CUDA='0', use_cuda='0', JT_BUILD_USE_MPI='0',
                       JT_BUILD_USE_NCCL='0', JT_BUILD_USE_MKL='0', JT_BUILD_USE_CUTT='0',
                       JT_BUILD_NVCC_PATH='', nvcc_path='', JT_USE_PARALLEL_OP_COMPILER='0')
        elif device == 'ascend':
            # Jittor discovers the ACL backend from the CANN environment; naming
            # another backend would select a build that this host cannot use.
            env.pop('JT_BACKEND', None)
            env.update(JT_USE_CUDA='1', use_cuda='1', JT_BUILD_USE_MPI='0',
                       JT_BUILD_USE_NCCL='0', JT_BUILD_USE_CUTT='0', JT_USE_PARALLEL_OP_COMPILER='0')
        else:
            env.update(JT_BACKEND='cuda', JT_USE_CUDA='1', use_cuda='1', JT_BUILD_USE_MPI='0',
                       JT_BUILD_USE_NCCL='0', JT_BUILD_USE_CUTT='0', JT_USE_PARALLEL_OP_COMPILER='0')
    else:
        for name in ('JITTOR_TORCH_SHIM', 'JITTOR_HOME', 'JITTOR_TORCH_KEEP_HOME',
                     'JITTOR_TORCH_SKIP_EXT_BUILD'):
            env.pop(name, None)
    return env


def reports(directory):
    return [json.loads(path.read_text()) for path in sorted(pathlib.Path(directory).glob('*.json'))]


def placement(device, log_text, payloads):
    """Device types actually observed, from Accelerator state or GPU allocations."""
    reported = {match.lower() for match in DEVICE_LINE.findall(log_text)}
    if device == 'cuda' and re.search(r'^\[i [^\]]+device\.cc:\d+\] CUDA enabled\.$', log_text, re.M):
        reported.add('cuda')
    chosen = set()
    for payload in payloads:
        value = (payload.get('accelerator_device') or '').lower()
        probe = (payload.get('probe_device') or '').lower()
        if payload.get('jittor_has_acl') and probe.startswith('npu'):
            # The Torch API names an Ascend device ``cuda`` when the Jittor ACL
            # backend backs it; the real placement is the NPU.
            chosen.add('ascend')
        elif value.startswith('cuda'):
            chosen.add('cuda')
        elif value.startswith(('npu', 'ascend')):
            chosen.add('ascend')
        elif value.startswith('rocm'):
            chosen.add('rocm')
        elif value.startswith('cpu'):
            chosen.add('cpu')
    if reported:
        return sorted(reported)
    if chosen:
        return sorted(chosen)
    if device == 'cpu':
        return ['cpu']
    if device == 'cuda' and any(payload.get('cuda_memory') for payload in payloads):
        return ['cuda']
    return []


def classify(exit_code, backend, fallback_delta, device, placed):
    if exit_code != 0:
        return 'failed', 'process exited with %d' % exit_code
    if backend == 'jittor' and fallback_delta != 0:
        return 'failed', 'backend fallback delta %s' % fallback_delta
    if not placed:
        return 'failed', 'no accelerator placement was observed'
    if device != 'cpu' and device not in placed:
        return 'failed', 'observed placement %s does not include %s' % (','.join(placed), device)
    return 'passed', ''


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def execute(command, cwd, env, log_path, timeout):
    log_path = pathlib.Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(log_path, 'w') as handle:
        # The launcher spawns the example as a child; a timeout must terminate
        # the whole group, otherwise the example keeps training and appends to
        # the log after the digest has been recorded.
        process = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    return exit_code, time.time() - start, log_path


def record(backend, device, task, script, checkout, config, command, exit_code, duration,
           log_path, payloads, fallback_delta, core, world_size, precision, extra):
    text = pathlib.Path(log_path).read_text(errors='replace')
    placed = placement(device, text, payloads)
    status, error = classify(exit_code, backend, fallback_delta, device, placed)
    data = dict(task=task, script=script, backend=backend, device=device,
                world_size=int(world_size), precision=precision, command=list(command),
                exit_code=int(exit_code), outcome=status,
                upstream_commit=config['commit'], library_version=config['version'],
                script_sha256=script_digest(checkout, script), log_sha256=sha256(log_path),
                log_path=str(log_path), duration_seconds=round(float(duration), 3),
                hostname=os.environ.get('HOSTNAME') or os.uname().nodename,
                placement=placed, reports=payloads)
    if backend == 'jittor':
        data['core_commit'] = core
        data['framework_source_commit'] = core
        data['fallback_delta'] = fallback_delta
    if error:
        data['error'] = error
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        detail = ''
        if exit_code == -9:
            data['detail'] = 'execution exceeded the run timeout'
            lines = []
        for line in reversed(lines):
            match = re.match(r'^[A-Za-z_][\w.]*(Error|Exception):\s', line)
            if match and not line.startswith('subprocess.'):
                detail = line
                break
        if not detail and lines:
            detail = lines[-1]
        if detail:
            data['detail'] = detail[:400]
    data.update({key: value for key, value in extra.items() if value is not None})
    return data
