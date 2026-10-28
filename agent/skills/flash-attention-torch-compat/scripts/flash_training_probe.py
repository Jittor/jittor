"""Actual public FlashAttention APIs, complete gradients and three SGD updates."""

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

APIS = (
    "flash_attn_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
)
UPSTREAM_COMMIT = "5231d95fe13733fb534c01895f7ea88c6a6c7793"


def main(argv=None):
    if not __debug__:
        raise RuntimeError("Probe checks require Python without -O")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", choices=("jittor", "torch"), required=True)
    p.add_argument("--api", choices=APIS, required=True)
    p.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    p.add_argument("--head-dim", type=int, choices=(32, 64), default=32)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--perturbation", type=float, default=0.0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jittor-root", type=Path, default=Path(__file__).resolve().parents[4])
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--source-proof", type=Path,
                   help="Actual local-wheel/build/Torch provenance configuration for protocol admission")
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Diagnostic only; the comparator rejects dirty sources",
    )
    args = p.parse_args(argv)
    checkout = args.jittor_root.resolve()
    upstream = args.upstream.resolve()

    def git_value(path, *arguments):
        return subprocess.check_output(["git", "-C", str(path), *arguments], text=True).strip()

    source_commit = git_value(checkout, "rev-parse", "HEAD")
    source_status = git_value(checkout, "status", "--porcelain")
    source_diff = hashlib.sha256(
        subprocess.check_output(["git", "-C", str(checkout), "diff", "--binary", "HEAD"])
    ).hexdigest()
    if args.output.suffix != ".npz":
        p.error("--output must end in .npz")
    if checkout == args.output.resolve() or checkout in args.output.resolve().parents:
        p.error("raw artifacts must be outside the Jittor checkout")
    if not np.isfinite(args.perturbation):
        p.error("--perturbation must be finite")
    if git_value(checkout, "status", "--porcelain") and not args.allow_dirty:
        p.error("commit the source before acceptance; --allow-dirty permits diagnostics only")
    assert git_value(upstream, "rev-parse", "HEAD") == UPSTREAM_COMMIT
    assert not git_value(upstream, "status", "--porcelain"), "upstream checkout must be clean"
    # The runtime/cache namespace is project-specific. Match the selected source
    # root rather than silently rebuilding under the caller's working directory.
    os.chdir(checkout)
    source_proof = None
    if args.source_proof:
        sys.path.insert(0, str(checkout / "agent/skills/torch-compat-repo-runbook/scripts"))
        from _flash_attention_scorecard import capture_implementation, source_descriptor, source_files

        source_proof = source_descriptor(args.source_proof)
        assert upstream == Path(json.loads(args.source_proof.read_text())['upstream']).resolve()
        framework_files = source_files(checkout)

    if args.runtime == "jittor":
        import jittor as jt

        assert Path(jt.__file__).resolve() == checkout / "python/jittor/__init__.py"
        from jittor._runtime.fallback import forbid_backend_fallbacks

        guard = forbid_backend_fallbacks()
    else:
        guard = contextlib.nullcontext()
    import torch
    import flash_attn

    assert hasattr(torch, "_torch_compat_install_context") == (args.runtime == "jittor")
    if args.runtime == "torch":
        assert "jittor" not in sys.modules, "the oracle must run in an independent interpreter"
        assert not getattr(flash_attn, "_jittor_flash_attn_stub", False)
        assert flash_attn.__version__ == "2.7.4.post1"
        if source_proof is None:
            assert Path(flash_attn.__file__).resolve().parent == upstream / "flash_attn"
        import flash_attn_2_cuda as extension

        assert Path(extension.__file__).suffix in (".so", ".pyd")
    else:
        assert os.environ["JITTOR_FLASH_ATTN_JITTOR_REQUIRED"] == "1"
        from jittor.compat.shim.backends import flash_attention

        facade_source = checkout / "compat/shim/resources/stubs/flash_attn"
        facade_package = Path(flash_attn.__file__).resolve().parent
        for source in facade_source.rglob("*.py"):
            assert (facade_package / source.relative_to(facade_source)).read_bytes() == source.read_bytes(), source.name
        backend = flash_attention.load_backend()
        if backend is None:
            raise RuntimeError('Required native FlashAttention backend failed: ' + str(flash_attention.last_error()))
        assert backend._flashattn_jittor_official
        extension = backend._flashattn_jittor_low_level
        assert Path(extension.__file__).suffix in (".so", ".pyd")
    assert torch.cuda.is_available()
    implementation = None
    if source_proof is not None:
        implementation = capture_implementation(checkout, source_proof, args.runtime, torch,
                                               flash_attn, extension, backend if args.runtime == "jittor" else None)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    precision = {
        "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    assert not precision["matmul"] and not precision["cudnn"]
    assert precision["matmul_precision"] == "highest"
    rng = np.random.RandomState(20261015)
    varlen = "varlen" in args.api
    shape = (11, 2, args.head_dim) if varlen else (2, 11, 2, args.head_dim)
    arrays = [(rng.randn(*shape) * 0.2).astype("float32") for _ in range(3)]
    parameters = [torch.nn.Parameter(torch.tensor(a, device="cuda")) for a in arrays]
    noise_array = (rng.randn(*shape) * 0.1).astype("float32")
    noise_array.reshape(-1)[0] += args.perturbation
    noise = torch.tensor(noise_array, device="cuda", requires_grad=True)
    target_array = rng.randn(*shape).astype("float32")
    target = torch.tensor(target_array, device="cuda")
    cu = torch.tensor([0, 4, 11], device="cuda", dtype=torch.int32) if varlen else None
    optimizer = torch.optim.SGD(parameters, lr=0.01)
    result = {f"initial_{i}": a for i, a in enumerate(arrays)}
    observations = []

    def observe(name, value):
        assert value.is_cuda, name
        ordinal = ctypes.c_int(-1)
        rc = ctypes.CDLL('libcuda.so.1').cuPointerGetAttribute(
            ctypes.byref(ordinal), 9, ctypes.c_void_p(value.data_ptr()))
        assert rc == 0 and ordinal.value == 0, (name, rc, ordinal.value)
        observations.append({'label': name, 'device': 'cuda', 'pointer_device': ordinal.value,
                             'dtype': str(value.dtype), 'shape': list(value.shape)})

    def capture(name, value):
        assert value is not None, name
        assert value.is_cuda, (name, str(value.device))
        if source_proof is not None:
            step, field = name.split('_', 1)
            observe(step + '/' + field, value)
        result[name] = value.detach().float().cpu().numpy().copy()
        assert np.isfinite(result[name]).all(), name

    with guard:
        for step in range(3):
            optimizer.zero_grad()
            noise.grad = None
            q = (parameters[0] + noise * 0.1).to(getattr(torch, args.dtype))
            k = parameters[1].to(getattr(torch, args.dtype))
            v = parameters[2].to(getattr(torch, args.dtype))
            for value in (q, k, v):
                assert value.is_cuda and value.dtype == getattr(torch, args.dtype)
            if source_proof is not None:
                for name, value in zip(('q', 'k', 'v'), (q, k, v)):
                    observe(f'step{step}/' + name, value)
            kwargs = dict(dropout_p=0.0, causal=args.causal)
            if args.api.endswith("varlen_qkvpacked_func"):
                inputs = (torch.stack((q, k, v), dim=1), cu, 7)
            elif args.api.endswith("varlen_kvpacked_func"):
                inputs = (q, torch.stack((k, v), dim=1), cu, cu, 7, 7)
            elif args.api.endswith("varlen_func"):
                inputs = (q, k, v, cu, cu, 7, 7)
            elif args.api.endswith("qkvpacked_func"):
                inputs = (torch.stack((q, k, v), dim=2),)
            elif args.api.endswith("kvpacked_func"):
                inputs = (q, torch.stack((k, v), dim=2))
            else:
                assert args.api == "flash_attn_func"
                inputs = (q, k, v)
            output = getattr(flash_attn, args.api)(*inputs, **kwargs)
            assert tuple(output.shape) == shape and output.dtype == getattr(torch, args.dtype)
            loss = (output.float() * target).sum() / (
                float(np.prod(shape)) ** 0.5
            ) + 0.1 * output.float().square().mean()
            loss.backward()
            capture(f"step{step}_output", output)
            capture(f"step{step}_loss", loss)
            for i, value in enumerate(parameters):
                capture(f"step{step}_grad{i}", value.grad)
            capture(f"step{step}_input_grad", noise.grad)
            assert noise.grad.dtype == noise.dtype == torch.float32
            for value in parameters:
                assert value.grad.dtype == value.dtype == torch.float32
            optimizer.step()
            for i, value in enumerate(parameters):
                capture(f"step{step}_param{i}", value)
        torch.cuda.synchronize()
    for i, a in enumerate(arrays):
        assert np.any(result[f"step2_param{i}"] != a), ("unchanged parameter", i)
    record = dict(
        runtime=args.runtime,
        torch=torch.__version__,
        flash_attn=flash_attn.__version__,
        flash_package=flash_attn.__file__,
        extension=getattr(extension, "__file__", None),
        api=args.api,
        dtype=args.dtype,
        head_dim=args.head_dim,
        causal=args.causal,
        perturbation=args.perturbation,
        device=torch.cuda.get_device_name(0),
        tf32=precision["matmul"],
        precision=precision,
        matmul_precision=torch.get_float32_matmul_precision(),
        arrays=len(result),
        fallback_count=jt.core.backend_fallback_count() if args.runtime == "jittor" else None,
    )
    if args.runtime == "jittor":
        record["backend"] = flash_attention.backend_name()
    import platform

    assert git_value(checkout, "rev-parse", "HEAD") == source_commit, (
        "source commit changed during execution"
    )
    assert git_value(checkout, "status", "--porcelain") == source_status, (
        "source status changed during execution"
    )
    assert (
        hashlib.sha256(
            subprocess.check_output(["git", "-C", str(checkout), "diff", "--binary", "HEAD"])
        ).hexdigest()
        == source_diff
    ), "source changed during execution"
    record["baseline"] = source_commit
    record["extension_sha256"] = hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()
    record["jittor_source_clean"] = not bool(git_value(checkout, "status", "--porcelain"))
    record["jittor_source_diff_sha256"] = hashlib.sha256(
        subprocess.check_output(["git", "-C", str(checkout), "diff", "--binary", "HEAD"])
    ).hexdigest()
    record["upstream"] = {
        "commit": git_value(upstream, "rev-parse", "HEAD"),
        "version": flash_attn.__version__,
        "remote": git_value(upstream, "remote", "get-url", "origin"),
    }
    record["inputs"] = {
        "seed": 20261015,
        "shape_kind": "tiny",
        "shape": list(shape),
        "batch": 2,
        "heads": 2,
        "sequence_lengths": [4, 7] if varlen else [11, 11],
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "steps": 3,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "dropout": 0.0,
        "parameter_dtype": "float32",
        "gradient_dtype": "float32",
    }
    record["input_hashes"] = {
        name: hashlib.sha256(value.tobytes()).hexdigest()
        for name, value in zip(("q", "k", "v", "noise", "target"), [*arrays, noise_array, target_array])
    }
    record["platform"] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "actual_device_count": torch.cuda.device_count(),
        "actual_device": torch.cuda.get_device_name(0),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_file": torch.__file__,
        "nvidia_wheels": {},
    }
    record["runtime_threads"] = torch.get_num_threads()
    sys.path.insert(0, str(checkout / "agent/skills/torch-compat-repo-runbook/scripts"))
    from _flash_attention_scorecard import selected_nvidia_wheels
    record['platform']['nvidia_wheels'] = selected_nvidia_wheels()
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        for line in cpu_info.read_text().splitlines():
            if line.startswith("model name"):
                record["platform"]["cpu"] = line.split(":", 1)[1].strip()
                break
    record["platform"]["cpu"] = record["platform"]["cpu"] or platform.machine()
    record["platform"]["gpu_driver_inventory"] = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"],
            text=True,
        )
        .strip()
        .splitlines()
    )
    if args.runtime == "jittor":
        nvcc = jt.compiler.nvcc_path
        cxx = jt.compiler.cc_path
    else:
        nvcc = str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc")
        cxx = os.environ["CXX"]
    record["platform"]["compiler_environment"] = {
        "nvcc": subprocess.check_output(
            [nvcc, "--version"], text=True, stderr=subprocess.STDOUT
        ).strip(),
        "cxx": subprocess.check_output(
            [cxx, "--version"], text=True, stderr=subprocess.STDOUT
        ).strip(),
    }
    from _flash_attention_scorecard import bound
    record['platform']['compiler_executables'] = {'nvcc': bound(nvcc), 'cxx': bound(cxx)}

    record["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sources = subprocess.check_output(
        ["git", "-C", str(upstream), "ls-files", "--", "flash_attn/"], text=True
    ).splitlines()
    record["source_hashes"] = {name: hashlib.sha256((upstream / name).read_bytes()).hexdigest()
                               for name in sources if name.endswith(".py")}
    record["mode"] = "correctness only; concurrent jobs prohibit performance acceptance"
    if args.runtime == "jittor":
        record["native_flags"] = {
            name: getattr(jt.flags, name)
            for name in ("use_cuda", "use_acl", "auto_graph_replay")
            if hasattr(jt.flags, name)
        }
        record["native_flag_aliases"] = {
            "use_acl": "legacy alias of use_cuda; does not identify the active backend"
        }
        record["actual_dispatch"] = list(jt.core.dispatch_context(parameters))
    else:
        record["binary_extension"] = torch._C.__file__
    if args.runtime == "jittor":
        assert tuple(record["actual_dispatch"]) == ("cuda", 0)
    names = ("q", "k", "v")
    record["measurement"] = {
        "case": f"{args.api}-{args.dtype}-d{args.head_dim}-causal{int(args.causal)}",
        "runtime": args.runtime, "device": "cuda", "jittor_commit": source_commit,
        "steps": 3, "tf32": precision,
        "runtime_conditions": {"precision": precision, "num_threads": torch.get_num_threads()},
        "backend": {"use_cuda": True, "use_acl": False},
        "fallback_policy": "error" if args.runtime == "jittor" else "not-applicable",
        "fallback_count": record["fallback_count"],
        "trainable_parameters": [name for name, value in zip(names, parameters) if value.requires_grad],
        "frozen_parameters": [name for name, value in zip(names, parameters) if not value.requires_grad],
        "input_gradients": ["noise"] if noise.requires_grad else [],
    }
    assert record["measurement"]["trainable_parameters"] == list(names)
    assert record["measurement"]["input_gradients"] == ["noise"]
    if source_proof is not None:
        from _flash_attention_scorecard import digest, recheck, selected_nvidia_metadata

        assert source_files(checkout) == framework_files, "framework/tool source changed during execution"
        recheck(implementation['external_files'])
        maps = Path('/proc/self/maps')
        libraries = {line.split()[-1] for line in maps.read_text().splitlines()
                     if '/' in line and any(name in line.split()[-1] for name in
                                             ('libcu', 'libnv', 'libnccl', 'libtorch', 'libc10'))}
        runtimes = {str(Path(p).resolve()) for p in libraries if Path(p).name.startswith('libcudart.so')}
        assert len(runtimes) == 1, runtimes
        runtime_path = runtimes.pop()
        runtime_version = ctypes.c_int(-1)
        assert ctypes.CDLL(runtime_path).cudaRuntimeGetVersion(ctypes.byref(runtime_version)) == 0
        record.update(source_descriptor=source_proof, source_files=framework_files,
                      implementation=implementation, runtime_pid=os.getpid(),
                      runtime_interpreter={'python': sys.executable, 'prefix': sys.prefix, 'argv': sys.argv},
                      device_observations=observations,
                      environment={name: os.environ.get(name) for name in ('HOME', 'JITTOR_HOME', 'TMPDIR', 'XDG_CACHE_HOME')},
                      cuda_runtime={'path': runtime_path, 'version': runtime_version.value},
                      mapped_libraries={str(Path(p).resolve()): digest(p) for p in sorted(libraries)})
        record['platform']['nvidia_metadata'] = selected_nvidia_metadata()
        if args.runtime == 'jittor':
            native_binaries = {line.split()[-1] for line in maps.read_text().splitlines()
                               if '/' in line and line.split()[-1].endswith('.so')
                               and any(str(Path(os.environ[name]).resolve()) in line.split()[-1]
                                       for name in ('JITTOR_HOME', 'JITTOR_TORCH_CACHE_ROOT'))}
            assert any('jittor_core' in p for p in native_binaries), native_binaries
            implementation['external_files'].update({str(Path(p).resolve()): digest(p) for p in native_binaries})
        driver = ctypes.CDLL('libcuda.so.1')
        native_device = ctypes.c_int(-1)
        assert driver.cuDeviceGet(ctypes.byref(native_device), 0) == 0
        uuid = (ctypes.c_ubyte * 16)()
        assert driver.cuDeviceGetUuid_v2(ctypes.byref(uuid), native_device) == 0
        hex_uuid = bytes(uuid).hex()
        uuid_string = 'GPU-' + '-'.join((hex_uuid[:8], hex_uuid[8:12], hex_uuid[12:16],
                                        hex_uuid[16:20], hex_uuid[20:]))
        visible = os.environ['CUDA_VISIBLE_DEVICES']
        rows = [r.split(', ') for r in record['platform']['gpu_driver_inventory']]
        allocated = [r for r in rows if visible in (r[0], r[2])]
        assert len(allocated) == 1 and allocated[0][2] == uuid_string
        record['physical_device_uuid'] = uuid_string
        from _flash_attention_scorecard import validate_implementation
        validate_implementation(record, source_proof, checkout, args.runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **result)
    record["artifact_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
