#!/usr/bin/env python3
"""Run a required two-rank HCCL collective smoke on Jittor or native torch_npu.

--help and --check-reports are host-only. All runtime imports happen in main
after arguments are parsed. Use the skill instructions to prewarm isolated
caches serially and select assigned devices before executing a card case.
"""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys


def check_reports(path):
    reports = [json.loads((path / ("rank%d.json" % rank)).read_text())
               for rank in range(2)]
    for rank, report in enumerate(reports):
        assert report["status"] == "PASS", report
        assert report["rank"] == rank and report["world_size"] == 2, report
        assert report["backend"] == "hccl", report
        assert report["tensor_device"].split(":")[0] == "npu", report
        assert report["fallback_delta"] in (None, 0), report
        assert report["tests"] == [
            "all_reduce_sum", "broadcast", "all_gather", "all_gather_object"], report
        expected = [float(value) for value in range(1, 33)]
        assert report["tensor_values"] == {
            "source": [value * (rank + 1) for value in expected],
            "all_reduce_sum": [value * 3 for value in expected],
            "broadcast": expected,
            "all_gather": [[value * (peer + 1) for value in expected]
                           for peer in range(2)],
        }, report
        assert all(str(device).split(":")[0] == "npu"
                   for device in report["tensor_devices"].values()), report
        if report.get("schema_version", 1) >= 2:
            assert set(report["tensor_dtypes"]) == set(report["tensor_devices"]), report
            assert all(value.rsplit(".", 1)[-1] == report["dtype"]
                       for value in report["tensor_dtypes"].values()), report
        else:
            assert report["dtype"] == "float32", "legacy reports have no actual dtype evidence"
    assert reports[0]["source_hashes"] == reports[1]["source_hashes"]
    assert reports[0]["runtime"] == reports[1]["runtime"]
    assert reports[0]["dtype"] == reports[1]["dtype"]
    print(json.dumps({"status": "PASS", "ranks": 2, "reports": str(path)}))


def check_oracle(jittor_path, torch_path):
    check_reports(jittor_path)
    check_reports(torch_path)
    for rank in range(2):
        jittor = json.loads((jittor_path / ("rank%d.json" % rank)).read_text())
        oracle = json.loads((torch_path / ("rank%d.json" % rank)).read_text())
        assert jittor["runtime"] == "jittor" and oracle["runtime"] == "torch"
        for name in ("dtype", "source_hashes", "tensor_values", "objects"):
            assert jittor[name] == oracle[name], (rank, name)
    print(json.dumps({"status": "PASS", "comparison": "complete collective tensors",
                      "ranks": 2, "jittor": str(jittor_path), "oracle": str(torch_path)}))


def prepare_rank_state(state_root):
    """Place each rank's mutable state below the explicitly selected lab."""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    lab = os.environ.get("JITTOR_LAB_ROOT")
    if not lab:
        raise RuntimeError("set JITTOR_LAB_ROOT before running a worker")
    lab = Path(lab).resolve()
    rank = int(os.environ.get("RANK", "0"))
    if state_root is not None:
        base = state_root.resolve() / ("rank%d" % rank)
        for name, child in (
                ("HOME", "home"), ("JITTOR_HOME", "jittor-home"),
                ("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg-cache")):
            value = base / child
            if os.path.commonpath((str(lab), str(value))) != str(lab):
                raise ValueError("state root must stay inside JITTOR_LAB_ROOT")
            value.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(value)
        os.environ.setdefault("HF_HOME", str(base / "hf-home"))
    for name in ("HOME", "JITTOR_HOME", "TMPDIR", "XDG_CACHE_HOME", "HF_HOME"):
        value = os.environ.get(name)
        if value is None or os.path.commonpath(
                (str(lab), str(Path(value).resolve()))) != str(lab):
            raise RuntimeError("%s must name a directory below JITTOR_LAB_ROOT" % name)
    source = Path(__file__).resolve().parents[4]
    selected = os.environ.get("JITTOR_SOURCE_ROOT")
    if selected is None or Path(selected).resolve() != source:
        raise RuntimeError("JITTOR_SOURCE_ROOT must name this exact worktree")


def prepare_runtime(runtime):
    import torch
    if runtime == "torch":
        assert not hasattr(torch, "_torch_compat_install_context"), torch
        assert "jittor" not in sys.modules, "oracle imported Jittor"
        import torch_npu  # noqa: F401 - registers native NPU device and FSDP patches.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.npu.set_device(local_rank)
        device = torch.device("npu", local_rank)
        jt = None
        scope = contextlib.nullcontext()
        fallback = None
    else:
        import jittor as jt
        from jittor._runtime.fallback import forbid_backend_fallbacks
        assert hasattr(torch, "_torch_compat_install_context"), torch
        torch_npu = sys.modules.get("torch_npu")
        assert torch_npu is not None, "Jittor shim did not publish its torch_npu facade"
        assert getattr(torch_npu, "__jittor_acl_facade__", False), "Jittor loaded native torch_npu"
        assert getattr(torch_npu, "__jittor_provider__", None) == "jittor.backends.acl"
        facade_path = Path(__file__).resolve().parents[4] / "compat/torch/installers/npu.py"
        assert Path(torch_npu.__file__).resolve() == facade_path
        assert Path(jt.__file__).resolve().parent == Path(__file__).resolve().parents[4] / "python/jittor"
        assert int(jt.world_size) == 2 and jt.compile_extern.hccl_ops is not None
        jt.flags.use_acl = 1
        jt.flags.use_cuda = 1
        device = torch.device("npu", 0)
        fallback = jt.core.backend_fallback_count()
        scope = forbid_backend_fallbacks()

    return torch, jt, device, scope, fallback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("jittor", "torch"))
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"),
                        default="float32")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--check-reports", type=Path)
    parser.add_argument("--check-oracle", type=Path, nargs=2)
    args = parser.parse_args()
    if args.check_oracle:
        check_oracle(*args.check_oracle)
        return
    if args.check_reports:
        check_reports(args.check_reports)
        return
    if args.runtime is None or args.output is None:
        parser.error("--runtime and --output are required for card execution")

    prepare_rank_state(args.state_root)
    import numpy as np
    torch, jt, device, scope, fallback = prepare_runtime(args.runtime)
    import torch.distributed as dist
    assert int(os.environ.get("WORLD_SIZE", "0")) == 2
    dist.init_process_group(backend="hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2 and dist.get_backend() == "hccl"
    dtype = getattr(torch, args.dtype)
    inputs = np.arange(1, 33, dtype=np.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with scope:
            source = torch.tensor(inputs * (rank + 1), device=device, dtype=dtype)
            assert source.device.type == "npu", source.device
            tensor_devices = {"source": str(source.device)}
            assert source.dtype == dtype, (source.dtype, dtype)
            tensor_dtypes = {"source": str(source.dtype)}
            tensor_values = {"source": source.float().cpu().numpy().tolist()}
            reduced = source.clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            assert reduced.device.type == "npu", reduced.device
            assert reduced.dtype == dtype, (reduced.dtype, dtype)
            tensor_dtypes["all_reduce_sum"] = str(reduced.dtype)
            tensor_devices["all_reduce_sum"] = str(reduced.device)
            actual = reduced.float().cpu().numpy()
            np.testing.assert_array_equal(actual, inputs * 3)
            tensor_values["all_reduce_sum"] = actual.tolist()
            broadcast = torch.tensor(inputs + 99 * rank, device=device, dtype=dtype)
            dist.broadcast(broadcast, src=0)
            assert broadcast.device.type == "npu", broadcast.device
            assert broadcast.dtype == dtype, (broadcast.dtype, dtype)
            tensor_dtypes["broadcast"] = str(broadcast.dtype)
            tensor_devices["broadcast"] = str(broadcast.device)
            actual = broadcast.float().cpu().numpy()
            np.testing.assert_array_equal(actual, inputs)
            tensor_values["broadcast"] = actual.tolist()
            gathered = [torch.zeros_like(source) for _ in range(world)]
            dist.all_gather(gathered, source)
            tensor_values["all_gather"] = []
            for peer, value in enumerate(gathered):
                assert value.device.type == "npu", value.device
                assert value.dtype == dtype, (value.dtype, dtype)
                tensor_dtypes["all_gather%d" % peer] = str(value.dtype)
                tensor_devices["all_gather%d" % peer] = str(value.device)
                actual = value.float().cpu().numpy()
                np.testing.assert_array_equal(actual, inputs * (peer + 1))
                tensor_values["all_gather"].append(actual.tolist())
            objects = [None] * world
            dist.all_gather_object(objects, {"rank": rank, "dtype": args.dtype})
            assert objects == [{"rank": peer, "dtype": args.dtype}
                               for peer in range(world)], objects
            if jt is not None:
                jt.sync_all(True)
                assert jt.core.backend_fallback_count() == fallback
            else:
                torch.npu.synchronize()
            dist.barrier()
        repo = Path(__file__).resolve().parents[4]
        paths = (
            "compat/collectives.py", "compat/torch/installers/distributed.py",
            "python/jittor/distributed/process_group.py",
            "backends/comm/hccl/src/hccl_wrapper.cc",
        )
        report = {
            "schema_version": 2,
            "status": "PASS", "runtime": args.runtime, "rank": rank,
            "world_size": world, "backend": dist.get_backend(), "dtype": args.dtype,
            "tensor_device": str(source.device), "pid": os.getpid(),
            "proc_starttime": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
            "tensor_devices": tensor_devices, "tensor_values": tensor_values,
            "tensor_dtypes": tensor_dtypes,
            "objects": objects,
            "torch_file": getattr(torch, "__file__", None),
            "torch_version": str(torch.__version__),
            "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "fallback_delta": None if jt is None else
                              jt.core.backend_fallback_count() - fallback,
            "tests": ["all_reduce_sum", "broadcast", "all_gather", "all_gather_object"],
            "source_hashes": {
                name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                for name in paths},
        }
        (args.output / ("rank%d.json" % rank)).write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"status": "PASS", "rank": rank, "runtime": args.runtime}),
              flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
