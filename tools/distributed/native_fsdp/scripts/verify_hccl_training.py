#!/usr/bin/env python3
"""Required two-rank NPU DDP/FSDP2 trajectory and fresh-process FULL resume.

The model, fixed weights, AdamW groups and value serialization are reused from
the maintained Accelerate integration test. This fixture intentionally has no
stochastic operation after initialization; dropout/RNG recovery is a separate
gate. --help and both --check-* modes do not import Torch or Jittor.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from verify_hccl_stack import prepare_rank_state, prepare_runtime


ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATHS = (
    "compat/collectives.py", "compat/torch/installers/distributed.py",
    "compat/fsdp2/api.py", "compat/fsdp2/dtensor.py", "compat/fsdp2/grad_sync.py",
    "compat/fsdp2/_state_dict.py", "python/jittor/distributed/process_group.py",
    "backends/comm/hccl/src/hccl_wrapper.cc",
    "tools/distributed/native_fsdp/scripts/verify_hccl_stack.py",
    "compat/tests/torch/test_accelerate_distributed_integration.py",
    "tools/distributed/native_fsdp/scripts/verify_hccl_training.py",
)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS}


def compare_tree(actual, expected, *, atol, rtol, path="root"):
    import numpy as np
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), path
        for key in expected:
            compare_tree(actual[key], expected[key], atol=atol, rtol=rtol,
                         path=path + "." + str(key))
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_tree(left, right, atol=atol, rtol=rtol,
                         path=path + "[%d]" % index)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol,
                                   equal_nan=False, err_msg=path)
    else:
        assert actual == expected, (path, actual, expected)


def read_world(path):
    import numpy as np
    reports = [json.loads((path / ("rank%d.json" % rank)).read_text())
               for rank in range(2)]
    for rank, report in enumerate(reports):
        assert report["status"] == "PASS" and report["rank"] == rank
        assert report["world_size"] == 2 and report["backend"] == "hccl"
        assert report["fallback_delta"] in (None, 0)
        assert all(value.split(":")[0] == "npu"
                   for value in report["parameter_devices"].values())
        assert report["records"], "no training records"
        count = report["config"]["updates"]
        cut = report["config"]["split"]
        expected = (list(range(1, cut + 1)) if report["phase"] == "prefix" else
                    list(range(cut + 1, count + 1)) if report["phase"] == "resume" else
                    list(range(1, count + 1)))
        assert [record["update"] for record in report["records"]] == expected
        assert report["completed_updates"] == expected[-1]
        for record in report["records"]:
            assert set(record["gradients"]) == set(record["clipped_gradients"]) == {"weight", "bias"}
            assert len(record["micro_losses"]) == 2
            if report["config"]["wrapping"] == "ddp":
                local = record["accumulation_gradients"]
                assert set(local) == {"weight", "bias"}
                assert np.asarray(local["weight"]).shape == (3, 4)
                assert np.asarray(local["bias"]).shape == (3,)
                assert all(np.isfinite(np.asarray(value)).all()
                           for value in local.values())
                sample = record["accumulation_sample"]
                assert sample["rank"] == rank and sample["micro"] == 0
                update = record["update"] - 1
                expected_x = (np.arange(16, dtype=np.float32).reshape(4, 4) / 100
                              + rank / 10 + update / 1000)
                expected_y = (np.arange(12, dtype=np.float32).reshape(4, 3) / 8
                              - rank / 20 + update / 2000)
                np.testing.assert_array_equal(
                    np.asarray(sample["input"], dtype=np.float32), expected_x)
                np.testing.assert_array_equal(
                    np.asarray(sample["target"], dtype=np.float32), expected_y)
            else:
                # FSDP2 can keep unsynchronized gradients in private full
                # buffers; its first-micro layout is a separate contract.
                assert record["accumulation_gradients"] is None
                assert record["accumulation_sample"] is None
        compare_tree(report["final"], {
            key: report["records"][-1][key] for key in ("model", "optimizer", "scheduler")},
            atol=0, rtol=0)
        assert all(record["gradients"] and record["clipped_gradients"]
                   for record in report["records"])
    assert reports[0]["source_hashes"] == reports[1]["source_hashes"]
    assert reports[0]["config"] == reports[1]["config"]
    # Loss depends on rank-local inputs; global gradients and updates must not.
    for left, right in zip(reports[0]["records"], reports[1]["records"]):
        if reports[0]["config"]["wrapping"] == "ddp":
            assert left["accumulation_sample"]["input"] != right["accumulation_sample"]["input"]
            assert any(not np.array_equal(
                left["accumulation_gradients"][key],
                right["accumulation_gradients"][key])
                for key in ("weight", "bias")), (
                    "DDP no_sync first-micro gradients were already synchronized")
        for key in ("gradients", "clipped_gradients", "model", "optimizer", "scheduler"):
            compare_tree(left[key], right[key], atol=0, rtol=0,
                         path="replica." + key)
    return reports


def check_training(paths):
    full, prefix, resumed = [read_world(path) for path in paths]
    for rank in range(2):
        left, first, second = full[rank], prefix[rank], resumed[rank]
        assert left["phase"] == "full" and first["phase"] == "prefix"
        assert second["phase"] == "resume"
        assert left["config"] == first["config"] == second["config"]
        assert left["source_hashes"] == first["source_hashes"] == second["source_hashes"]
        assert left["runtime"] == first["runtime"] == second["runtime"]
        assert first["process_identity"] != second["process_identity"], (
            "prefix and resume must run in separate processes")
        cut = left["config"]["split"]
        compare_tree(first["records"], left["records"][:cut], atol=0, rtol=0)
        compare_tree(second["records"], left["records"][cut:], atol=0, rtol=0)
        compare_tree(second["restored"], first["final"], atol=0, rtol=0)
        assert second["completed_updates"] == left["config"]["updates"]
    print(json.dumps({"status": "PASS", "gate": "fresh-process-resume",
                      "numeric_tolerance": "exact", "ranks": 2}))


def check_oracle(paths, atol, rtol):
    jittor, oracle = [read_world(path) for path in paths]
    for rank in range(2):
        actual, expected = jittor[rank], oracle[rank]
        assert actual["runtime"] == "jittor" and expected["runtime"] == "torch"
        assert actual["phase"] == expected["phase"] == "full"
        assert actual["config"] == expected["config"]
        assert actual["source_hashes"] == expected["source_hashes"]
        compare_tree(actual["initial"], expected["initial"], atol=0, rtol=0)
        compare_tree(actual["records"], expected["records"], atol=atol, rtol=rtol)
    print(json.dumps({"status": "PASS", "gate": "native-torch-npu-oracle",
                      "atol": atol, "rtol": rtol, "ranks": 2}))


def load_helpers():
    path = ROOT / "compat/tests/torch/test_accelerate_distributed_integration.py"
    spec = importlib.util.spec_from_file_location("_hccl_maintained_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def full_tensor(value):
    getter = getattr(value, "full_tensor", None)
    return getter() if callable(getter) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("jittor", "torch"))
    parser.add_argument("--wrapping", choices=("ddp", "fsdp2"), default="ddp")
    parser.add_argument("--phase", choices=("full", "prefix", "resume"), default="full")
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--split", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--check-training", type=Path, nargs=3,
                        metavar=("FULL", "PREFIX", "RESUMED"))
    parser.add_argument("--check-oracle", type=Path, nargs=2,
                        metavar=("JITTOR", "TORCH"))
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--rtol", type=float, default=2e-5)
    args = parser.parse_args()
    if args.check_training:
        check_training(args.check_training)
        return
    if args.check_oracle:
        check_oracle(args.check_oracle, args.atol, args.rtol)
        return
    if args.runtime is None or args.output is None:
        parser.error("--runtime and --output are required")
    if not 6 <= args.updates <= 10 or not 0 < args.split < args.updates:
        parser.error("require 6 <= updates <= 10 and 0 < split < updates")
    if args.phase != "full" and args.checkpoint is None:
        parser.error("--checkpoint is required for prefix/resume")

    prepare_rank_state(args.state_root)
    import numpy as np
    torch, jt, device, scope, fallback = prepare_runtime(args.runtime)
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_state_dict, set_state_dict)

    assert int(os.environ.get("WORLD_SIZE", "0")) == 2
    dist.init_process_group(backend="hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2 and dist.get_backend() == "hccl"
    helpers = load_helpers()
    options = StateDictOptions(full_state_dict=True, cpu_offload=False)
    config = {"wrapping": args.wrapping, "updates": args.updates, "split": args.split,
              "dtype": "float32", "accumulation": 2, "clip_max_norm": .25,
              "scheduler": "StepLR(step_size=1,gamma=0.8)"}
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_file = (None if args.checkpoint is None else
                       args.checkpoint / ("rank%d.pt" % rank))

    def cpu_tree(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone().cpu()
        if isinstance(value, dict):
            return {key: cpu_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu_tree(item) for item in value)
        return value

    try:
        with scope:
            raw, optimizer = helpers._model_optimizer(device)
            initial = helpers._values(dict(raw.named_parameters()))
            if args.wrapping == "fsdp2":
                from torch.distributed.device_mesh import init_device_mesh
                from torch.distributed.fsdp import fully_shard
                # Keep the maintained optimizer options but bind them to the
                # public parameters after fully_shard replaces local storage.
                names = {id(param): name for name, param in raw.named_parameters()}
                specs = [(tuple(names[id(param)] for param in group["params"]),
                          {key: value for key, value in group.items() if key != "params"})
                         for group in optimizer.param_groups]
                mesh = init_device_mesh("npu", (2,))
                fully_shard(raw, mesh=mesh)
                parameters = dict(raw.named_parameters())
                optimizer = torch.optim.AdamW([
                    dict(settings, params=[parameters[name] for name in group_names])
                    for group_names, settings in specs])
                model = raw
            else:
                model = torch.nn.parallel.DistributedDataParallel(
                    raw, device_ids=[device.index], output_device=device.index,
                    broadcast_buffers=False)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.8)

            def snapshot():
                model_state, optimizer_state = get_state_dict(raw, optimizer, options=options)
                assert set(model_state) == {"weight", "bias"}, model_state.keys()
                assert set(optimizer_state["state"]) == {"weight", "bias"}, optimizer_state
                for fields in optimizer_state["state"].values():
                    assert {"step", "exp_avg", "exp_avg_sq"} <= set(fields), fields.keys()
                return {"model": helpers._values(model_state),
                        "optimizer": helpers._values(optimizer_state),
                        "scheduler": helpers._values(scheduler.state_dict())}

            def gradients():
                result = {}
                for name, parameter in raw.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    assert parameter.dtype == torch.float32, (name, parameter.dtype)
                    assert parameter.grad is not None, "missing gradient: " + name
                    value = full_tensor(parameter.grad)
                    assert value.device.type == "npu", (name, value.device)
                    result[name] = helpers._values(value)
                    assert np.isfinite(np.asarray(result[name])).all(), name
                assert set(result) == {"weight", "bias"}, result
                assert np.asarray(result["weight"]).shape == (3, 4)
                assert np.asarray(result["bias"]).shape == (3,)
                return result

            restored = None
            start = 0
            if args.phase == "resume":
                assert checkpoint_file.is_file(), checkpoint_file
                saved = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
                assert saved["config"] == config and saved["completed_updates"] == args.split
                assert saved["source_hashes"] == source_hashes()
                set_state_dict(raw, optimizer, model_state_dict=saved["model"],
                               optim_state_dict=saved["optimizer"], options=options)
                scheduler.load_state_dict(saved["scheduler"])
                restored = snapshot()
                compare_tree(restored, saved["values"], atol=0, rtol=0)
                start = args.split
            stop = args.split if args.phase == "prefix" else args.updates
            records = []
            for update in range(start, stop):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                accumulation_gradients = accumulation_sample = None
                for micro in range(2):
                    x_array = (np.arange(16, dtype=np.float32).reshape(4, 4) / 100
                               + rank / 10 + update / 1000 + micro / 100)
                    y_array = (np.arange(12, dtype=np.float32).reshape(4, 3) / 8
                               - rank / 20 + update / 2000)
                    x = torch.tensor(x_array, device=device)
                    y = torch.tensor(y_array, device=device)
                    if args.wrapping == "fsdp2":
                        model.set_requires_gradient_sync(micro == 1)
                        accumulation_scope = contextlib.nullcontext()
                    else:
                        accumulation_scope = model.no_sync() if micro == 0 else contextlib.nullcontext()
                    # no_sync must include forward as well as backward.
                    with accumulation_scope:
                        loss = ((model(x) - y) ** 2).mean()
                        (loss / 2).backward()
                    losses.append(helpers._values(loss))
                    if args.wrapping == "ddp" and micro == 0:
                        # Observe complete local gradients before the second
                        # microbatch enables DDP synchronization.
                        accumulation_gradients = gradients()
                        accumulation_sample = {
                            "rank": rank, "micro": micro,
                            "input": helpers._values(x),
                            "target": helpers._values(y)}
                before_clip = gradients()
                norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), max_norm=.25)
                after_clip = gradients()
                assert float(helpers._values(full_tensor(norm))) > .25, "fixture did not exercise clipping"
                optimizer.step()
                scheduler.step()
                record = {"update": update + 1, "micro_losses": losses,
                          "accumulation_gradients": accumulation_gradients,
                          "accumulation_sample": accumulation_sample,
                          "gradients": before_clip, "clipped_gradients": after_clip,
                          "clip_norm": helpers._values(full_tensor(norm))}
                record.update(snapshot())
                records.append(record)
                print("HCCL_TRAIN rank=%d wrapping=%s update=%d phase=%s" %
                      (rank, args.wrapping, update + 1, args.phase), flush=True)
            final = snapshot()
            if args.phase == "prefix":
                args.checkpoint.mkdir(parents=True, exist_ok=True)
                assert not checkpoint_file.exists(), "refusing checkpoint overwrite"
                model_state, optimizer_state = get_state_dict(raw, optimizer, options=options)
                payload = {"config": config, "completed_updates": stop,
                           "source_hashes": source_hashes(), "values": final,
                           "model": cpu_tree(model_state), "optimizer": cpu_tree(optimizer_state),
                           "scheduler": scheduler.state_dict()}
                torch.save(payload, checkpoint_file)
            if jt is not None:
                jt.sync_all(True)
                assert jt.core.backend_fallback_count() == fallback
            else:
                torch.npu.synchronize()
            parameter_devices = {
                name: str((value.to_local() if callable(getattr(value, "to_local", None))
                           else value).device)
                for name, value in raw.named_parameters()}
            assert all(value.split(":")[0] == "npu" for value in parameter_devices.values())
            dist.barrier()
        identity = {"pid": os.getpid(), "process_start_ticks":
                    Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]}
        report = {"status": "PASS", "runtime": args.runtime, "phase": args.phase,
                  "rank": rank, "world_size": world, "backend": dist.get_backend(),
                  "config": config, "source_hashes": source_hashes(),
                  "process_identity": identity, "completed_updates": stop,
                  "torch_file": getattr(torch, "__file__", None),
                  "torch_version": str(torch.__version__), "parameter_devices": parameter_devices,
                  "fallback_delta": None if jt is None else
                                    jt.core.backend_fallback_count() - fallback,
                  "rng_scope": "no stochastic operations after fixed initialization",
                  "initial": initial, "restored": restored, "records": records, "final": final}
        (args.output / ("rank%d.json" % rank)).write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
