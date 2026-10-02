"""Independent-process DeepSpeed L0 inventory; no training-level support claim."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket


def configuration(explicit=False):
    result = {"train_micro_batch_size_per_gpu": 2,
              "gradient_accumulation_steps": 1,
              "zero_optimization": {"stage": 0},
              "fp16": {"enabled": False}, "bf16": {"enabled": False},
              "steps_per_print": 1000}
    if explicit:
        result["train_batch_size"] = 2
    return result


def run(args):
    # Adapter activation precedes any DeepSpeed import. The oracle is the
    # unmodified installed distribution, without the adapter or source overlay.
    if args.runtime == "shim":
        from jittor_adapters.deepspeed import activate
        activate(device=args.device)
    import numpy as np
    import torch
    shim = hasattr(torch, "_torch_compat_install_context")
    assert shim == (args.runtime == "shim"), "Incorrect torch interpreter"
    if not shim:
        assert not hasattr(torch, "_torch_compat_install_context")
        assert hasattr(torch._C, "_c10d_init"), "Binary PyTorch required"
        if args.device == "npu":
            import torch_npu  # noqa: F401 - registers the torch.npu backend
            assert torch.npu.is_available()
            torch.npu.set_device(0)
    else:
        import jittor as jt
        assert bool(jt.flags.use_cuda) == (args.device == "npu")
        if args.device == "npu":
            assert jt.compiler.has_acl
        fallback_before = jt.core.backend_fallback_count()
    import deepspeed
    from deepspeed.runtime.config import DeepSpeedConfig
    from deepspeed.runtime.engine import DeepSpeedEngine
    assert deepspeed.__version__ == "0.17.6"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    device = "npu:0" if args.device == "npu" else "cpu"

    class BufferedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh())
            self.head = torch.nn.Linear(8, 2)
            self.encoder.register_buffer("scale", torch.tensor([1., 2.], dtype=torch.float32))
            self.register_buffer("enabled", torch.tensor([True, False], dtype=torch.bool))
            self.register_buffer("scratch", torch.tensor([.5], dtype=torch.float32), persistent=False)

        def forward(self, inputs):
            return self.head(self.encoder(inputs))

    def make_model():
        model = BufferedModel().to(device)
        with torch.no_grad():
            for index, (_, value) in enumerate(model.named_parameters()):
                data = np.arange(int(np.prod(list(value.shape))), dtype=np.float32)
                data = ((data.reshape(tuple(value.shape)) % 13) - 6 + index) / 32
                value.copy_(torch.tensor(data, dtype=torch.float32, device=device))
        return model

    def inventory(module):
        def records(items):
            result = []
            for name, value in items:
                assert value.device.type == args.device, name
                if args.device == "npu":
                    assert value.device.index == 0, name
                data = value.detach().cpu().numpy().copy()
                result.append({"name": name, "shape": list(value.shape),
                               "dtype": str(value.dtype).replace("torch.", ""),
                               "storage_dtype": str(data.dtype),
                               "requires_grad": bool(value.requires_grad),
                               "device": value.device.type,
                               "values": data.tolist()})
            return result
        return {"type": type(module).__module__ + "." + type(module).__qualname__,
                "parameters": records(module.named_parameters()),
                "buffers": records(module.named_buffers()),
                "state_keys": sorted(module.state_dict().keys())}

    snapshots = []
    try:
        for explicit in (False, True):
            config = configuration(explicit)
            parsed = DeepSpeedConfig(config)
            fields = ("train_batch_size", "train_micro_batch_size_per_gpu",
                      "gradient_accumulation_steps", "zero_optimization_stage",
                      "gradient_clipping")
            model = make_model()
            result = {"config": {key: getattr(parsed, key) for key in fields},
                      "model": inventory(model)}
            result["config"].update(fp16_enabled=parsed.float16_config.enabled,
                                    bfloat16_enabled=parsed.bfloat16_config.enabled)
            assert len(result["model"]["parameters"]) == 4
            assert len(result["model"]["buffers"]) == 3
            assert "scratch" not in result["model"]["state_keys"]
            if args.device == "npu":
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                returned = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
                assert isinstance(returned, tuple) and len(returned) == 4
                engine = returned[0]
                assert isinstance(engine, DeepSpeedEngine) and engine.module is model
                result["engine"] = inventory(engine)
                result["model_after"] = inventory(engine.module)
                assert result["model"] == result["model_after"]
                result["backend"] = str(torch.distributed.get_backend())
                assert result["backend"] == "hccl"
            snapshots.append(result)
        errors = []
        for invalid in ({"train_micro_batch_size_per_gpu": 0},
                        {"train_micro_batch_size_per_gpu": 2, "train_batch_size": 3,
                         "gradient_accumulation_steps": 1}):
            try:
                DeepSpeedConfig(invalid)
            except (AssertionError, ValueError) as error:
                errors.append(type(error).__name__)
            else:
                raise AssertionError("Invalid batch configuration was accepted")
        rejected = {}
        if shim:
            candidates = [("cpu_engine", configuration())] if args.device == "cpu" else [
                ("zero1", dict(configuration(), zero_optimization={"stage": 1})),
                ("fp16", dict(configuration(), fp16={"enabled": True})),
                ("bf16", dict(configuration(), bf16={"enabled": True})),
                ("profiling", dict(configuration(), wall_clock_breakdown=True)),
                ("unknown", dict(configuration(), unsupported_option=True))]
            for label, config in candidates:
                model = make_model()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                try:
                    deepspeed.initialize(model=model, optimizer=optimizer, config=config)
                except NotImplementedError as error:
                    rejected[label] = str(error)
                else:
                    raise AssertionError("Out-of-scope configuration accepted: " + label)
            jt.sync_all()
            assert jt.core.backend_fallback_count() == fallback_before
        report = {"runtime": args.runtime, "device": args.device,
                  "torch_version": str(torch.__version__),
                  "torch_origin": getattr(torch, "__file__", None),
                  "deepspeed_version": deepspeed.__version__,
                  "dependencies": {name: version(name) for name in ("numpy", "einops", "hjson", "msgpack", "ninja",
                      "packaging", "psutil", "py-cpuinfo", "pydantic", "tqdm")},
                  "deepspeed_origin": deepspeed.__file__,
                  "installed_engine_sha256": hashlib.sha256(Path(deepspeed.runtime.engine.__file__).read_bytes()).hexdigest(),
                  "snapshots": snapshots, "invalid_config_errors": errors,
                  "rejected": rejected, "fallback_delta": 0 if shim else None}
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"runtime": args.runtime, "device": args.device,
                          "cases": len(snapshots), "parameters": 4, "buffers": 3,
                          "rejected": list(rejected), "fallback_delta": report["fallback_delta"]}))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def compare(args):
    expected = json.loads(args.oracle.read_text(encoding="utf-8"))
    actual = json.loads(args.candidate.read_text(encoding="utf-8"))
    assert expected["runtime"] == "oracle" and actual["runtime"] == "shim"
    for key in ("device", "deepspeed_version", "dependencies", "installed_engine_sha256", "snapshots", "invalid_config_errors"):
        assert expected[key] == actual[key], key
    assert actual["rejected"] and actual["fallback_delta"] == 0
    print(json.dumps({"status": "passed", "device": actual["device"], "cases": 2,
                      "scope": "import/config/model" + (" plus NPU Stage0 engine construction" if actual["device"] == "npu" else "")}))


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    child = modes.add_parser("run")
    child.add_argument("--runtime", required=True, choices=("oracle", "shim"))
    child.add_argument("--device", required=True, choices=("cpu", "npu"))
    child.add_argument("--out", required=True, type=Path)
    child = modes.add_parser("compare")
    child.add_argument("--oracle", required=True, type=Path)
    child.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    run(args) if args.mode == "run" else compare(args)


if __name__ == "__main__":
    main()
