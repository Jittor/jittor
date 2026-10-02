"""Compare four stored Qwen3 construction manifests; never infer Engine support."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

PROBE_SHA = "423dfbfc57c2a5fa36d6d2876e8e8a8d908c305b85978603e5ae1d0c9e7cd63c"
EVIDENCE_PROBE_SHA = "d6a3ab95965697a4d40a848cf452d23e85b52a80227740681a6b124f3c7a385e"
CONFIG = dict(model_type="qwen3", hidden_size=1024, num_hidden_layers=28,
              num_attention_heads=16, num_key_value_heads=8, vocab_size=151936,
              rms_norm_eps=1e-6)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root, stage):
    require(not hasattr(torch, "_torch_compat_install_context") and hasattr(torch, "_C"),
            "independent binary PyTorch oracle required")
    probe = Path(__file__).with_name("qwen3_construct_manifest.py")
    require(hashlib.sha256(probe.read_bytes()).hexdigest() == PROBE_SHA,
            "review the construction probe before accepting a changed source")
    reports = {}
    hashes = {}
    for runtime in ("oracle", "shim"):
        for rank in (0, 1):
            label = runtime + "/rank%d" % rank
            path = root / label / "manifest.json"
            item = json.loads(path.read_text())
            require(item["runtime"] == runtime and item["rank"] == rank,
                    label + ": runtime/rank mismatch")
            require(item["requested_zero_stage"] == stage, label + ": requested stage mismatch")
            require(item["probe_sha256"] in (PROBE_SHA, EVIDENCE_PROBE_SHA),
                    label + ": unverified probe")
            if item["probe_sha256"] == EVIDENCE_PROBE_SHA:
                enhanced = Path(__file__).with_name("qwen3_construct_manifest_evidence.py")
                require(hashlib.sha256(enhanced.read_bytes()).hexdigest() == EVIDENCE_PROBE_SHA,
                        label + ": enhanced probe changed")
                require(item["placement_checked"] == 311 and item["placement_synchronized"] is True,
                        label + ": incomplete physical placement checks")
                require(item["torch_is_shim"] is (runtime == "shim"),
                        label + ": false oracle/runtime identity")
                require(isinstance(item["torch_file"], str) and item["torch_file"],
                        label + ": missing torch provenance")
                require(set(item["versions"]) == {"deepspeed", "transformers", "numpy", "safetensors"}
                        and all(isinstance(v, str) and v for v in item["versions"].values()),
                        label + ": incomplete dependency versions")
                if runtime == "shim":
                    require(type(item["fallback_before"]) is int
                            and item["fallback_before"] >= 0
                            and item["fallback_after"] == item["fallback_before"]
                            and item["fallback_delta"] == 0, label + ": fallback mismatch")
                else:
                    require(item["torch_has_c_extension"] is True
                            and item["fallback_before"] is None
                            and item["fallback_after"] is None
                            and item["fallback_delta"] is None, label + ": oracle evidence mismatch")
            require(item["model_class"] == "Qwen3ForCausalLM" and item["config"] == CONFIG,
                    label + ": model/config mismatch")
            require(item["device"] == "npu" and item["dtype"] == "float32",
                    label + ": device/dtype mismatch")
            require(item["parameter_count"] == len(item["parameters"]) == 310,
                    label + ": incomplete parameters")
            require(item["buffer_count"] == len(item["buffers"]) == 1,
                    label + ": incomplete buffers")
            require(item["buffers"] == {
                "model.rotary_emb.inv_freq":
                {"shape": [64], "dtype": "torch.float32", "device": "npu"}},
                label + ": buffer contract mismatch")
            total = 0
            for name, detail in item["parameters"].items():
                shape = detail["shape"]
                require(isinstance(shape, list) and shape
                        and all(type(n) is int and n > 0 for n in shape),
                        label + ": invalid shape " + name)
                require(detail["dtype"] == "torch.float32" and detail["device"] == "npu",
                        label + ": parameter placement/dtype mismatch " + name)
                total += math.prod(shape)
            require(total == 596049920, label + ": parameter element count mismatch")
            # Legacy reports omit these fields; contradictory added fields must fail.
            if "torch_is_shim" in item:
                require(item["torch_is_shim"] is (runtime == "shim"),
                        label + ": false oracle/runtime identity")
            if runtime == "oracle" and "torch_has_c_extension" in item:
                require(item["torch_has_c_extension"] is True, label + ": no binary oracle")
            if "fallback_delta" in item:
                require(item["fallback_delta"] == (0 if runtime == "shim" else None),
                        label + ": fallback mismatch")
            reports[label] = item
            hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    base = reports["oracle/rank0"]
    for label, item in reports.items():
        require(item["probe_sha256"] == base["probe_sha256"]
                and item["parameters"] == base["parameters"]
                and item["buffers"] == base["buffers"], label + ": manifest mismatch")
    versions = [item.get("versions") for item in reports.values()]
    if any(v is not None for v in versions):
        require(all(isinstance(v, dict) and v for v in versions)
                and all(v == versions[0] for v in versions), "dependency version mismatch")
    gaps = [] if base["probe_sha256"] == EVIDENCE_PROBE_SHA else ["dependency_versions_not_recorded"]
    return dict(status="partial" if gaps else "passed", manifest_comparison="passed",
                evidence_gaps=gaps, manifests=4, parameter_tensors=310, buffers=1,
                requested_zero_stage=stage, engine_construction_verified=False,
                device="npu", dtype="float32", manifest_sha256=hashes,
                identity_and_zero_fallback_evidence=("explicit enhanced fields and probe assertions"
                    if base["probe_sha256"] == EVIDENCE_PROBE_SHA
                    else "fixed construction probe assertions; fields not explicitly recorded"),
                oracle_version=str(torch.__version__),
                comparator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--zero-stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "fresh output required")
    result = verify(args.root, args.zero_stage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    return 2 if result["evidence_gaps"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
