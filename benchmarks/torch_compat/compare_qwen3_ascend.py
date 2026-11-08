"""Compare independently produced Jittor and native torch_npu benchmark JSON."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(jittor, torch, max_relative_l2):
    if jittor["backend"] != "jittor" or torch["backend"] != "torch":
        raise ValueError("supply Jittor and native torch_npu results in that order")
    for key in ("model", "transformers", "dtype", "parameters", "prompt_tokens",
                "new_tokens", "generated_tokens", "fixed_length", "allow_hf32"):
        if jittor[key] != torch[key]:
            raise ValueError("benchmark configuration differs: " + key)
    if (jittor.get("fallback_policy") != "error"
            or not jittor.get("fallback_count_available")
            or jittor.get("fallback_count") != 0):
        raise ValueError("Jittor result does not prove zero backend fallback attempts")
    expected = torch["new_token_ids"]
    for result in (jittor, torch):
        if result["new_token_ids"] != expected or any(
                sample != expected for sample in result["generation_token_samples"]):
            raise ValueError("primary prompt tokens differ across runs or backends")

    def validations(result):
        return {(item["prompt"], item["prompt_tokens"]): item
                for item in result.get("validation", [])}

    jt_prompts, torch_prompts = validations(jittor), validations(torch)
    if not jt_prompts or jt_prompts.keys() != torch_prompts.keys():
        raise ValueError("run both benchmarks with the same --validate-prompts options")
    for key, reference in torch_prompts.items():
        expected = reference["reference_token_ids"]
        for item in (jt_prompts[key], reference):
            if (not item["matches"] or item["reference_token_ids"] != expected
                    or any(ids != expected for ids in item["optimized_token_samples"])):
                raise ValueError("validation prompt tokens differ: " + key[0])

    report = {"prompt_tokens": jittor["prompt_tokens"], "generated_tokens": jittor["generated_tokens"],
              "validation_prompts": len(jt_prompts), "tokens_match": True,
              "fallback_count": jittor["fallback_count"]}
    for key in ("generate_median_seconds", "generation_prefill_median_seconds",
                "decode_tokens_per_second"):
        a, b = jittor[key], torch[key]
        report[key] = {"jittor": a, "torch": b,
                       "jittor_speedup": (a / b if key.endswith("per_second") else b / a)
                       if a and b else None}
    if not jittor.get("logits_output") or not torch.get("logits_output"):
        raise ValueError("run both benchmarks with --logits-output")
    a = np.load(jittor["logits_output"]).astype(np.float64)
    b = np.load(torch["logits_output"]).astype(np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("logits shapes differ or contain non-finite values")
    relative_l2 = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.finfo(float).tiny))
    argmax_match = bool(np.array_equal(a.argmax(-1), b.argmax(-1)))
    if not argmax_match or relative_l2 > max_relative_l2:
        raise ValueError("logits parity failed: relative_l2=%g, argmax_match=%s"
                         % (relative_l2, argmax_match))
    report["logits"] = {"relative_l2": relative_l2,
                        "max_absolute_error": float(np.abs(a - b).max()),
                        "argmax_match": argmax_match,
                        "relative_l2_limit": max_relative_l2}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jittor_json", type=Path)
    parser.add_argument("torch_json", type=Path)
    parser.add_argument("--max-relative-l2", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.max_relative_l2) or args.max_relative_l2 <= 0:
        parser.error("--max-relative-l2 must be finite and positive")
    report = compare(json.loads(args.jittor_json.read_text()),
                     json.loads(args.torch_json.read_text()), args.max_relative_l2)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
