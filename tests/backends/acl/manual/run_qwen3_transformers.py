"""Run a local Qwen3 checkpoint through Transformers on an Ascend NPU."""

import argparse
import json
import statistics
import subprocess
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from _helpers import capability as _test_capability

import numpy as np
import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks


def main():
    from contextlib import ExitStack as _TestPolicyStack
    with _TestPolicyStack() as _test_policy_stack:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", required=True, help="Local Qwen3 checkpoint directory")
        parser.add_argument(
            "--dtype", choices=("float32", "bfloat16"), default="float32")
        parser.add_argument("--max-new-tokens", type=int, default=1)
        parser.add_argument("--runs", type=int, default=1)
        args = parser.parse_args()
        if args.max_new_tokens < 1 or args.runs < 1:
            parser.error("max-new-tokens and runs must be positive")

        if not _test_capability.check_accelerator('acl', backend=jt).enabled:
            raise RuntimeError("ACL was not detected; source the CANN environment first")

        # Load on CPU, then migrate explicitly. This keeps model initialization from
        # being mistaken for an accelerator computation by Transformers.
        _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=0))
        _test_policy_stack.enter_context(jt.runtime.scope(use_acl=0))
        _test_policy_stack.enter_context(jt.runtime.scope(use_parallel_op_compiler=0))

        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch is not jt:
            raise RuntimeError("set JITTOR_TORCH_SHIM=1 before starting Python")

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        started = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=getattr(torch, args.dtype),
            attn_implementation="eager",
            local_files_only=True,
        )
        jt.runtime.backend_fallback = "error"
        _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1))
        _test_policy_stack.enter_context(jt.runtime.scope(use_acl=1))
        fallback_before = jt.core.backend_fallback_count()
        with forbid_backend_fallbacks():
            model.to(device=torch.device("cuda"))
            model.eval()
            jt.sync_all(True)
            load_seconds = time.monotonic() - started

            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            first_parameter = next(iter(model.parameters()))
            if not first_parameter.is_cuda:
                raise RuntimeError("model parameters are not resident on the accelerator")

            print("NPU_SMI_AFTER_LOAD_BEGIN", flush=True)
            subprocess.run(["npu-smi", "info"], check=True)
            print("NPU_SMI_AFTER_LOAD_END", flush=True)

            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "What is 2+2? Answer briefly."}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = tokenizer(prompt, return_tensors="np")
            input_ids = torch.from_numpy(encoded["input_ids"].astype(np.int64))
            attention_mask = torch.from_numpy(encoded["attention_mask"].astype(np.int64))

            generated_token_samples = []
            generate_samples = []
            for _ in range(args.runs):
                started = time.monotonic()
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                jt.sync_all(True)
                generate_samples.append(time.monotonic() - started)
                generated_ids = generated.detach().cpu().numpy()[0].tolist()
                generated_token_samples.append(
                    generated_ids[int(input_ids.shape[1]):])
                del generated

        fallback_count = jt.core.backend_fallback_count() - fallback_before

        new_ids = generated_token_samples[-1]
        if any(ids != new_ids for ids in generated_token_samples):
            raise RuntimeError(
                "non-deterministic greedy generation: " +
                repr(generated_token_samples))
        result = {
            "dtype": str(first_parameter.dtype),
            "fallback_count": fallback_count,
            "fallback_policy": "error",
            "generate_seconds": statistics.median(generate_samples),
            "generate_median_seconds": statistics.median(generate_samples),
            "generate_samples": generate_samples,
            "generated_token_samples": generated_token_samples,
            "has_acl": int(_test_capability.check_accelerator('acl', backend=jt).enabled),
            "is_cuda": bool(first_parameter.is_cuda),
            "jittor": jt.__version__,
            "load_seconds": load_seconds,
            "model": args.model,
            "new_token_ids": new_ids,
            "parameters": parameter_count,
            "prompt_tokens": int(input_ids.shape[1]),
            "text": tokenizer.decode(new_ids, skip_special_tokens=True),
            "transformers": transformers.__version__,
            "use_acl": int(jt.introspection.policy.runtime.use_cuda),
            "use_cuda": int(jt.introspection.policy.runtime.use_cuda),
        }
        print("QWEN_RESULT " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
