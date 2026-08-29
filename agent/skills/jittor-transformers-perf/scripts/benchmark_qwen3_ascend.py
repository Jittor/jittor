"""Benchmark a local Qwen3 checkpoint with Jittor ACL or native torch_npu."""

import argparse
import cProfile
import gc
import json
import os
import statistics
import time

import numpy as np


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("jittor", "torch"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"),
                        default="float32")
    parser.add_argument("--new-tokens", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--logits-output")
    parser.add_argument("--profile-output")
    parser.add_argument("--python-profile-output")
    parser.add_argument("--pipeline-ops", type=int, default=0)
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="eager")
    args = parser.parse_args()

    if args.warmups < 0 or args.samples < 1 or args.new_tokens < 1:
        parser.error(
            "warmups must be nonnegative; samples and new-tokens must be positive")
    if args.profile_output and args.backend != "jittor":
        parser.error("--profile-output is only available for the Jittor backend")

    if args.backend == "jittor":
        if os.environ.get("JITTOR_TORCH_SHIM") != "1":
            raise RuntimeError("set JITTOR_TORCH_SHIM=1 before starting Jittor")

        import jittor as jt

        if not getattr(jt.compiler, "has_acl", 0):
            raise RuntimeError(
                "Jittor did not detect ACL; source the CANN environment")
        jt.flags.use_cuda = 0
        jt.flags.use_acl = 0
        jt.flags.use_parallel_op_compiler = 0

        import torch

        if torch is not jt:
            raise RuntimeError("Jittor does not own the torch namespace")
        device = torch.device("cuda")
        synchronize = lambda: jt.sync_all(True)
        backend_version = jt.__version__
    else:
        import torch
        import torch_npu

        if "site-packages/torch/" not in torch.__file__:
            raise RuntimeError("native PyTorch does not own the torch namespace")
        if not torch.npu.is_available():
            raise RuntimeError("torch_npu cannot access an NPU")
        device = torch.device("npu:0")
        synchronize = torch.npu.synchronize
        backend_version = f"{torch.__version__}/torch_npu-{torch_npu.__version__}"

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    print("BENCHMARK_PHASE load_model", flush=True)
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=model_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    if args.backend == "jittor":
        jt.flags.use_cuda = 1
        jt.flags.use_acl = 1
    model.to(device=device)
    model.eval()
    synchronize()
    load_seconds = time.perf_counter() - load_started
    if args.backend == "jittor":
        torch.nn.Module.set_execution_pipelining(args.pipeline_ops)

    first_parameter = next(iter(model.parameters()))
    if args.backend == "jittor" and not first_parameter.is_cuda:
        raise RuntimeError("Jittor model parameter is not on the NPU")
    if args.backend == "torch" and first_parameter.device.type != "npu":
        raise RuntimeError("PyTorch model parameter is not on the NPU")

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2? Answer briefly."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(prompt, return_tensors="np")
    input_ids = torch.from_numpy(
        encoded["input_ids"].astype(np.int64)).to(device)
    attention_mask = torch.from_numpy(
        encoded["attention_mask"].astype(np.int64)).to(device)
    prompt_tokens = int(input_ids.shape[1])

    def generated_new_ids(output):
        generated_ids = output.detach().cpu().numpy()[0].tolist()
        return generated_ids[prompt_tokens:]

    def run_prefill():
        build_started = time.perf_counter()
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        build_seconds = time.perf_counter() - build_started
        sync_started = time.perf_counter()
        synchronize()
        sync_seconds = time.perf_counter() - sync_started
        return output, build_seconds, sync_seconds

    def run_generate():
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.new_tokens,
                do_sample=False,
                use_cache=True,
            )
        synchronize()
        return output

    print("BENCHMARK_PHASE first_prefill", flush=True)
    started = time.perf_counter()
    output, _, _ = run_prefill()
    first_prefill_seconds = time.perf_counter() - started
    del output
    for _ in range(args.warmups):
        output, _, _ = run_prefill()
        del output

    prefill_samples = []
    prefill_build_samples = []
    prefill_sync_samples = []
    for _ in range(args.samples):
        started = time.perf_counter()
        output, build_seconds, sync_seconds = run_prefill()
        prefill_samples.append(time.perf_counter() - started)
        prefill_build_samples.append(build_seconds)
        prefill_sync_samples.append(sync_seconds)
        del output

    if args.logits_output:
        output, _, _ = run_prefill()
        last_logits = output.logits[:, -1, :].detach().float().cpu().numpy()
        np.save(args.logits_output, last_logits)
        del output, last_logits

    print("BENCHMARK_PHASE first_generate", flush=True)
    started = time.perf_counter()
    generated = run_generate()
    first_generate_seconds = time.perf_counter() - started
    generation_token_samples = [generated_new_ids(generated)]
    print("BENCHMARK_PHASE steady_generate", flush=True)
    for _ in range(args.warmups):
        del generated
        generated = run_generate()
        generation_token_samples.append(generated_new_ids(generated))
    generate_samples = []
    for _ in range(args.samples):
        del generated
        started = time.perf_counter()
        generated = run_generate()
        generate_samples.append(time.perf_counter() - started)
        generation_token_samples.append(generated_new_ids(generated))

    fallback_count = 0
    sdpa_flash_stats = {}
    if args.backend == "jittor":
        with jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100"
        ) as logs:
            output, _, _ = run_prefill()
            validation_generated = run_generate()
        generation_token_samples.append(
            generated_new_ids(validation_generated))
        del output, validation_generated
        fallback_messages = [
            entry["msg"] for entry in logs
            if "fallback cpu" in entry["msg"].lower()
        ]
        fallback_count = len(fallback_messages)
        sdpa_flash_stats = dict(getattr(jt, "_torch_sdpa_flash_stats", {}))
        if fallback_messages:
            raise RuntimeError(
                "CPU fallback detected during inference: " +
                fallback_messages[0] +
                "; SDPA flash stats: " + repr(sdpa_flash_stats))

    new_ids = generated_new_ids(generated)
    if any(ids != new_ids for ids in generation_token_samples):
        raise RuntimeError(
            "non-deterministic greedy generation: " +
            repr(generation_token_samples))
    result = {
        "backend": args.backend,
        "backend_version": backend_version,
        "attn_implementation": args.attn_implementation,
        "dtype": str(first_parameter.dtype),
        "fallback_count": fallback_count,
        "first_generate_seconds": first_generate_seconds,
        "first_prefill_seconds": first_prefill_seconds,
        "generate_median_seconds": statistics.median(generate_samples),
        "generate_p90_seconds": percentile(generate_samples, 0.9),
        "generate_samples": generate_samples,
        "generation_token_samples": generation_token_samples,
        "generated_tokens": len(new_ids),
        "load_seconds": load_seconds,
        "logits_output": args.logits_output,
        "model": args.model,
        "new_token_ids": new_ids,
        "new_tokens": args.new_tokens,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "pipeline_ops": args.pipeline_ops,
        "prefill_build_median_seconds": statistics.median(
            prefill_build_samples),
        "prefill_build_samples": prefill_build_samples,
        "prefill_median_seconds": statistics.median(prefill_samples),
        "prefill_p90_seconds": percentile(prefill_samples, 0.9),
        "prefill_samples": prefill_samples,
        "prefill_sync_median_seconds": statistics.median(prefill_sync_samples),
        "prefill_sync_samples": prefill_sync_samples,
        "prompt_tokens": prompt_tokens,
        "sdpa_flash_backend": sdpa_flash_stats.get("backend"),
        "sdpa_flash_hits": sdpa_flash_stats.get("hits", 0),
        "sdpa_flash_misses": sdpa_flash_stats.get("misses", {}),
        "text": tokenizer.decode(new_ids, skip_special_tokens=True),
        "tokens_per_second": len(new_ids) / statistics.median(generate_samples),
        "transformers": transformers.__version__,
    }
    print("BENCHMARK_RESULT " + json.dumps(result, sort_keys=True), flush=True)

    if args.python_profile_output:
        python_profile_parent = os.path.dirname(
            os.path.abspath(args.python_profile_output))
        os.makedirs(python_profile_parent, exist_ok=True)
        python_profiler = cProfile.Profile()
        profiled_generated = python_profiler.runcall(run_generate)
        python_profiler.dump_stats(args.python_profile_output)
        print(
            "BENCHMARK_PYTHON_PROFILE " + json.dumps({
                "output": os.path.abspath(args.python_profile_output),
            }, sort_keys=True),
            flush=True,
        )
        del profiled_generated

    if args.profile_output:
        with jt.profile_scope(
            warmup=0,
            rerun=0,
            profiler_hide_relay=1,
            profiler_record_shape=1,
        ) as profile_report:
            profiled_generated = run_generate()
        profile_parent = os.path.dirname(os.path.abspath(args.profile_output))
        os.makedirs(profile_parent, exist_ok=True)
        with open(args.profile_output, "w", encoding="utf-8") as output_file:
            json.dump(profile_report, output_file, indent=2)
        print(
            "BENCHMARK_PROFILE " + json.dumps({
                "output": os.path.abspath(args.profile_output),
                "rows": max(0, len(profile_report) - 1),
            }, sort_keys=True),
            flush=True,
        )
        del profiled_generated

    del model, generated, input_ids, attention_mask
    gc.collect()


if __name__ == "__main__":
    main()
