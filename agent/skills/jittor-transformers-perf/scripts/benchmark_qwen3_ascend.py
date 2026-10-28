"""Benchmark a local Qwen3 checkpoint with Jittor ACL or native torch_npu."""

import argparse
import cProfile
from contextlib import ExitStack, nullcontext
import copy
import gc
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def build_prompt(tokenizer, context_tokens):
    """The deterministic long prompt used by the original 310P comparison."""
    paragraph = (
        "The quarterly infrastructure report describes steady growth in "
        "inference compute demand across the serving fleet, with accelerator "
        "utilization rising as model deployments consolidate onto dedicated "
        "hardware and shared preprocessing pools are retired. ")
    question = "In one short sentence, what does the report describe?"

    def make(reps):
        content = paragraph * reps + "\n\nQuestion: " + question
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    paragraph_tokens = tokenizer(paragraph)["input_ids"]
    overhead = len(tokenizer(make(0))["input_ids"])
    reps = max(1, (context_tokens - overhead + len(paragraph_tokens) - 1)
               // len(paragraph_tokens))
    return make(reps)


def output_parent(path):
    Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)


def compile_report(model):
    """Report observed replay, including an explicit refusal or eager path."""
    compiled = getattr(model, "_compiled_call", None)
    return {
        "present": compiled is not None,
        "type": type(compiled).__name__ if compiled is not None else None,
        "stats": copy.deepcopy(getattr(compiled, "stats", None)),
        "refused": getattr(compiled, "refused", None),
        "graph_refused": getattr(compiled, "_graph_refused", None),
        "last_invalidated": getattr(compiled, "last_invalidated", None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("jittor", "torch"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"),
                        default="float32")
    parser.add_argument("--new-tokens", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--logits-output")
    parser.add_argument("--profile-output")
    parser.add_argument("--python-profile-output")
    parser.add_argument("--pipeline-ops", type=int, default=0)
    parser.add_argument("--context-tokens", type=int,
                        help="approximate long-prompt length; default is the original 2+2 prompt")
    parser.add_argument("--cache-implementation", "--cache",
                        choices=("dynamic", "static"), default="dynamic")
    parser.add_argument("--compile", nargs="?", const="reduce-overhead",
                        choices=("none", "reduce-overhead", "max-autotune"), default="none",
                        help="compile only static-cache decode through Transformers")
    parser.add_argument("--fixed-length", action="store_true",
                        help="require exactly --new-tokens, including EOS suppression")
    parser.add_argument("--allow-hf32", action="store_true",
                        help="explicitly allow reduced-precision FP32 matmul on both backends")
    parser.add_argument("--freeze-linear", action="store_true",
                        help="snapshot Jittor FP16 Linear weights into device NZ buffers for inference")
    parser.add_argument("--torch-jit-compile", choices=("auto", "on", "off"), default="auto",
                        help="native torch_npu operator compile mode; off selects installed binary kernels")
    parser.add_argument("--torch-pack-linear", action="store_true",
                        help="snapshot native torch_npu Linear weights in format 29 before timing")
    parser.add_argument("--validate-prompts", action="store_true",
                        help="compare primary and two extra prompts with uncompiled dynamic generation")
    parser.add_argument("--validation-prompt", action="append", default=[],
                        help="additional user message to validate; repeat for multiple prompts")
    parser.add_argument("--output", help="also write the complete result as JSON")
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="eager")
    args = parser.parse_args()

    if args.warmups < 0 or args.samples < 1 or args.new_tokens < 1:
        parser.error(
            "warmups must be nonnegative; samples and new-tokens must be positive")
    if args.profile_output and args.backend != "jittor":
        parser.error("--profile-output is only available for the Jittor backend")
    if args.context_tokens is not None and args.context_tokens < 1:
        parser.error("--context-tokens must be positive")
    if args.compile != "none" and args.cache_implementation != "static":
        parser.error("--compile requires --cache-implementation static")
    if args.freeze_linear and (args.backend != "jittor" or args.dtype != "float16"):
        parser.error("--freeze-linear requires --backend jittor --dtype float16")
    if args.torch_jit_compile != "auto" and args.backend != "torch":
        parser.error("--torch-jit-compile is only available for native torch_npu")
    if args.torch_pack_linear and (args.backend != "torch" or args.dtype != "float16"):
        parser.error("--torch-pack-linear requires --backend torch --dtype float16")

    with ExitStack() as runtime:
        benchmark(args, runtime)


def benchmark(args, runtime):
    jt = None
    load_scope = ExitStack()
    runtime.callback(load_scope.close)

    if args.backend == "jittor":
        import jittor as jt
        from jittor._runtime.fallback import forbid_backend_fallbacks

        runtime.enter_context(jt.runtime.scope(
            backend_fallback="error", use_parallel_op_compiler=0))
        if args.allow_hf32:
            runtime.enter_context(jt.runtime.scope(acl_allow_hf32=1))
        fallback_scope = forbid_backend_fallbacks
        fallback_before = jt.core.backend_fallback_count()

        if not getattr(jt.compiler, "has_acl", 0):
            raise RuntimeError(
                "Jittor did not detect ACL; source the CANN environment")
        load_scope.enter_context(jt.runtime.scope(use_cuda=0, use_acl=0))

        import torch

        if not hasattr(torch, "_torch_compat_install_context"):
            raise RuntimeError("torch is not the Jittor frontend; install jittor-torch or activate the shim")
        device = torch.device("cuda")
        synchronize = lambda: jt.sync_all(True)
        backend_version = jt.__version__
    else:
        import torch
        import torch_npu

        fallback_scope = nullcontext
        fallback_before = None

        if hasattr(torch, "_torch_compat_install_context") or not hasattr(torch, "_C"):
            raise RuntimeError("native PyTorch does not own the torch namespace")
        if not torch.npu.is_available():
            raise RuntimeError("torch_npu cannot access an NPU")
        if args.torch_jit_compile != "auto":
            torch.npu.set_compile_mode(jit_compile=args.torch_jit_compile == "on")
        device = torch.device("npu:0")
        torch.npu.set_device(device)
        if args.allow_hf32:
            torch.npu.matmul.allow_hf32 = True
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
    load_scope.close()
    if jt is not None:
        runtime.enter_context(jt.runtime.scope(use_cuda=1, use_acl=1))
        runtime.enter_context(forbid_backend_fallbacks())
    model.to(device=device)
    model.eval()
    source_parameters = sum(parameter.numel() for parameter in model.parameters())
    packing_seconds = 0.0
    frozen_linear_count = 0
    torch_packed_linear_count = 0
    if args.freeze_linear:
        from jittor.backends.acl.inference import FrozenLinear, freeze_linear_layers_
        synchronize()
        packing_started = time.perf_counter()
        with torch.no_grad():
            model = freeze_linear_layers_(model, linear_type=torch.nn.Linear)
        synchronize()
        packing_seconds = time.perf_counter() - packing_started
        frozen_linear_count = sum(isinstance(layer, FrozenLinear) for layer in model.modules())
        if not frozen_linear_count:
            raise RuntimeError("--freeze-linear converted no Linear modules")
    if args.torch_pack_linear:
        synchronize()
        packing_started = time.perf_counter()
        packed_parameters = {}
        with torch.no_grad():
            for layer in model.modules():
                if type(layer) is not torch.nn.Linear:
                    continue
                identity = id(layer.weight)
                if identity not in packed_parameters:
                    packed = torch_npu.npu_format_cast(layer.weight.detach().clone(), 29)
                    if torch_npu.get_npu_format(packed) != 29:
                        raise RuntimeError("native Linear weight did not retain NZ format")
                    packed_parameters[identity] = torch.nn.Parameter(packed, requires_grad=False)
                # Rebind only Linear: an embedding may share the source weight
                # and must keep its original ND representation.
                layer.weight = packed_parameters[identity]
                torch_packed_linear_count += 1
        synchronize()
        packing_seconds = time.perf_counter() - packing_started
        if not torch_packed_linear_count:
            raise RuntimeError("--torch-pack-linear converted no Linear modules")
    model.generation_config.disable_compile = args.compile == "none"
    if args.compile != "none":
        from transformers import CompileConfig
        compile_config = CompileConfig(
            fullgraph=False, dynamic=False, mode=args.compile)
        compile_config._compile_all_devices = True
        model.generation_config.compile_config = compile_config
    synchronize()
    load_seconds = time.perf_counter() - load_started
    if args.backend == "jittor":
        torch.nn.Module.set_execution_pipelining(args.pipeline_ops)

    first_parameter = next(iter(model.parameters()))
    if args.backend == "jittor" and not first_parameter.is_cuda:
        raise RuntimeError("Jittor model parameter is not on the NPU")
    if args.backend == "torch" and first_parameter.device.type != "npu":
        raise RuntimeError("PyTorch model parameter is not on the NPU")

    def chat_prompt(message):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": message}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    prompt = (build_prompt(tokenizer, args.context_tokens)
              if args.context_tokens is not None
              else chat_prompt("What is 2+2? Answer briefly."))

    def encode(text):
        encoded = tokenizer(text, return_tensors="np")
        return tuple(torch.from_numpy(encoded[key].astype(np.int64)).to(device)
                     for key in ("input_ids", "attention_mask"))

    input_ids, attention_mask = encode(prompt)
    prompt_tokens = int(input_ids.shape[1])

    def generated_new_ids(output, length=prompt_tokens):
        generated_ids = output.detach().cpu().numpy()[0].tolist()
        return generated_ids[length:]

    def run_prefill():
        with fallback_scope():
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

    def run_generate(max_new=None, inputs=None, generation_config=None,
                     cache=None):
        max_new = args.new_tokens if max_new is None else max_new
        ids, mask = (input_ids, attention_mask) if inputs is None else inputs
        kwargs = {"min_new_tokens": max_new} if args.fixed_length else {}
        if generation_config is not None:
            kwargs["generation_config"] = generation_config
        with fallback_scope():
            with torch.no_grad():
                output = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_new_tokens=max_new,
                    do_sample=False,
                    use_cache=True,
                    cache_implementation=cache or args.cache_implementation,
                    **kwargs,
                )
            synchronize()
            if args.fixed_length and int(output.shape[-1]) != int(ids.shape[-1]) + max_new:
                raise RuntimeError("generation did not produce the requested fixed length")
            return output

    def timed_generate(max_new=None):
        synchronize()
        started = time.perf_counter()
        output = run_generate(max_new)
        return output, time.perf_counter() - started

    print("BENCHMARK_PHASE first_prefill", flush=True)
    synchronize()
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
        synchronize()
        started = time.perf_counter()
        output, build_seconds, sync_seconds = run_prefill()
        prefill_samples.append(time.perf_counter() - started)
        prefill_build_samples.append(build_seconds)
        prefill_sync_samples.append(sync_seconds)
        del output

    if args.logits_output:
        output, _, _ = run_prefill()
        last_logits = output.logits[:, -1, :].detach().float().cpu().numpy()
        output_parent(args.logits_output)
        np.save(args.logits_output, last_logits)
        del output, last_logits

    print("BENCHMARK_PHASE first_generate", flush=True)
    generated, first_generate_seconds = timed_generate()
    generation_token_samples = [generated_new_ids(generated)]
    print("BENCHMARK_PHASE steady_generate", flush=True)
    effective_warmups = max(args.warmups, 6 if args.compile != "none" else 0)
    for _ in range(effective_warmups):
        del generated
        generated = run_generate()
        generation_token_samples.append(generated_new_ids(generated))

    # The original 310P report defines prefill as generate(1). Retain the
    # forward-only metric above, and report this separately for comparable
    # decode latency: (generate(N) - generate(1)) / (N - 1).
    generation_prefill_samples = []
    if args.new_tokens > 1:
        one_token = run_generate(1)
        del one_token
        for _ in range(args.samples):
            one_token, elapsed = timed_generate(1)
            generation_prefill_samples.append(elapsed)
            del one_token
        # Re-establish the full-generation path after the one-token series;
        # any cache reset/capture work must settle outside the timed samples.
        del generated
        generated = run_generate()
        generation_token_samples.append(generated_new_ids(generated))
    generate_samples = []
    for _ in range(args.samples):
        del generated
        generated, elapsed = timed_generate()
        generate_samples.append(elapsed)
        generation_token_samples.append(generated_new_ids(generated))

    sdpa_flash_stats = {}
    if args.backend == "jittor":
        output, _, _ = run_prefill()
        validation_generated = run_generate()
        generation_token_samples.append(
            generated_new_ids(validation_generated))
        del output, validation_generated
        from jittor.compat.diagnostics import sdpa_flash_stats as get_sdpa_stats
        sdpa_flash_stats = copy.deepcopy(get_sdpa_stats(jt))

    measured_compile = compile_report(model)

    new_ids = generated_new_ids(generated)
    if any(ids != new_ids for ids in generation_token_samples):
        raise RuntimeError(
            "non-deterministic greedy generation: " +
            repr(generation_token_samples))
    validation = []
    if args.validate_prompts or args.validation_prompt:
        messages = list(args.validation_prompt)
        if args.validate_prompts:
            messages += ["What is 3+5? Answer briefly.",
                         "Name the capital of France. Answer briefly."]
        reference_config = copy.deepcopy(model.generation_config)
        reference_config.disable_compile = True
        reference_config.compile_config = None
        for validation_prompt in [prompt] + [chat_prompt(message) for message in messages]:
            inputs = encode(validation_prompt)
            length = int(inputs[0].shape[1])
            optimized_samples = []
            for _ in range(2):
                validation_output = run_generate(inputs=inputs)
                optimized_samples.append(generated_new_ids(validation_output, length))
                del validation_output
            reference_output = run_generate(
                inputs=inputs, generation_config=reference_config, cache="dynamic")
            reference_ids = generated_new_ids(reference_output, length)
            del reference_output
            matches = all(ids == reference_ids for ids in optimized_samples)
            validation.append({"prompt": validation_prompt, "prompt_tokens": length,
                               "optimized_token_samples": optimized_samples,
                               "reference_token_ids": reference_ids, "matches": matches})
            if not matches:
                raise RuntimeError("generation differs from uncompiled dynamic reference: "
                                   + json.dumps(validation[-1], ensure_ascii=False))
            del inputs

    generation_prefill_median = (statistics.median(generation_prefill_samples)
                                if generation_prefill_samples else None)
    decode_seconds = ((statistics.median(generate_samples) - generation_prefill_median)
                      / (len(new_ids) - 1)
                      if generation_prefill_median is not None and len(new_ids) > 1 else None)
    result = {
        "backend": args.backend,
        "backend_version": backend_version,
        "torch_jit_compile": (not torch.npu.is_jit_compile_false()
                              if jt is None else None),
        "attn_implementation": args.attn_implementation,
        "dtype": str(first_parameter.dtype),
        "fallback_policy": "error" if args.backend == "jittor" else None,
        "fallback_count_available": jt is not None,
        "cache_implementation": args.cache_implementation,
        "compile": args.compile,
        "compile_observed": measured_compile,
        "context_tokens_requested": args.context_tokens,
        "fixed_length": args.fixed_length,
        "allow_hf32_requested": args.allow_hf32,
        "allow_hf32": (bool(jt.runtime.acl_allow_hf32) if jt is not None
                       else bool(torch.npu.matmul.allow_hf32)),
        "warmups": args.warmups,
        "generation_warmups": effective_warmups,
        "auto_graph_replay": bool(jt.runtime.auto_graph_replay) if jt is not None else None,
        "generation_prefill_samples": generation_prefill_samples,
        "generation_prefill_median_seconds": generation_prefill_median,
        "decode_seconds_per_token": decode_seconds,
        "decode_tokens_per_second": (1.0 / decode_seconds
                                     if decode_seconds is not None and decode_seconds > 0 else None),
        "validation": validation,
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
        "parameters": source_parameters,
        "parameters_after_conversion": sum(parameter.numel() for parameter in model.parameters()),
        "freeze_linear": args.freeze_linear,
        "frozen_linear_count": frozen_linear_count,
        "torch_pack_linear": args.torch_pack_linear,
        "torch_packed_linear_count": torch_packed_linear_count,
        "packing_seconds": packing_seconds,
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
        "torch_file": torch.__file__,
    }

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

    fallback_count = (jt.core.backend_fallback_count() - fallback_before
                      if fallback_before is not None else 0)
    if fallback_count:
        raise RuntimeError("backend fallback attempted during inference: %d" % fallback_count)
    result["fallback_count"] = fallback_count
    result["compile_observed_final"] = compile_report(model)
    if args.output:
        output_parent(args.output)
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(result, output_file, ensure_ascii=False, indent=2)
    print("BENCHMARK_RESULT " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)

    del model, generated, input_ids, attention_mask
    gc.collect()


if __name__ == "__main__":
    main()
