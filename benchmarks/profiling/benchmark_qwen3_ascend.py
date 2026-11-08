"""Benchmark a local Qwen3 checkpoint with Jittor ACL or native torch_npu."""

import argparse
import cProfile
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import sys
import statistics
import time

import numpy as np


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]



class _SampleControl:
    # The parent alternates two resident workers; all admission time is excluded.
    def __init__(self, enabled, stream_in=None, stream_out=None, clock=None):
        self.enabled = enabled
        self.stream_in = sys.stdin if stream_in is None else stream_in
        self.stream_out = sys.stdout if stream_out is None else stream_out
        self.clock = time.perf_counter if clock is None else clock
        self.wait_seconds = 0.0
        self.calls = 0
        self.phase = "unspecified"

    def before(self, operation, slot, use_cache=True):
        if not self.enabled:
            return
        started = self.clock()
        self.calls += 1
        event = {"call": self.calls, "phase": self.phase, "operation": operation,
                 "slot": slot, "use_cache": bool(use_cache)}
        print("BENCHMARK_CONTROL_READY " + json.dumps(event, sort_keys=True),
              file=self.stream_out, flush=True)
        line = self.stream_in.readline()
        if not line:
            raise RuntimeError("benchmark controller closed before admission")
        try:
            command = json.loads(line)
        except ValueError as error:
            raise RuntimeError("invalid benchmark admission JSON") from error
        if command != {"go": self.calls}:
            raise RuntimeError("benchmark admission must match the pending call")
        self.wait_seconds += self.clock() - started

    def elapsed(self, started, wait_before):
        return self.clock() - started - (self.wait_seconds - wait_before)


def _assert_npu_values(torch, values, jt=None):
    expected_index = int(torch.npu.current_device())
    count = 0
    for name, value in values:
        if value.device.type != "npu" or value.device.index != expected_index:
            raise RuntimeError("%s is on %s, expected npu:%s" %
                               (name, value.device, expected_index))
        if jt is not None:
            actual = jt.core.dispatch_context([value])
            if tuple(actual) != ("acl", expected_index) or value.location() != "device":
                raise RuntimeError("%s lacks native ACL residency: %s" % (name, actual))
        count += 1
    if not count:
        raise RuntimeError("NPU placement check has no tensor evidence")
    return count


def _prefill_tensors(output):
    yield "logits", output.logits
    cache = output.past_key_values
    if hasattr(cache, "layers"):
        for index, layer in enumerate(cache.layers):
            yield "key/%d" % index, layer.keys
            yield "value/%d" % index, layer.values
    elif isinstance(cache, (tuple, list)):
        for index, pair in enumerate(cache):
            yield "key/%d" % index, pair[0]
            yield "value/%d" % index, pair[1]
    else:
        raise RuntimeError("cannot audit prefill cache tensor residency")


def _npu_memory(torch):
    free, total = torch.npu.mem_get_info()
    return {
        "allocator_live_bytes": int(torch.npu.memory_allocated()),
        "allocator_reserved_bytes": int(torch.npu.memory_reserved()),
        "allocator_peak_live_bytes": int(torch.npu.max_memory_allocated()),
        "allocator_peak_reserved_bytes": int(torch.npu.max_memory_reserved()),
        "device_free_bytes": int(free), "device_total_bytes": int(total),
    }



def _input_arrays(encoded_ids, batch_size, sequence_length, slots):
    # Fixed valid checkpoint tokens, varied by a deterministic circular offset.
    base = np.asarray(encoded_ids, dtype=np.int64)
    if base.ndim != 2 or base.shape[0] != 1 or not base.shape[1]:
        raise ValueError("input fixture requires one nonempty tokenized prompt")
    if batch_size < 1 or sequence_length < 0 or slots < 1:
        raise ValueError("batch/slots must be positive and length nonnegative")
    base = base[0]
    length = sequence_length or int(base.size)
    result = []
    for slot in range(slots):
        rows = []
        for row in range(batch_size):
            rotated = np.roll(base, -(slot * 7 + row))
            rows.append(np.tile(rotated, (length + base.size - 1) // base.size)[:length])
        ids = np.stack(rows).astype(np.int64, copy=False)
        result.append((ids, np.ones_like(ids)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("jittor", "torch"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"),
                        default="float32")
    parser.add_argument("--new-tokens", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=0,
                        help="exact context length; 0 uses the tokenized prompt length")
    parser.add_argument("--input-slots", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--logits-output")
    parser.add_argument("--validation-output", help="NPZ with full prefill logits and greedy IDs")
    parser.add_argument("--report-output", help="structured JSON result under JITTOR_LAB_ROOT")
    parser.add_argument("--profile-output")
    parser.add_argument("--python-profile-output")
    parser.add_argument("--pipeline-ops", type=int, default=0)
    parser.add_argument("--cached-decode", action="store_true",
                        help="also time cached token forwards, excluding their separate prefill")
    parser.add_argument("--control-stdin", action="store_true",
                        help="wait for one matching JSON admission before each synchronized call")
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="eager")
    args = parser.parse_args()

    if args.warmups < 0 or args.samples < 1 or args.new_tokens < 1 or args.pipeline_ops < 0:
        parser.error(
            "warmups/pipeline-ops must be nonnegative; samples and new-tokens must be positive")
    if args.batch_size < 1 or args.sequence_length < 0 or args.input_slots < 1:
        parser.error("batch-size/input-slots must be positive and sequence-length nonnegative")
    if args.profile_output and args.backend != "jittor":
        parser.error("--profile-output is only available for the Jittor backend")

    if args.control_stdin and (args.profile_output or args.python_profile_output):
        parser.error("controlled paired runs require profiling in a separate process")
    if args.cached_decode and args.new_tokens < 2:
        parser.error("cached-decode requires new-tokens >= 2")
    control = _SampleControl(args.control_stdin)

    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("set ASCEND_RT_VISIBLE_DEVICES to an allocated NPU")
    lab = Path(os.environ.get("JITTOR_LAB_ROOT", "")).expanduser().resolve()
    if not os.environ.get("JITTOR_LAB_ROOT"):
        parser.error("set JITTOR_LAB_ROOT for this run")
    checkout = Path(__file__).resolve().parents[2]
    for name in ("logits_output", "validation_output", "report_output",
                 "profile_output", "python_profile_output"):
        raw = getattr(args, name)
        if raw:
            path = Path(raw).expanduser().resolve()
            if lab not in path.parents or checkout in path.parents:
                parser.error("--%s must be outside the checkout under JITTOR_LAB_ROOT" %
                             name.replace("_", "-"))
            if path.exists():
                parser.error("refusing to overwrite existing output: %s" % path)
            path.parent.mkdir(parents=True, exist_ok=True)
            setattr(args, name, str(path))
    jt = None
    if args.backend == "jittor":
        if not os.environ.get("JITTOR_HOME"):
            parser.error("set a dedicated JITTOR_HOME for Jittor")
        cache_home = Path(os.environ["JITTOR_HOME"]).expanduser().resolve()
        if lab not in cache_home.parents or checkout in cache_home.parents:
            parser.error("JITTOR_HOME must be outside the checkout under JITTOR_LAB_ROOT")
        if os.environ.get("JITTOR_TORCH_SHIM") != "1":
            raise RuntimeError("set JITTOR_TORCH_SHIM=1 before starting Jittor")

        import jittor as jt
        from jittor._runtime.fallback import forbid_backend_fallbacks

        jt.runtime.backend_fallback = "error"
        fallback_scope = forbid_backend_fallbacks
        fallback_before = jt.core.backend_fallback_count()

        if not getattr(jt.compiler, "has_acl", 0):
            raise RuntimeError(
                "Jittor did not detect ACL; source the CANN environment")
        jt.flags.use_cuda = 0
        jt.flags.use_acl = 0
        jt.flags.use_parallel_op_compiler = 0

        import torch

        if (torch is jt or torch.Tensor is jt.Var or
                getattr(torch.Tensor, "_frontend_backend", None) is not jt or
                not hasattr(torch, "_torch_compat_install_context")):
            raise RuntimeError("independent Jittor Torch frontend is required")
        device = torch.device("npu:0")
        synchronize = lambda: jt.sync_all(True)
        backend_version = jt.__version__
    else:
        import torch
        import torch_npu

        fallback_scope = nullcontext
        fallback_before = None

        if (not hasattr(torch, "_C") or
                hasattr(torch, "_torch_compat_install_context") or
                "jittor" in sys.modules):
            raise RuntimeError("an independent native PyTorch process is required")
        if not torch.npu.is_available():
            raise RuntimeError("torch_npu cannot access an NPU")
        device = torch.device("npu:0")
        synchronize = torch.npu.synchronize
        backend_version = f"{torch.__version__}/torch_npu-{torch_npu.__version__}"

    if torch.npu.device_count() != 1:
        raise RuntimeError("whole-model benchmark requires exactly one allocated visible NPU")
    torch.npu.set_device(0)
    torch.set_float32_matmul_precision("highest")
    if jt is not None:
        jt.acl_allow_hf32 = False
        if jt.acl_allow_hf32 is not False:
            raise RuntimeError("Jittor HF32 policy did not take effect")
        precision_settings = {"acl_allow_hf32": False}
    else:
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
        if torch.npu.matmul.allow_hf32 is not False or torch.npu.conv.allow_hf32 is not False:
            raise RuntimeError("native torch_npu HF32 policy did not take effect")
        precision_settings = {"matmul_allow_hf32": False, "conv_allow_hf32": False}
    precision_settings["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    if precision_settings["float32_matmul_precision"] != "highest":
        raise RuntimeError("highest float32 matmul policy is required")
    if torch.is_autocast_enabled("npu"):
        raise RuntimeError("benchmark uses explicit model dtype without ambient autocast")

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, MinNewTokensLengthLogitsProcessor

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
    model_tensor_count = _assert_npu_values(
        torch, list(model.named_parameters()) + list(model.named_buffers()), jt)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2? Answer briefly."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(prompt, return_tensors="np")
    input_arrays = _input_arrays(encoded["input_ids"], args.batch_size,
                                 args.sequence_length, args.input_slots)
    input_slots = [(torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device))
                   for ids, mask in input_arrays]
    synchronize()
    for index, (input_ids, attention_mask) in enumerate(input_slots):
        _assert_npu_values(torch, [("input_ids/%d" % index, input_ids),
                                  ("attention_mask/%d" % index, attention_mask)], jt)
    input_ids, attention_mask = input_slots[0]
    prompt_tokens = int(input_ids.shape[1])
    eos_ids = model.generation_config.eos_token_id
    cached_decode_processor = (
        MinNewTokensLengthLogitsProcessor(
            prompt_tokens, args.new_tokens, eos_ids, device=device)
        if args.cached_decode and eos_ids is not None else None
    )
    torch.npu.reset_peak_memory_stats()
    memory_before_inference = _npu_memory(torch)

    def generated_new_ids(output):
        _assert_npu_values(torch, [("generated_ids", output)], jt)
        generated_ids = output.detach().cpu().numpy().tolist()
        if len(generated_ids) != args.batch_size:
            raise RuntimeError("generation batch size changed")
        return [row[prompt_tokens:] for row in generated_ids]

    def run_prefill(slot=0):
        control.before("prefill", slot)
        input_ids, attention_mask = input_slots[slot]
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

    def run_generate(use_cache=True, slot=0):
        control.before("generate", slot, use_cache)
        input_ids, attention_mask = input_slots[slot]
        with fallback_scope():
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.new_tokens,
                    min_new_tokens=args.new_tokens,
                    do_sample=False,
                    use_cache=use_cache,
                )
            synchronize()
            return output

    def run_cached_decode(slot, phase):
        # This prefill and first-token selection are deliberately outside the timer.
        control.phase = "decode_setup"
        prefill, _, _ = run_prefill(slot)
        decode_input_ids, _ = input_slots[slot]
        with torch.no_grad():
            next_token_logits = prefill.logits[:, -1, :].to(
                copy=True, dtype=torch.float32, device=device)
            if cached_decode_processor is not None:
                next_token_logits = cached_decode_processor(decode_input_ids, next_token_logits)
            token = next_token_logits.argmax(dim=-1, keepdim=True)
        cache = prefill.past_key_values
        tokens, decode_outputs = [token], []
        masks = [torch.ones((args.batch_size, prompt_tokens + step + 1),
                            dtype=attention_mask.dtype, device=device)
                 for step in range(args.new_tokens - 1)]
        synchronize()
        _assert_npu_values(torch, [("decode_initial_token", token)] +
                           [("decode_mask/%d" % i, mask) for i, mask in enumerate(masks)], jt)
        del prefill
        control.phase = phase
        control.before("decode", slot)
        with fallback_scope():
            started = time.perf_counter()
            with torch.no_grad():
                for mask in masks:
                    decode_input_ids = torch.cat((decode_input_ids, token), dim=-1)
                    output = model(input_ids=token, attention_mask=mask,
                                   past_key_values=cache, use_cache=True, return_dict=True)
                    cache = output.past_key_values
                    next_token_logits = output.logits[:, -1, :].to(
                        copy=True, dtype=torch.float32, device=device)
                    if cached_decode_processor is not None:
                        next_token_logits = cached_decode_processor(decode_input_ids, next_token_logits)
                    token = next_token_logits.argmax(dim=-1, keepdim=True)
                    tokens.append(token)
                    # Retain every output through the final sync for lazy execution.
                    decode_outputs.append(output)
            synchronize()
            seconds = time.perf_counter() - started
        for index, output in enumerate(decode_outputs):
            _assert_npu_values(torch, _prefill_tensors(output), jt)
        ids = torch.cat(tokens, dim=1).detach().cpu().numpy().copy()
        logits = np.concatenate([output.logits.detach().float().cpu().numpy()
                                 for output in decode_outputs], axis=1)
        if ids.shape != (args.batch_size, args.new_tokens) or not np.isfinite(logits).all():
            raise RuntimeError("cached decode is incomplete or contains nonfinite logits")
        return seconds, ids, logits

    print("BENCHMARK_PHASE first_prefill", flush=True)
    wait_before = control.wait_seconds
    started = time.perf_counter()
    control.phase = "first_prefill"
    output, _, _ = run_prefill()
    first_prefill_seconds = control.elapsed(started, wait_before)
    prefill_tensor_count = _assert_npu_values(torch, _prefill_tensors(output), jt)
    if prefill_tensor_count != 1 + 2 * int(model.config.num_hidden_layers):
        raise RuntimeError("prefill did not expose every layer key/value cache")
    del output
    input_touch_wait_before = control.wait_seconds
    input_touch_started = time.perf_counter()
    control.phase = "prefill_warmup"
    for index in range(max(args.warmups, args.input_slots)):
        output, _, _ = run_prefill(index % args.input_slots)
        del output

    input_slot_touch_seconds = control.elapsed(input_touch_started, input_touch_wait_before)
    prefill_samples = []
    prefill_build_samples = []
    prefill_sync_samples = []
    control.phase = "prefill"
    for index in range(args.samples):
        wait_before = control.wait_seconds
        started = time.perf_counter()
        output, build_seconds, sync_seconds = run_prefill(index % args.input_slots)
        prefill_samples.append(control.elapsed(started, wait_before))
        prefill_build_samples.append(build_seconds)
        prefill_sync_samples.append(sync_seconds)
        del output

    print("BENCHMARK_PHASE first_generate", flush=True)
    wait_before = control.wait_seconds
    started = time.perf_counter()
    control.phase = "first_generate"
    generated = run_generate()
    first_generate_seconds = control.elapsed(started, wait_before)
    generation_token_samples = [generated_new_ids(generated)]
    generation_sample_slots = [0]
    print("BENCHMARK_PHASE steady_generate", flush=True)
    control.phase = "generate_warmup"
    for index in range(max(args.warmups, args.input_slots)):
        del generated
        slot = index % args.input_slots
        generated = run_generate(slot=slot)
        generation_token_samples.append(generated_new_ids(generated))
        generation_sample_slots.append(slot)
    generate_samples = []
    control.phase = "generate"
    for index in range(args.samples):
        del generated
        slot = index % args.input_slots
        wait_before = control.wait_seconds
        started = time.perf_counter()
        generated = run_generate(slot=slot)
        generate_samples.append(control.elapsed(started, wait_before))
        generation_token_samples.append(generated_new_ids(generated))
        generation_sample_slots.append(slot)

    first_decode_seconds = None
    decode_samples, decode_token_samples = [], []
    if args.cached_decode:
        print("BENCHMARK_PHASE cached_decode", flush=True)
        first_decode_seconds, ids, logits = run_cached_decode(0, "first_decode")
        decode_token_samples.append((0, ids.tolist()))
        del ids, logits
        for index in range(max(args.warmups, args.input_slots)):
            slot = index % args.input_slots
            _, ids, logits = run_cached_decode(slot, "decode_warmup")
            decode_token_samples.append((slot, ids.tolist()))
            del ids, logits
        for index in range(args.samples):
            slot = index % args.input_slots
            seconds, ids, logits = run_cached_decode(slot, "decode")
            decode_samples.append(seconds)
            decode_token_samples.append((slot, ids.tolist()))
            del ids, logits

    print("BENCHMARK_PHASE validation", flush=True)
    control.phase = "validation"
    validation_payload = {}
    validated_ids = {}
    for slot in range(args.input_slots):
        output, _, _ = run_prefill(slot)
        if _assert_npu_values(torch, _prefill_tensors(output), jt) != prefill_tensor_count:
            raise RuntimeError("prefill cache tensor evidence changed between runs")
        full_logits = output.logits.detach().float().cpu().numpy().copy()
        if not np.isfinite(full_logits).all():
            raise RuntimeError("full prefill logits contain nonfinite values")
        if slot == 0 and args.logits_output:
            np.save(args.logits_output, full_logits[:, -1, :])
        del output
        cached_validation = run_generate(slot=slot)
        uncached_validation = run_generate(use_cache=False, slot=slot)
        cached_ids = generated_new_ids(cached_validation)
        uncached_ids = generated_new_ids(uncached_validation)
        if cached_ids != uncached_ids:
            raise RuntimeError("cached and uncached greedy generation disagree at slot %d" % slot)
        if any(len(row) != args.new_tokens for row in cached_ids):
            raise RuntimeError("generation did not execute the fixed decode length")
        if args.cached_decode:
            _, decode_ids, decode_logits = run_cached_decode(slot, "decode_validation")
            if decode_ids.tolist() != cached_ids:
                raise RuntimeError("manual cached token forwards disagree with generate")
            control.phase = "validation"
        validated_ids[slot] = cached_ids
        if args.validation_output:
            suffix = "" if slot == 0 else "_slot_%04d" % slot
            validation_payload["logits" + suffix] = full_logits
            validation_payload["input_ids" + suffix] = input_arrays[slot][0]
            validation_payload["attention_mask" + suffix] = input_arrays[slot][1]
            validation_payload["greedy_cached" + suffix] = np.asarray(cached_ids, dtype=np.int64)
            validation_payload["greedy_uncached" + suffix] = np.asarray(uncached_ids, dtype=np.int64)
            if args.cached_decode:
                validation_payload["decode_logits" + suffix] = decode_logits
                validation_payload["greedy_decode" + suffix] = decode_ids
        if args.cached_decode:
            del decode_ids, decode_logits
        del full_logits, cached_validation, uncached_validation
    if args.validation_output:
        np.savez(args.validation_output, **validation_payload)
    del validation_payload
    memory_after_inference = _npu_memory(torch)
    sdpa_flash_stats = (dict(getattr(jt, "_torch_sdpa_flash_stats", {}))
                        if jt is not None else {})
    fallback_count = (jt.core.backend_fallback_count() - fallback_before
                      if fallback_before is not None else 0)
    if fallback_count:
        raise RuntimeError("backend fallback attempted during inference: %d" % fallback_count)

    new_ids = validated_ids[0]
    for slot, ids in zip(generation_sample_slots, generation_token_samples):
        if ids != validated_ids[slot]:
            raise RuntimeError("non-deterministic greedy generation at input slot %d" % slot)
    for slot, ids in decode_token_samples:
        if ids != validated_ids[slot]:
            raise RuntimeError("cached decode trajectory changed between samples")
    generated_positions = sum(len(row) for row in new_ids)
    result = {
        "backend": args.backend,
        "backend_version": backend_version,
        "attn_implementation": args.attn_implementation,
        "dtype": str(first_parameter.dtype),
        "precision_settings": precision_settings,
        "fallback_count": fallback_count if jt is not None else None,
        "fallback_measurement": ("jittor_core_attempt_counter" if jt is not None
                                 else "not_available_in_native_torch_npu"),
        "fallback_policy": "error" if args.backend == "jittor" else None,
        "cached_decode_enabled": args.cached_decode,
        "cached_decode_steps": args.new_tokens - 1 if args.cached_decode else 0,
        "cached_decode_exact": True if args.cached_decode else None,
        "first_decode_seconds": first_decode_seconds,
        "decode_samples": decode_samples,
        "decode_median_seconds": statistics.median(decode_samples) if decode_samples else None,
        "decode_tokens_per_second": (args.batch_size * (args.new_tokens - 1) /
                                     statistics.median(decode_samples)) if decode_samples else None,
        "decode_scope": "cached forwards + HF FP32 greedy/min-new EOS filter/history concat through sync; prefill/initial token/mask allocation/D2H excluded",
        "first_generate_seconds": first_generate_seconds,
        "first_prefill_seconds": first_prefill_seconds,
        "generate_median_seconds": statistics.median(generate_samples),
        "generate_p90_seconds": percentile(generate_samples, 0.9),
        "generate_samples": generate_samples,
        "generation_token_samples": generation_token_samples,
        "generation_sample_slots": generation_sample_slots,
        "timed_sample_slots": [i % args.input_slots for i in range(args.samples)],
        "batch_size": args.batch_size,
        "input_slots": args.input_slots,
        "warmups": args.warmups,
        "samples": args.samples,
        "input_slot_touch_seconds": input_slot_touch_seconds,
        "generated_tokens": generated_positions,
        "generated_tokens_per_sequence": [len(row) for row in new_ids],
        "load_seconds": load_seconds,
        "logits_output": args.logits_output,
        "validation_output": args.validation_output,
        "greedy_cache_exact": True,
        "placement": {"public_device": "npu", "native_backend": "acl" if jt is not None else None,
                      "logical_index": int(torch.npu.current_device()),
                      "model_tensors_checked": model_tensor_count,
                      "prefill_tensors_checked": prefill_tensor_count},
        "memory_before_inference": memory_before_inference,
        "memory_after_inference": memory_after_inference,
        "memory_scope": ("per-runtime allocator counters after model load; Jittor pool counters "
                         "may exclude CANN workspace and are not equivalent to total process HBM; "
                         "device free/total includes all processes"),
        "torch_file": getattr(torch, "__file__", None),
        "torch_tensor_module": torch.Tensor.__module__,
        "timing_scope": "per-call host construction through device synchronization; no D2H; first/JIT separate; controller admission excluded",
        "sample_control": {"enabled": control.enabled, "calls": control.calls,
                           "admission_wait_seconds_excluded": control.wait_seconds},
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
        "text": [tokenizer.decode(row, skip_special_tokens=True) for row in new_ids],
        "tokens_per_second": generated_positions / statistics.median(generate_samples),
        "transformers": transformers.__version__,
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

    if jt is not None and jt.core.backend_fallback_count() != fallback_before:
        raise RuntimeError("backend fallback attempted during final profiling")
    if args.report_output:
        Path(args.report_output).write_text(json.dumps(result, indent=2) + "\n")
    print("BENCHMARK_RESULT " + json.dumps(result, sort_keys=True), flush=True)

    model = generated = input_ids = attention_mask = input_slots = None
    gc.collect()


if __name__ == "__main__":
    main()
