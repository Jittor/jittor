"""TP1/TP2/TP4 eager vLLM correctness probe on Jittor/ACL.

Run only with the Jittor Torch shim and explicit NPU platform opt-in. A small
fixed KV allocation avoids using the shim's optimistic device-memory query
to size the cache on a shared machine.
"""

import json
import os
import statistics
import time

assert os.environ.get("JITTOR_TORCH_SHIM") == "1"
assert os.environ.get("JITTOR_VLLM_NPU_PLATFORM") == "1"
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import jittor as jt

jt.flags.use_acl = 1
jt.flags.use_cuda = 1
jt.flags.use_parallel_op_compiler = 0

import torch

# vLLM 0.24 imports a vision warmup path even for Qwen3 text-only inference.
# The Torch shim has no torchvision binary extension; that warmup is a no-op
# for this model, matching the CUDA text-only acceptance harness.
import sys
import types
import transformers.utils as transformers_utils
import transformers.utils.import_utils as transformers_import_utils

transformers_utils.is_torchvision_available = lambda: False
transformers_import_utils.is_torchvision_available = (
    transformers_utils.is_torchvision_available
)
minimax_name = "vllm.model_executor.warmup.minimax_m3_msa_warmup"
minimax_stub = types.ModuleType(minimax_name)
minimax_stub.minimax_m3_msa_warmup = lambda worker: None
sys.modules[minimax_name] = minimax_stub

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

def main():
    tp_size = int(os.environ.get("JT_NPU_VLLM_TP", "1"))
    output_tokens = int(os.environ.get("JT_NPU_VLLM_OUTPUT_TOKENS", "4"))
    batch = int(os.environ.get("JT_NPU_VLLM_BATCH", "1"))
    batch_sequence_raw = os.environ.get("JT_NPU_VLLM_BATCH_SEQUENCE")
    batch_sequence = ([int(value) for value in batch_sequence_raw.split(",")]
                      if batch_sequence_raw else None)
    if batch_sequence:
        batch = max(batch_sequence)
    print("platform", type(current_platform).__name__, current_platform.dist_backend,
          "devices", torch.cuda.device_count(), "tp", tp_size,
          "auto_graph_replay", jt.flags.auto_graph_replay, flush=True)
    assert type(current_platform).__name__ == "JittorNpuPlatform"
    assert 1 <= tp_size <= torch.cuda.device_count()
    assert 1 <= batch <= 4
    assert batch_sequence is None or all(1 <= size <= 4
                                         for size in batch_sequence)

    model = os.environ.get(
        "JT_NPU_VLLM_MODEL",
        "/beegfs/home/tuser028/models/qwen/Qwen3-0.6B",
    )
    dtype = os.environ.get("JT_NPU_VLLM_DTYPE", "bfloat16")
    started = time.perf_counter()
    llm = LLM(
        model=model,
        dtype=dtype,
        tensor_parallel_size=tp_size,
        distributed_executor_backend="mp",
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_custom_all_reduce=True,
        max_model_len=512,
        max_num_seqs=batch,
        async_scheduling=False,
        kv_cache_memory_bytes=512 * 1024 * 1024,
        attention_config={"backend": "FLASH_ATTN"},
    )
    try:
        print("load_s", round(time.perf_counter() - started, 3), flush=True)
        if os.environ.get("JT_NPU_SAMPLE_READBACK_PROBE") == "1":
            from hw_npu_trace import install_npu_sample_readback_probe
            executor = llm.llm_engine.engine_core.engine_core.model_executor
            print("sample_readback_probe", json.dumps(executor.collective_rpc(
                install_npu_sample_readback_probe, timeout=120)), flush=True)
        trace_executor = None
        trace_after_first = os.environ.get("JT_NPU_TRACE_AFTER_FIRST") == "1"
        if os.environ.get("JT_NPU_HOST_TRACE") == "1":
            from hw_npu_trace import install_npu_trace
            trace_executor = (
                llm.llm_engine.engine_core.engine_core.model_executor)
            if not trace_after_first:
                print("trace_install", json.dumps(trace_executor.collective_rpc(
                    install_npu_trace, timeout=120)), flush=True)
        prompt = "Explain in one sentence what a tensor is."
        repeat = int(os.environ.get("JT_NPU_VLLM_PROMPT_REPEAT", "1"))
        if repeat > 1:
            from transformers import AutoTokenizer
            prompt_ids = AutoTokenizer.from_pretrained(model).encode(prompt) * repeat
            request = {"prompt_token_ids": prompt_ids}
        else:
            prompt_ids = None
            request = prompt
        requests = ([{"prompt_token_ids": list(prompt_ids)} for _ in range(batch)]
                    if prompt_ids is not None else [request] * batch)
        requested_logprobs = int(os.environ.get("JT_NPU_VLLM_LOGPROBS", "0"))
        sample = SamplingParams(temperature=0.0, max_tokens=output_tokens,
                                ignore_eos=True,
                                logprobs=requested_logprobs or None)
        repetitions = int(os.environ.get("JT_NPU_VLLM_REPETITIONS", "1"))
        if batch_sequence is not None:
            repetitions = len(batch_sequence)
        assert repetitions >= 1
        runs_s = []
        token_runs = []
        for index in range(repetitions):
            if index == 1 and trace_executor is not None and trace_after_first:
                from hw_npu_trace import install_npu_trace
                print("trace_install", json.dumps(trace_executor.collective_rpc(
                    install_npu_trace, timeout=120)), flush=True)
            request_count = (batch_sequence[index] if batch_sequence is not None
                             else batch)
            started = time.perf_counter()
            outputs = llm.generate(requests[:request_count], sample)
            runs_s.append(round(time.perf_counter() - started, 6))
            token_runs.append([out.outputs[0].token_ids for out in outputs])
        assert all(tokens == token_runs[0][0]
                   for run in token_runs for tokens in run)
        assert all(len(tokens) == output_tokens
                   for run in token_runs for tokens in run)
        oracle_match = None
        oracle_kind = None
        oracle = None
        if (dtype == "bfloat16" and model.rstrip("/").endswith("Qwen3-0.6B")
                and output_tokens <= 64):
            if repeat == 1:
                oracle = ([7281, 11, 10339, 304, 825, 11652, 1128,
                           264, 15626, 2070, 374, 13] +
                          [5005, 11, 10339, 304, 825, 11652, 1128,
                           264, 15626, 2070, 374, 13] * 6)[:output_tokens]
            elif repeat == 13:
                oracle = ([840, 20772, 304, 825, 11652, 1128, 264,
                           15626, 374, 13] * 7)[:output_tokens]
            oracle_kind = "native_pytorch_qwen3" if oracle is not None else None
        elif (model.rstrip("/").endswith("Qwen2.5-3B-Instruct")
              and repeat == 1 and output_tokens <= 16):
            if dtype == "bfloat16" and tp_size == 1:
                oracle = [362, 15626, 374, 264, 35972, 1633, 429,
                          16555, 264, 2745, 70933, 5025, 1948,
                          22879, 11, 24004][:output_tokens]
                oracle_kind = "native_pytorch_qwen25_tp1"
            elif dtype == "bfloat16" and tp_size == 2:
                # Native vLLM-Ascend 0.20.2rc1 TP2 greedy reference on the
                # same 910B3 host. Jittor currently diverges at token 9;
                # keep this regression red until its cause is corrected.
                oracle = [362, 15626, 374, 264, 35972, 1633, 429,
                          16555, 264, 2745, 70933, 5025, 1948,
                          22879, 11, 24004][:output_tokens]
                oracle_kind = "native_vllm_ascend_qwen25_tp2"
            elif dtype == "float16" and tp_size == 2:
                # Actual native vLLM-Ascend TP2 FP16 reference. BF16 differs
                # at a near-tie; keep the two numerical regimes separate.
                oracle = [362, 15626, 374, 264, 35972, 1633, 429,
                          16555, 13482, 4300, 1948, 22879, 11,
                          24004, 1561, 11][:output_tokens]
                oracle_kind = "native_vllm_ascend_qwen25_tp2_fp16"
        if oracle is not None:
            oracle_match = token_runs[0][0] == oracle
            assert oracle_match, f"{oracle_kind} greedy oracle mismatch"
        result = {
            "prompt": prompt,
            "dtype": dtype,
            "prompt_tokens": len(prompt_ids) if prompt_ids is not None else None,
            "batch": batch,
            "batch_sequence": batch_sequence,
            "token_ids": token_runs[0][0],
            "all_token_ids": token_runs[0] if batch > 1 else None,
            "all_runs_token_ids": token_runs if batch_sequence is not None else None,
            "oracle_match": oracle_match,
            "oracle_kind": oracle_kind,
            "text": outputs[0].outputs[0].text,
            "generate_s": statistics.median(runs_s),
            "runs_s": runs_s,
        }
        if requested_logprobs:
            result["logprobs"] = [
                {str(token_id): round(value.logprob, 6)
                 for token_id, value in step.items()}
                for step in outputs[0].outputs[0].logprobs
            ]
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if trace_executor is not None:
            from hw_npu_trace import read_npu_trace
            print("host_trace", json.dumps(trace_executor.collective_rpc(
                read_npu_trace, timeout=120)), flush=True)
        assert len(result["token_ids"]) == output_tokens
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
