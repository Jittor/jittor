"""Run a local Qwen3 checkpoint through Transformers on an Ascend NPU."""

import argparse
import json
import subprocess
import time

import numpy as np
import jittor as jt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local Qwen3 checkpoint directory")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    args = parser.parse_args()

    if not getattr(jt.compiler, "has_acl", 0):
        raise RuntimeError("ACL was not detected; source the CANN environment first")

    # Load on CPU, then migrate explicitly. This keeps model initialization from
    # being mistaken for an accelerator computation by Transformers.
    jt.flags.use_cuda = 0
    jt.flags.use_acl = 0
    jt.flags.use_parallel_op_compiler = 1

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch is not jt:
        raise RuntimeError("set JITTOR_TORCH_SHIM=1 before starting Python")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    )
    jt.flags.use_cuda = 1
    jt.flags.use_acl = 1
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

    started = time.monotonic()
    with jt.log_capture_scope(log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        jt.sync_all(True)
    generate_seconds = time.monotonic() - started

    messages = [entry["msg"].lower() for entry in logs]
    fallbacks = [message for message in messages if "fallback cpu" in message]
    if fallbacks:
        raise RuntimeError("CPU fallback detected during generation: " + fallbacks[0])

    cpu_compile_ops = []
    for index, message in enumerate(messages):
        if "compile cpu" not in message:
            continue
        context = messages[max(0, index - 2) : index]
        if any("arg_reduce" in entry for entry in context):
            cpu_compile_ops.append("arg_reduce")
            continue
        raise RuntimeError("unexpected CPU-compiled operation during generation: " + message)

    generated_ids = generated.numpy()[0].tolist()
    new_ids = generated_ids[input_ids.shape[1] :]
    result = {
        "cpu_compile_count": len(cpu_compile_ops),
        "cpu_compile_ops": cpu_compile_ops,
        "dtype": str(first_parameter.dtype),
        "fallback_count": len(fallbacks),
        "generate_seconds": generate_seconds,
        "has_acl": int(jt.compiler.has_acl),
        "is_cuda": bool(first_parameter.is_cuda),
        "jittor": jt.__version__,
        "load_seconds": load_seconds,
        "model": args.model,
        "new_token_ids": new_ids,
        "parameters": parameter_count,
        "prompt_tokens": int(input_ids.shape[1]),
        "text": tokenizer.decode(new_ids, skip_special_tokens=True),
        "transformers": transformers.__version__,
        "use_acl": int(jt.flags.use_acl),
        "use_cuda": int(jt.flags.use_cuda),
    }
    print("QWEN_RESULT " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
