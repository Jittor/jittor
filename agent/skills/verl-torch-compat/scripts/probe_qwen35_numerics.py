#!/usr/bin/env python3
"""Locate the first non-finite Qwen3.5 hidden state on one accelerator."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


def stats(value):
    array = value.detach().float().cpu().numpy()
    finite = np.isfinite(array)
    finite_values = array[finite]
    return {
        "shape": list(array.shape),
        "finite": int(finite.sum()),
        "total": int(array.size),
        "min": float(finite_values.min()) if finite_values.size else None,
        "max": float(finite_values.max()) if finite_values.size else None,
        "mean": float(finite_values.mean()) if finite_values.size else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--mrope-position-ids", action="store_true")
    parser.add_argument("--task", choices=("actor", "critic"), default="actor")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--ppo-prompt", action="store_true")
    parser.add_argument("--generate-response", action="store_true")
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    torch.npu.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_class = (
        AutoModelForImageTextToText
        if args.task == "actor"
        else AutoModelForTokenClassification
    )
    model_kwargs = dict(dtype=dtype, attn_implementation="eager")
    if args.task == "critic":
        model_kwargs["num_labels"] = 1
    model = model_class.from_pretrained(args.model, **model_kwargs)
    if args.fsdp:
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("npu", mesh_shape=(1,), mesh_dim_names=("fsdp",))
        full_state = model.state_dict()
        fsdp_kwargs = dict(
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=dtype,
                reduce_dtype=dtype,
                cast_forward_inputs=True,
            ),
            offload_policy=None,
            reshard_after_forward=True,
        )
        wrap_names = set(getattr(model, "_no_split_modules", ()) or ())
        for _name, module in model.named_modules():
            if module.__class__.__name__ in wrap_names:
                fully_shard(module, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)
        model = model.to(device="npu:0", non_blocking=True)
        set_model_state_dict(
            model,
            full_state,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=False,
                broadcast_from_rank0=True,
            ),
        )
    else:
        model = model.to("npu")
    model = model.train() if args.train else model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6) if args.backward else None
    if args.ppo_prompt:
        encoded = tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": "What is 1 + 1? End the response with #### followed by the number.",
            }],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        prompt = "Give one short reason why the sky looks blue. "
        tokenize_kwargs = {"return_tensors": "pt"}
        if args.sequence_length:
            prompt *= max(args.sequence_length, 1)
            tokenize_kwargs.update(
                truncation=True,
                max_length=args.sequence_length,
                padding="max_length",
            )
        encoded = tokenizer(prompt, **tokenize_kwargs)
    encoded = {key: value.to("npu") for key, value in encoded.items()}
    generated_response = None
    if args.generate_response:
        was_training = model.training
        model.eval()
        with torch.no_grad(), (
            torch.autocast(device_type="npu", dtype=dtype)
            if args.autocast else nullcontext()
        ):
            generated = model.generate(
                **encoded,
                max_new_tokens=8,
                do_sample=False,
                use_cache=True,
            )
        prompt_length = int(encoded["input_ids"].shape[-1])
        generated_response = generated[:, prompt_length:]
        encoded = {
            "input_ids": generated,
            "attention_mask": torch.ones_like(generated),
        }
        model.train(was_training)
    if args.mrope_position_ids:
        positions = torch.arange(encoded["input_ids"].shape[-1])
        encoded["position_ids"] = torch.stack(
            [positions] * 4, dim=0).unsqueeze(1).to("npu")
    autocast = (
        torch.autocast(device_type="npu", dtype=dtype)
        if args.autocast
        else nullcontext()
    )
    grad_context = nullcontext() if args.backward else torch.no_grad()
    with grad_context, autocast:
        output = model(
            **encoded,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        loss = None
        if args.backward:
            if args.task == "actor":
                prediction = output.logits[:, :-1, :]
                labels = encoded["input_ids"][:, 1:]
                selected = torch.nn.functional.log_softmax(
                    prediction, dim=-1).gather(
                        dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                loss = -selected.mean()
            else:
                loss = output.logits.float().square().mean()

    hidden_states = getattr(output, "hidden_states", None)
    report = {
        "schema": 1,
        "model": str(Path(args.model).resolve()),
        "dtype": args.dtype,
        "task": args.task,
        "fsdp": args.fsdp,
        "autocast": args.autocast,
        "train": args.train,
        "sequence_length": int(encoded["input_ids"].shape[-1]),
        "mrope_position_ids": args.mrope_position_ids,
        "backward": args.backward,
        "ppo_prompt": args.ppo_prompt,
        "generate_response": args.generate_response,
        "input_ids": encoded["input_ids"].detach().cpu().numpy().tolist(),
        "logits": stats(output.logits),
        "hidden_states": [],
    }
    if generated_response is not None:
        report["generated_response_ids"] = (
            generated_response.detach().cpu().numpy().tolist())
    if args.task == "actor":
        normalized_logits = output.logits
        log_probs = torch.nn.functional.log_softmax(normalized_logits, dim=-1)
        probs = torch.nn.functional.softmax(normalized_logits, dim=-1)
        report["all_token_log_softmax"] = stats(log_probs)
        report["all_token_softmax"] = stats(probs)
        report["all_token_probability_sum"] = stats(probs.sum(dim=-1))
        report["all_token_logsumexp"] = stats(
            torch.logsumexp(normalized_logits, dim=-1))
        report["all_token_entropy"] = stats(
            torch.logsumexp(normalized_logits, dim=-1)
            - torch.sum(probs * normalized_logits, dim=-1)
        )
    if hidden_states is not None:
        report["hidden_states"] = [stats(value) for value in hidden_states]
    report["first_nonfinite_hidden_state"] = next(
        (index for index, item in enumerate(report["hidden_states"])
         if item["finite"] != item["total"]),
        None,
    )
    if loss is not None:
        report["loss"] = stats(loss)
        loss.backward()
        grad_report = {
            "parameters_with_grad": 0,
            "first_nonfinite_parameter": None,
        }
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            grad_report["parameters_with_grad"] += 1
            finite = bool(torch.isfinite(gradient).all().item())
            if not finite and grad_report["first_nonfinite_parameter"] is None:
                grad_report["first_nonfinite_parameter"] = name
                grad_report["first_nonfinite_gradient"] = stats(gradient)
        report["gradients"] = grad_report
        optimizer.zero_grad()
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
