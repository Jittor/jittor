"""Save, reload, merge and generation checks through public HF APIs."""

import gc
from pathlib import Path

import numpy as np

from _release_hf_data import array, load_base, model_class


MERGE_BUDGET = {"logits_max_abs": 1e-3, "logits_max_global_relative": 5e-5,
                "loss_abs": 1e-5}


def on_device(torch, model, device, jt=None):
    values = list(model.named_parameters()) + list(model.named_buffers())
    devices = sorted({str(value.device) for _, value in values})
    misplaced = [(name, str(value.device)) for name, value in values
                 if value.device.type != device]
    if misplaced:
        raise AssertionError("model parameters/buffers are not all on %s: %s" % (device, misplaced))
    if jt is not None:
        for name, value in values:
            backend, _ = jt.core.dispatch_context([value])
            if backend != device:
                raise AssertionError("native tensor %s backend %s differs from %s" %
                                     (name, backend, device))
    return devices


def batch_on_device(batch, device):
    return {key: value.to(device) for key, value in batch.items() if key != "sample_id"}


def evaluation_probe(model, collator, dataset, device, count=2, autocast=None):
    from contextlib import nullcontext
    import torch

    batch = batch_on_device(collator([dataset[index] for index in range(min(count, len(dataset)))]), device)
    model.eval()
    with torch.no_grad(), autocast if autocast is not None else nullcontext():
        output = model(**batch)
    if output.logits.device.type != device or output.loss.device.type != device:
        raise AssertionError("evaluation outputs are not on the requested device")
    logits, loss = array(output.logits), array(output.loss)
    if not np.isfinite(logits).all() or not np.isfinite(loss).all():
        raise AssertionError("nonfinite evaluation logits or loss")
    result = {"input/" + key: array(value) for key, value in batch.items()}
    result.update({"logits": logits, "loss": loss})
    return result, {"logits_dtype": str(output.logits.dtype), "loss_dtype": str(output.loss.dtype),
                    "device": str(output.logits.device), "loss": float(loss)}


def assert_logits_equal(expected, observed, atol, rtol, label):
    if expected.shape != observed.shape or not np.isfinite(observed).all():
        raise AssertionError("%s logits shape/nonfinite mismatch" % label)
    maximum = float(np.max(np.abs(expected.astype(np.float64) - observed.astype(np.float64))))
    if not np.allclose(expected, observed, atol=atol, rtol=rtol):
        raise AssertionError("%s changed full logits, max_abs=%g" % (label, maximum))
    return maximum


def assert_merge_equal(expected, observed, expected_loss, observed_loss, label):
    """Check the separately frozen FP32 LoRA algebra-reordering budget.

    Public reload retains its tighter elementwise allclose contract. Merge
    changes floating-point evaluation from Wx+B(Ax) to (W+BA)x; its independent
    PyTorch calibration and old elementwise failure remain in the run ledger.
    """
    if expected.shape != observed.shape or not np.isfinite(observed).all():
        raise AssertionError("%s logits shape/nonfinite mismatch" % label)
    if not np.isfinite(expected).all():
        raise AssertionError("%s reference logits are nonfinite" % label)
    losses = [np.asarray(expected_loss), np.asarray(observed_loss)]
    if any(value.size != 1 or not np.isfinite(value).all() for value in losses):
        raise AssertionError("%s loss is not a finite scalar" % label)
    maximum = float(np.max(np.abs(expected.astype(np.float64) - observed.astype(np.float64))))
    scale = float(np.max(np.abs(expected.astype(np.float64))))
    relative = maximum / scale if scale else (0.0 if maximum == 0 else float("inf"))
    result = {"logits_max_abs": maximum, "logits_global_reference_max": scale,
              "logits_max_global_relative": relative,
              "loss_abs": abs(float(losses[0].item()) - float(losses[1].item())),
              "budget": dict(MERGE_BUDGET)}
    for metric, limit in MERGE_BUDGET.items():
        if result[metric] > limit:
            raise AssertionError("%s exceeds independent FP32 merge budget: %s=%g > %g" %
                                 (label, metric, result[metric], limit))
    return result


def unwrap_fp32_model(torch, trainer):
    """End AMP training before comparing public save/reload in FP32.

    Accelerate wraps ``forward`` with autocast and converts its returned logits
    to FP32.  Calling that prepared model outside an autocast block therefore
    still uses reduced-precision kernels, unlike a fresh public reload.  The
    public unwrap option removes that wrapper after all training observations.
    """
    model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
    reduced = {str(value.dtype) for value in model.parameters()
               if value.is_floating_point() and value.dtype != torch.float32}
    if reduced:
        raise AssertionError("FP32 lifecycle found non-FP32 master parameters: %s" % sorted(reduced))
    return model


def generation(model, tokenizer, task, device, seed, max_new_tokens):
    import torch

    prompts = (["Hi.", "Please write a friendly greeting and describe a quiet village where "
                "people walk through a green park in the morning, share fresh bread with "
                "their neighbors, read interesting books by the river, and help visitors "
                "find the small library near the old bridge."]
               if task == "qwen-lora" else
               ["translate English to German: Hi.", "summarize: A person walks through a "
                "quiet village in the morning and meets several friends near a green park. "
                "They share fresh bread, read interesting books by the river, help visitors "
                "find the small library near the old bridge, and return home before evening."])
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left" if task == "qwen-lora" else "right"
    batch = tokenizer(prompts, padding=True, truncation=True, max_length=64,
                      return_tensors="pt").to(device)
    lengths = array(batch["attention_mask"]).sum(-1).tolist()
    if len(set(lengths)) < 2:
        raise AssertionError("generation fixture did not produce a variable-length batch")
    model.eval()
    options = {"max_new_tokens": max_new_tokens, "do_sample": False,
               "pad_token_id": tokenizer.pad_token_id}
    with torch.no_grad():
        cached = model.generate(**batch, use_cache=True, **options)
        uncached = model.generate(**batch, use_cache=False, **options)
        if cached.device.type != device or uncached.device.type != device:
            raise AssertionError("generated token IDs landed on the wrong device")
        cached_array, uncached_array = array(cached), array(uncached)
        if not np.array_equal(cached_array, uncached_array):
            raise AssertionError("cached and uncached greedy generation disagree")
        torch.manual_seed(seed + 913)
        sampled = model.generate(**batch, use_cache=True, max_new_tokens=max_new_tokens,
                                 do_sample=True, top_k=10, top_p=0.9, temperature=0.8,
                                 pad_token_id=tokenizer.pad_token_id)
        if sampled.device.type != device:
            raise AssertionError("sampled token IDs landed on the wrong device")
    tokenizer.padding_side = old_padding
    return {"greedy": cached_array, "sampled": array(sampled)}, {
        "input_lengths": lengths, "greedy_cache_exact": True,
        "sampling": {"top_k": 10, "top_p": 0.9, "temperature": 0.8},
        "compute_policy": "FP32 model without autocast", "actual_device": str(cached.device),
        "max_new_tokens": max_new_tokens,
    }


def lifecycle(args, torch, trainer, tokenizer, eval_dataset, collator, base_dir, jt):
    from transformers import AutoTokenizer

    model = unwrap_fp32_model(torch, trainer)
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    original, _ = evaluation_probe(model, collator, eval_dataset, args.device)
    arrays = {"logits_sample": original["logits"][:, :4, :64]
              if original["logits"].ndim == 3 else original["logits"],
              "loss": original["loss"]}
    result = {"final_dir": str(final_dir),
              "compute_policy": "FP32 master parameters; Accelerate AMP forward wrapper removed"}
    original_generation = None
    if args.task != "bert":
        original_generation, result["generation"] = generation(
            model, tokenizer, args.task, args.device, args.seed, args.max_new_tokens)
        arrays.update({"generation/" + key: value for key, value in original_generation.items()})
    reloaded_tokenizer = AutoTokenizer.from_pretrained(str(final_dir), local_files_only=True)
    probe_text = ["A short useful sentence.", "Another sentence with a few more words."]
    if tokenizer(probe_text) != reloaded_tokenizer(probe_text):
        raise AssertionError("tokenizer save/reload changed encoded inputs")
    if args.task == "qwen-lora":
        from peft import PeftModel

        fresh = load_base(torch, args.task, base_dir)
        reloaded = PeftModel.from_pretrained(fresh, str(final_dir))
        result["adapter_saved"] = (final_dir / "adapter_model.safetensors").is_file()
        if not result["adapter_saved"]:
            raise AssertionError("public Trainer.save_model did not save a Safetensors adapter")
    else:
        reloaded = model_class(args.task).from_pretrained(
            str(final_dir), local_files_only=True, attn_implementation="eager")
    reloaded.to(args.device)
    result["reloaded_devices"] = on_device(torch, reloaded, args.device, jt)
    observed, _ = evaluation_probe(reloaded, collator, eval_dataset, args.device)
    result["reload_logits_max_abs"] = assert_logits_equal(
        original["logits"], observed["logits"], args.reload_atol, args.reload_rtol, "public reload")
    if original_generation is not None:
        reloaded_generation, _ = generation(reloaded, reloaded_tokenizer, args.task,
                                            args.device, args.seed, args.max_new_tokens)
        result["reload_generation_exact"] = np.array_equal(
            original_generation["greedy"], reloaded_generation["greedy"])
        if not result["reload_generation_exact"]:
            raise AssertionError("public reload changed greedy generation")
    del reloaded, observed
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    if args.task == "qwen-lora" and not args.skip_merge:
        merged = model.merge_and_unload(safe_merge=True)
        merged_dir = Path(args.output_dir) / "merged"
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        merged_probe, _ = evaluation_probe(merged, collator, eval_dataset, args.device)
        result["merge_numerics"] = assert_merge_equal(
            original["logits"], merged_probe["logits"], original["loss"],
            merged_probe["loss"], "LoRA merge")
        result["merge_logits_max_abs"] = result["merge_numerics"]["logits_max_abs"]
        result["merged_devices"] = on_device(torch, merged, args.device, jt)
        immediate_generation, _ = generation(merged, tokenizer, args.task, args.device,
                                              args.seed, args.max_new_tokens)
        result["merge_generation_exact"] = np.array_equal(
            original_generation["greedy"], immediate_generation["greedy"])
        if not result["merge_generation_exact"]:
            raise AssertionError("LoRA merge changed greedy generation")
        merged_reload = model_class(args.task).from_pretrained(
            str(merged_dir), local_files_only=True, torch_dtype=torch.float32,
            attn_implementation="eager").to(args.device)
        result["merged_reload_devices"] = on_device(torch, merged_reload, args.device, jt)
        reload_probe, _ = evaluation_probe(merged_reload, collator, eval_dataset, args.device)
        result["merged_reload_numerics"] = assert_merge_equal(
            original["logits"], reload_probe["logits"], original["loss"],
            reload_probe["loss"], "merged public reload")
        result["merged_reload_logits_max_abs"] = result["merged_reload_numerics"]["logits_max_abs"]
        result["merged_artifact_reload_logits_max_abs"] = assert_logits_equal(
            merged_probe["logits"], reload_probe["logits"], args.reload_atol, args.reload_rtol,
            "merged artifact public reload")
        merged_generation, _ = generation(merged_reload, tokenizer, args.task, args.device,
                                          args.seed, args.max_new_tokens)
        result["merged_generation_exact"] = np.array_equal(
            original_generation["greedy"], merged_generation["greedy"])
        if not result["merged_generation_exact"]:
            raise AssertionError("merged public reload changed greedy generation")
        result["merged_dir"] = str(merged_dir)
    np.savez(Path(args.output_dir) / "lifecycle.npz", **arrays)
    return result
