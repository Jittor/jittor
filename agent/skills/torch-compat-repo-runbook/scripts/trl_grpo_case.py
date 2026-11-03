"""A bounded official GRPOTrainer case with observed or replayed rollouts.

Only generation inputs and batch ordering are controlled. Reward normalization,
reference-policy KL, GRPO loss, backward and optimizer updates remain upstream.
Replay is a training-numerics experiment, not online sampling compatibility.
"""

from difflib import SequenceMatcher
import json
from pathlib import Path


def records():
    questions = (
        ("What is 2 + 3?", "5"),
        ("What is 8 - 2?", "6"),
        ("What is 3 times 4?", "12"),
        ("What is 10 divided by 2?", "5"),
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What color is a clear daytime sky?", "blue"),
        ("What is the opposite of hot?", "cold"),
        ("How many days are in one week?", "7"),
        ("What is the chemical formula of water?", "H2O"),
    )
    return [{"prompt": "Question: " + question + "\nAnswer briefly: ", "target": target}
            for question, target in questions]


def answer_reward(completion, target):
    """Dense diagnostic reward: answer similarity plus matching answer length."""
    answer = completion.strip().casefold()
    target = target.strip().casefold()
    similarity = SequenceMatcher(None, answer, target, autojunk=False).ratio()
    brevity = 1.0 / (1.0 + abs(len(answer) - len(target)))
    return 0.8 * similarity + 0.2 * brevity


def _tensor_rows(value):
    return value.detach().cpu().tolist()


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(args, torch, tokenizer, config_kwargs, record):
    """Return (trainer, mutable metadata); record observes tensors immediately."""
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    rows = records()
    metadata = {
        "algorithm": "grpo", "records": rows, "unique_prompts": len(rows),
        "num_generations": 2, "optimizer_updates": args.steps,
        "beta": 0.04, "loss_type": "grpo", "num_iterations": 1,
        "max_prompt_length": 64, "max_completion_length": 16,
        "generation_mode": "oracle_generate_candidate_token_replay",
        "use_model_defaults": False,
        "batch_order": "lexicographic_prompt_then_completion_token_ids",
        "reward_formula": "0.8 * character_sequence_similarity + 0.2 / (1 + abs(answer_length - target_length))",
        "reward_groups": [], "training_groups": [], "rollout_calls": 0,
        "scope": "Full GRPO training math; controlled generated-token replay; clipping need not activate.",
    }

    def reward_function(prompts, completions, target, **kwargs):
        del kwargs
        if len(completions) != 2 or len(set(prompts)) != 1:
            raise AssertionError("Each GRPO update must contain exactly one repeated prompt")
        values = [answer_reward(completion, expected)
                  for completion, expected in zip(completions, target)]
        if len(values) != 2:
            raise AssertionError("Missing completion reward")
        metadata["reward_groups"].append({
            "prompt": prompts[0], "completions": list(completions), "targets": list(target),
            "rewards": values, "nonzero_variance": values[0] != values[1],
        })
        metadata["nonzero_reward_groups"] = sum(group["nonzero_variance"] for group in metadata["reward_groups"])
        return values

    class ObservedGRPOTrainer(GRPOTrainer):
        def _prepare_inputs(self, inputs):
            prepared = super()._prepare_inputs(inputs)
            # Both backends execute the upstream random shuffle; canonicalize
            # afterwards so its backend-specific RNG does not change tensor
            # ordering in numerical comparisons. All batch-aligned values move
            # together, preserving the exact same two-example mean objective.
            if prepared["prompt_ids"].shape[0] != 2:
                raise AssertionError("Expected two generated sequences per group")
            prompts = _tensor_rows(prepared["prompt_ids"])
            completions = _tensor_rows(prepared["completion_ids"])
            order = sorted(range(2), key=lambda i: (prompts[i], completions[i]))
            indices = torch.tensor(order, dtype=torch.long, device=prepared["prompt_ids"].device)
            for name, value in list(prepared.items()):
                if torch.is_tensor(value):
                    if value.ndim and value.shape[0] == 2:
                        prepared[name] = value[indices]
                elif isinstance(value, (list, tuple)) and len(value) == 2:
                    prepared[name] = type(value)(value[index] for index in order)
            for name, value in prepared.items():
                if torch.is_tensor(value):
                    record("prepared::" + name, value,
                           exact=name in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask"))
            group = {name: _tensor_rows(prepared[name]) for name in
                     ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask")}
            reward_group = metadata["reward_groups"][-1]
            targets = reward_group["targets"]
            if len(set(targets)) != 1:
                raise AssertionError("A repeated prompt must have one reward target")
            texts = tokenizer.batch_decode(group["completion_ids"], skip_special_tokens=True)
            rewards = [answer_reward(text, targets[0]) for text in texts]
            if sorted(rewards) != sorted(reward_group["rewards"]):
                raise AssertionError("Canonicalization changed completion rewards")
            group.update(prompt=reward_group["prompt"], completions=texts,
                         rewards=rewards, target=targets[0],
                         before_global_step=int(self.state.global_step))
            metadata["training_groups"].append(group)
            return prepared

        def _calculate_rewards(self, *positional, **kwargs):
            value = super()._calculate_rewards(*positional, **kwargs)
            record("rewards_per_function", value, exact=True)
            return value

        def _get_per_token_logps_and_entropies(self, model, *positional, **kwargs):
            values = super()._get_per_token_logps_and_entropies(model, *positional, **kwargs)
            role = "reference" if model is self.ref_model else "policy"
            for name, value in zip(("logps", "entropies"), values):
                if value is not None:
                    record(role + "::" + name, value)
            if role == "policy" and torch.is_grad_enabled():
                group = metadata["training_groups"][-1]
                if "before_policy_logps" in group:
                    raise AssertionError("Expected one policy training forward per GRPO update")
                group["before_policy_logps"] = _tensor_rows(values[0])
            return values

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            value = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                         num_items_in_batch=num_items_in_batch)
            record("loss", value)
            return value

    configuration = dict(config_kwargs)
    configuration.update(
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=1, num_generations=2, steps_per_generation=1,
        num_iterations=1, shuffle_dataset=False, max_prompt_length=64,
        max_completion_length=16, beta=0.04, loss_type="grpo", scale_rewards=True,
        temperature=1.0, top_p=1.0, top_k=0, use_vllm=False,
        generation_kwargs={"use_cache": False},
        model_init_kwargs={"torch_dtype": "float32", "attn_implementation": "eager",
                           "local_files_only": True},
    )
    trainer = ObservedGRPOTrainer(
        model=str(args.model), args=GRPOConfig(**configuration),
        reward_funcs=reward_function, train_dataset=Dataset.from_list(rows), processing_class=tokenizer,
    )
    trainer.model.config.use_cache = False
    if trainer.ref_model is None:
        raise AssertionError("This GRPO case requires an explicit reference model")
    trainer.ref_model.config.use_cache = False
    metadata["official_trainer"] = GRPOTrainer.__module__ + "." + GRPOTrainer.__name__
    metadata["observed_subclass"] = type(trainer).__name__
    metadata["canonical_batch_ordering_only"] = True

    target_path = Path(args.output) / "rollouts.json"
    rollouts = {"format_version": 2, "records": rows, "calls": []}
    reference_calls = None
    if args.runtime == "jittor":
        if args.reference is None:
            raise ValueError("Candidate GRPO requires an oracle rollout directory")
        reference = json.loads((Path(args.reference) / "rollouts.json").read_text())
        if reference.get("format_version") != 2 or reference.get("records") != rows:
            raise ValueError("Oracle GRPO rollout protocol differs")
        reference_calls = reference["calls"]

    original_generate = trainer.model.generate

    def observed_generate(*positional, **kwargs):
        if positional:
            raise AssertionError("Expected keyword-only upstream generate inputs")
        input_ids, attention_mask = kwargs["input_ids"], kwargs["attention_mask"]
        actual_inputs = {"input_ids": _tensor_rows(input_ids), "attention_mask": _tensor_rows(attention_mask)}
        # Transformers >=4.50 can replace explicit global-default values in a
        # supplied GenerationConfig with checkpoint-specific defaults. Disable
        # that public option, and observe the same official config preparation
        # that generate() performs, including special-token fallback and kwargs.
        if "use_model_defaults" in kwargs:
            raise AssertionError("Upstream unexpectedly supplied generation-default policy")
        effective_config, _ = trainer.model._prepare_generation_config(
            kwargs["generation_config"], use_model_defaults=False,
            **{name: value for name, value in kwargs.items() if name != "generation_config"},
        )
        generation = {"use_model_defaults": False, "config": effective_config.to_dict()}
        index = len(rollouts["calls"])
        if reference_calls is None:
            output = original_generate(use_model_defaults=False, **kwargs)
        else:
            if index >= len(reference_calls):
                raise AssertionError("Candidate requested more rollout batches than oracle")
            expected = reference_calls[index]
            if actual_inputs != expected["inputs"]:
                raise AssertionError("Candidate GRPO prompt tokens/order differ from oracle")
            if generation != expected["generation"]:
                raise AssertionError("Candidate GRPO effective generation configuration differs from oracle")
            output = torch.tensor(expected["output_ids"], dtype=input_ids.dtype, device=input_ids.device)
        output_ids = _tensor_rows(output)
        if len(output_ids) != 2 or any(row[:len(prompt)] != prompt
                                       for row, prompt in zip(output_ids, actual_inputs["input_ids"])):
            raise AssertionError("Generated/replayed output must preserve both prompt prefixes")
        rollouts["calls"].append({"inputs": actual_inputs, "output_ids": output_ids, "generation": generation})
        metadata["rollout_calls"] = len(rollouts["calls"])
        _write(target_path, rollouts)
        record("generation::input_ids", input_ids, exact=True)
        record("generation::attention_mask", attention_mask, exact=True)
        record("generation::output_ids", output, exact=True)
        return output

    # Replacing this instance method controls discrete training inputs only;
    # the candidate still runs official masking, rewards, KL and loss code.
    trainer.model.generate = observed_generate
    trainer._probe_records = metadata["training_groups"]
    _write(target_path, rollouts)
    return trainer, metadata


def _masked_means(logps, masks):
    if len(logps) != len(masks):
        raise AssertionError("Log probabilities and completion masks differ")
    values = []
    for row, mask in zip(logps, masks):
        if len(row) != len(mask) or not sum(mask):
            raise AssertionError("A completion must contain at least one scored token")
        values.append(sum(value * weight for value, weight in zip(row, mask)) / sum(mask))
    return values


def evaluate_rollouts(trainer, args, torch, record):
    """Re-score fixed completions after training with the official logp helper.

    The before value is taken immediately before this group's own update, not
    from a common initial model. This measures a tiny training-set trajectory,
    not reward improvement under fresh generation or held-out generalization.
    """
    del args  # Placement follows the actual model, never a host-side fallback.
    groups = trainer._probe_records
    if not groups or any("before_policy_logps" not in group for group in groups):
        raise AssertionError("Missing pre-update policy log probabilities")
    device = next(trainer.model.parameters()).device
    training = trainer.model.training
    result = {"before_definition": "Immediately before each group's own update; different policy states.",
              "after_definition": "One final policy after all optimizer updates, fixed training completions.",
              "groups": []}
    trainer.model.eval()
    try:
        with torch.no_grad():
            for index, group in enumerate(groups):
                values = {name: torch.tensor(group[name], dtype=torch.long, device=device)
                          for name in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask")}
                input_ids = torch.cat([values["prompt_ids"], values["completion_ids"]], dim=1)
                mask = torch.cat([values["prompt_mask"], values["completion_mask"]], dim=1)
                logps, _ = trainer._get_per_token_logps_and_entropies(
                    trainer.model, input_ids, mask, values["completion_ids"].shape[1],
                    batch_size=2, compute_entropy=False,
                )
                record("evaluation::group/%d/final_policy_logps" % index, logps)
                before_means = _masked_means(group["before_policy_logps"], group["completion_mask"])
                after_logps = _tensor_rows(logps)
                after_means = _masked_means(after_logps, group["completion_mask"])
                rewards = group["rewards"]
                ranking = sorted(range(2), key=lambda row: rewards[row])
                low, high = ranking
                distinct = rewards[low] != rewards[high]
                result["groups"].append({
                    "prompt": group["prompt"], "completions": group["completions"], "rewards": rewards,
                    "before_global_step": group["before_global_step"],
                    "before_mean_completion_logps": before_means,
                    "final_mean_completion_logps": after_means,
                    "mean_logp_changes": [after - before for before, after in zip(before_means, after_means)],
                    "high_reward_index": high if distinct else None,
                    "low_reward_index": low if distinct else None,
                    "before_high_minus_low_logp": before_means[high] - before_means[low] if distinct else None,
                    "final_high_minus_low_logp": after_means[high] - after_means[low] if distinct else None,
                })
    finally:
        trainer.model.train(training)
    return result


def generate_smoke(torch, model, tokenizer, all_prompts=False):
    """Execute unmodified sampling, optionally for all ten diagnostic prompts."""
    from transformers.generation.utils import GenerationMixin
    device = next(model.parameters()).device
    results = []
    training = model.training
    model.eval()
    try:
        for row in records() if all_prompts else records()[:1]:
            inputs = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False)
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.no_grad():
                output = GenerationMixin.generate(
                    model, **inputs, max_new_tokens=16 if all_prompts else 4, do_sample=True,
                    temperature=1.0, top_p=1.0, top_k=0, use_cache=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    disable_compile=True, use_model_defaults=False,
                )
            completion = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append({"prompt": row["prompt"], "input_ids": _tensor_rows(inputs["input_ids"]),
                            "output_ids": _tensor_rows(output), "texts": tokenizer.batch_decode(output),
                            "completion": completion, "reward": answer_reward(completion, row["target"]),
                            "device": str(output.device), "mode": "unmodified_transformers_sampling_smoke"})
    finally:
        model.train(training)
    return results if all_prompts else results[0]
