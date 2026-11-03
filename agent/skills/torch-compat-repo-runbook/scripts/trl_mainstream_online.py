"""Official online TRL trainers on a small, explicitly synthetic Qwen3 case."""

from contextlib import contextmanager


@contextmanager
def observe_backward_loss(accelerator, record):
    """Observe the official scalar before backward without replacing its math."""
    original = accelerator.backward

    def backward(loss, *positional, **kwargs):
        record('loss', loss)
        return original(loss, *positional, **kwargs)

    accelerator.backward = backward
    try:
        yield
    finally:
        accelerator.backward = original


def records():
    return [{"prompt": "t%d t%d t%d" % (10 + index, 30 + index, 50 + index),
             "completion": "t%d t%d" % (70 + index, 90 + index)} for index in range(10)]


def build(args, torch, tokenizer, models, record):
    from datasets import Dataset
    from trl import (GKDConfig, GKDTrainer, GRPOConfig, GRPOTrainer,
                     OnlineDPOConfig, OnlineDPOTrainer, PPOConfig, PPOTrainer,
                     RLOOConfig, RLOOTrainer)

    policy = models("policy")
    rows = records()
    common = dict(args.training_kwargs)
    common.update(per_device_train_batch_size=1, per_device_eval_batch_size=1)
    metadata = {"algorithm": args.algorithm, "model_scope": "random tiny Qwen3, not Qwen3-0.6B",
                "records": rows, "unique_prompts": 10, "expected_optimizer_steps": args.steps,
                "controlled_generation": "native generate and native scores with oracle multinomial-token replay",
                "max_new_tokens": 4}

    def freeze(model):
        model.requires_grad_(False)
        model.eval()
        return model

    class Sequential:
        def _get_train_sampler(self, train_dataset=None):
            dataset = self.train_dataset if train_dataset is None else train_dataset
            return torch.utils.data.SequentialSampler(dataset)

    if args.algorithm == "gkd":
        teacher = freeze(models("teacher"))
        data = Dataset.from_list([{"messages": [{"role": "user", "content": row["prompt"]},
                                                   {"role": "assistant", "content": row["completion"]}]}
                                  for row in rows])

        class ObservedGKD(Sequential, GKDTrainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                for key, value in inputs.items():
                    if torch.is_tensor(value):
                        record("input/" + key, value, exact=True)
                result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
                record("loss", result[0] if return_outputs else result)
                return result

        common.update(max_length=32, max_new_tokens=4, lmbda=1.0, beta=0.5,
                      temperature=1.0, seq_kd=False, dataset_kwargs={"skip_prepare_dataset": True})
        trainer = ObservedGKD(model=policy, teacher_model=teacher, args=GKDConfig(**common),
                              train_dataset=data, processing_class=tokenizer)
        trainer.generation_config.use_cache = False
        metadata.update(teacher="independent fixed random Qwen3, synchronized across runtimes", lmbda=1.0)
    elif args.algorithm == "online_dpo":
        reference = freeze(models("reference", copy_from="policy"))
        reward = freeze(models("reward", kind="sequence_classification"))

        class ObservedOnlineDPO(Sequential, OnlineDPOTrainer):
            def _forward(self, model, *values):
                result = super()._forward(model, *values)
                record("policy_logps" if model is self.model else "reference_logps", result)
                return result

            def training_step(self, *values, **kwargs):
                result = super().training_step(*values, **kwargs)
                record("loss", result)
                return result

        common.update(max_length=32, max_new_tokens=4, temperature=1.0, beta=0.1,
                      loss_type="sigmoid", use_vllm=False)
        trainer = ObservedOnlineDPO(model=policy, ref_model=reference, reward_model=reward,
                                   reward_processing_class=tokenizer, args=OnlineDPOConfig(**common),
                                   train_dataset=Dataset.from_list(rows), processing_class=tokenizer)
        trainer.generation_config.use_cache = False
    elif args.algorithm in ("ppo", "rloo"):
        reference = freeze(models("reference", copy_from="policy"))
        reward = freeze(models("reward", kind="sequence_classification"))
        tokenized = Dataset.from_list([{"input_ids": tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]}
                                       for row in rows])
        common.update(response_length=4, local_rollout_forward_batch_size=2,
                      num_sample_generations=0, num_ppo_epochs=1, num_mini_batches=1,
                      temperature=1.0, whiten_rewards=False, missing_eos_penalty=None,
                      total_episodes=args.steps, dataset_num_proc=None)
        if args.algorithm == "ppo":
            if args.steps % 2 == 0:
                common.update(per_device_train_batch_size=2, num_mini_batches=2)
            value = models("value", kind="sequence_classification")
            trainer = PPOTrainer(args=PPOConfig(**common), processing_class=tokenizer, model=policy,
                                 ref_model=reference, reward_model=reward, value_model=value,
                                 train_dataset=tokenized, eval_dataset=tokenized)
            metadata.update(roles="policy/value trained; reference/reward fixed", generated_sequences=args.steps)
        else:
            common.update(per_device_train_batch_size=2, total_episodes=args.steps * 2,
                          rloo_k=2, normalize_reward=False, normalize_advantage=False)
            if args.steps % 2 == 0:
                common["num_mini_batches"] = 2
            trainer = RLOOTrainer(config=RLOOConfig(**common), processing_class=tokenizer, policy=policy,
                                  ref_policy=reference, reward_model=reward,
                                  train_dataset=tokenized, eval_dataset=tokenized)
            metadata.update(rloo_k=2, generated_sequences=args.steps * 2)
        # The official constructors hardcode shuffle=True. Only the input
        # schedule changes; the same public DataLoader/Accelerate path is used.
        loader = torch.utils.data.DataLoader(tokenized, batch_size=trainer.local_dataloader_batch_size,
                                             shuffle=False, collate_fn=trainer.data_collator, drop_last=True)
        trainer.dataloader = trainer.accelerator.prepare(loader)
        metadata["data_order"] = "sequential synthetic prompts; official internal PPO NumPy shuffle retained"
        metadata.update(rollout_batches=trainer.args.num_total_batches,
                        prompts_per_rollout_batch=trainer.local_dataloader_batch_size,
                        total_prompt_visits=trainer.args.num_total_batches * trainer.local_dataloader_batch_size,
                        mini_batches_per_rollout=trainer.args.num_mini_batches,
                        local_rollout_batch_size=trainer.args.local_batch_size,
                        total_episodes=trainer.args.total_episodes)
    elif args.algorithm == "grpo":
        # GRPO constructs its reference from model.config._name_or_path.
        # A tiny oracle-created checkpoint supplies those exact initial bytes.
        models.checkpoint("policy")

        def diagnostic_reward(completion_ids, **kwargs):
            del kwargs
            return [sum((index + 1) * int(token) for index, token in enumerate(row)) / 1024.0
                    for row in completion_ids]

        class ObservedGRPO(GRPOTrainer):
            def _prepare_inputs(self, inputs):
                prepared = super()._prepare_inputs(inputs)
                sequences = prepared["completion_ids"].detach().cpu().tolist()
                indices = torch.tensor(sorted(range(len(sequences)), key=lambda i: sequences[i]),
                                       dtype=torch.long, device=prepared["completion_ids"].device)
                for key, value in list(prepared.items()):
                    if torch.is_tensor(value) and value.ndim and value.shape[0] == len(sequences):
                        prepared[key] = value[indices]
                    if torch.is_tensor(prepared[key]):
                        record("prepared/" + key, prepared[key], exact=key.endswith(("ids", "mask")))
                return prepared

            def compute_loss(self, *values, **kwargs):
                result = super().compute_loss(*values, **kwargs)
                record("loss", result)
                return result

        common.update(per_device_train_batch_size=2, num_generations=2, steps_per_generation=1,
                      num_iterations=1, shuffle_dataset=False, max_prompt_length=16, max_completion_length=4,
                      beta=0.04, loss_type="grpo", temperature=1.0, top_p=1.0, top_k=0,
                      generation_kwargs={"use_cache": False})
        trainer = ObservedGRPO(model=policy, args=GRPOConfig(**common), reward_funcs=diagnostic_reward,
                               train_dataset=Dataset.from_list(rows), processing_class=tokenizer)
        models.adopt("reference", trainer.ref_model, copy_from="policy")
        freeze(trainer.ref_model)
        metadata.update(generated_sequences=args.steps * 2, reward="synthetic position-weighted token sum /1024",
                        batch_order="canonical completion-token order within one two-answer prompt group")
    else:
        raise ValueError("Unknown online method: " + args.algorithm)
    return trainer, metadata
