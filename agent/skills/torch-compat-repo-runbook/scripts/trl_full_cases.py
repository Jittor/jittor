"""Official TRL cases using ten short prompts and the complete Qwen3 checkpoint.

The caller supplies independently synchronized pretrained model roles. Training
math is inherited unchanged; only input schedule and observations are controlled.
Reward/value heads and the perturbed GKD teacher are diagnostic fixtures, not
quality-trained reward models or a distilled-model quality benchmark.
"""

ALGORITHMS = ("sft", "dpo", "kto", "orpo", "cpo", "reward")


def offline_records(algorithm):
    from trl_dpo_case import records as preference_records
    pairs = preference_records()
    if algorithm == "reward":
        return [{"chosen": row["prompt"] + row["chosen"],
                 "rejected": row["prompt"] + row["rejected"]} for row in pairs]
    if algorithm == "kto":
        return [{"prompt": row["prompt"],
                 "completion": row["chosen"] if index % 2 == 0 else row["rejected"],
                 "label": index % 2 == 0} for index, row in enumerate(pairs)]
    return pairs


def build_offline(args, torch, tokenizer, models, record):
    """Return (official-trainer subclass, JSON metadata) for one offline method.

    ``models(role, kind='causal', copy_from=None)`` must return a model whose
    initial parameters/buffers have already been shared with the oracle.
    ``record(name, tensor_or_number, exact=False)`` snapshots observations.
    The caller owns complete gradient and parameter-update observations.
    """
    from datasets import Dataset
    import trl

    algorithm = args.algorithm
    rows = offline_records(algorithm)
    for row in rows:
        for field, text in row.items():
            if field == "label":
                continue
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not tokens or tokenizer.unk_token_id in tokens:
                raise AssertionError("Data must use known nonempty tokenizer tokens")
    batch_size = 2 if algorithm == "kto" else 1
    configuration = dict(
        output_dir=str(args.output / "trainer"), use_cpu=args.device == "cpu",
        bf16=False, fp16=False, tf32=False, max_steps=args.steps,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1, learning_rate=args.learning_rate,
        weight_decay=0., optim="sgd", lr_scheduler_type="constant",
        warmup_steps=0, max_grad_norm=0., gradient_checkpointing=False,
        dataloader_num_workers=0, dataloader_pin_memory=False,
        remove_unused_columns=False, logging_steps=1, disable_tqdm=True,
        report_to=[], save_strategy="no", eval_strategy="no", seed=0, data_seed=0,
    )
    configuration.update(getattr(args, "training_kwargs", {}))
    configuration.update(per_device_train_batch_size=batch_size,
                         per_device_eval_batch_size=batch_size, max_steps=args.steps,
                         bf16=False, fp16=False, max_length=64)
    names = {"sft": "SFT", "dpo": "DPO", "kto": "KTO", "orpo": "ORPO",
             "cpo": "CPO", "reward": "Reward"}
    base = getattr(trl, names[algorithm] + "Trainer")
    config_type = getattr(trl, names[algorithm] + "Config")

    def record_inputs(inputs):
        for name, value in inputs.items():
            if torch.is_tensor(value):
                record("input::" + name, value, exact=not value.is_floating_point())
            elif name == "label" and isinstance(value, list):
                # KTO keeps these labels as Python booleans; they are real
                # objective inputs and must participate in the exact check.
                record("input::label", value, exact=True)

    class ObservedOfflineTrainer(base):
        def _get_train_sampler(self, train_dataset=None):
            dataset = self.train_dataset if train_dataset is None else train_dataset
            return torch.utils.data.SequentialSampler(dataset)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            record_inputs(inputs)
            result = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                          num_items_in_batch=num_items_in_batch)
            record("loss", result[0] if isinstance(result, tuple) else result)
            if isinstance(result, tuple) and isinstance(result[1], dict):
                for name, value in result[1].items():
                    if torch.is_tensor(value) or isinstance(value, (float, int)):
                        record("loss_output::" + name, value)
            return result

        def get_batch_loss_metrics(self, *positional, **kwargs):
            # SFT and RewardTrainer do not call this method. Preference
            # trainers use their own unchanged implementation and metrics.
            loss, metrics = super().get_batch_loss_metrics(*positional, **kwargs)
            for name, value in metrics.items():
                if torch.is_tensor(value) or isinstance(value, (float, int)):
                    record("metric::" + name, value)
            return loss, metrics

    kind = "sequence_classification" if algorithm == "reward" else "causal"
    policy = models("policy", kind=kind)
    policy.config.use_cache = False
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.train()
    trainer_kwargs = dict(model=policy, train_dataset=Dataset.from_list(rows),
                          eval_dataset=Dataset.from_list(rows), processing_class=tokenizer)
    if algorithm in ("dpo", "kto"):
        reference = models("reference", copy_from="policy")
        reference.config.use_cache = False
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        trainer_kwargs["ref_model"] = reference
        configuration.update(precompute_ref_log_probs=False, beta=0.1)
    if algorithm == "sft":
        configuration.update(dataset_text_field="text", packing=False)
    elif algorithm in ("dpo", "kto", "orpo", "cpo"):
        configuration.update(max_prompt_length=48, max_completion_length=16,
                             disable_dropout=True, generate_during_eval=False)
        if algorithm in ("dpo", "cpo"):
            configuration["loss_type"] = "sigmoid"
        if algorithm == "kto":
            configuration.update(loss_type="kto", desirable_weight=1., undesirable_weight=1.)
        if algorithm == "cpo":
            configuration["cpo_alpha"] = 1.
        if algorithm == "orpo":
            configuration["beta"] = .1
    elif algorithm == "reward":
        configuration.update(disable_dropout=True, center_rewards_coefficient=None)
    trainer_kwargs["args"] = config_type(**configuration)
    trainer = ObservedOfflineTrainer(**trainer_kwargs)
    metadata = {
        "algorithm": algorithm, "records": rows, "sample_count": len(rows),
        "expected_optimizer_steps": args.steps, "per_device_train_batch_size": batch_size,
        "training_examples_seen": args.steps * batch_size,
        "equivalent_dataset_passes": args.steps * batch_size / len(rows),
        "sampler": "SequentialSampler", "model_kind": kind,
        "upstream_trainer": base.__module__ + "." + base.__name__,
        "observed_subclass": type(trainer).__name__,
        "loss_math": "Unmodified official compute_loss and get_batch_loss_metrics via super()",
        "scope": "Ten short preference examples on pretrained Qwen3-0.6B; not a quality evaluation",
        "reference": "Separate frozen copy of initial policy" if algorithm in ("dpo", "kto") else None,
        "kto_batch_note": "KTO requires actual batch size > 1 for cross-pair KL; ten steps consume twenty examples"
                          if algorithm == "kto" else None,
    }
    if len(trainer.train_dataset) != len(rows):
        raise AssertionError("Official preprocessing unexpectedly changed dataset size")
    return trainer, metadata


def online_records():
    from trl_dpo_case import records as preference_records
    return [{"prompt": row["prompt"], "completion": row["chosen"]}
            for row in preference_records()]


def build_online(args, torch, tokenizer, models, record):
    from datasets import Dataset
    from trl import (GKDConfig, GKDTrainer,
                     OnlineDPOConfig, OnlineDPOTrainer, PPOConfig, PPOTrainer,
                     RLOOConfig, RLOOTrainer)

    policy = models("policy")
    rows = online_records()
    common = dict(args.training_kwargs)
    common.update(per_device_train_batch_size=1, per_device_eval_batch_size=1)
    metadata = {"algorithm": args.algorithm, "model_scope": "full pretrained Qwen3-0.6B, all policy parameters trainable",
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

        common.update(max_length=128, max_new_tokens=4, lmbda=1.0, beta=0.5,
                      temperature=1.0, seq_kd=False, dataset_kwargs={"skip_prepare_dataset": True})
        trainer = ObservedGKD(model=policy, teacher_model=teacher, args=GKDConfig(**common),
                              train_dataset=data, processing_class=tokenizer)
        trainer.generation_config.use_cache = False
        metadata.update(teacher="same pretrained checkpoint with final RMSNorm weight multiplied by 1.01 in NumPy FP32, synchronized across runtimes", lmbda=1.0)
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

        common.update(max_length=128, max_new_tokens=4, temperature=1.0, beta=0.1,
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
        metadata["data_order"] = "sequential fixed prompts; official internal PPO NumPy shuffle retained"
        metadata.update(rollout_batches=trainer.args.num_total_batches,
                        prompts_per_rollout_batch=trainer.local_dataloader_batch_size,
                        total_prompt_visits=trainer.args.num_total_batches * trainer.local_dataloader_batch_size,
                        mini_batches_per_rollout=trainer.args.num_mini_batches,
                        local_rollout_batch_size=trainer.args.local_batch_size,
                        total_episodes=trainer.args.total_episodes)
    else:
        raise ValueError("Unknown online method: " + args.algorithm)
    return trainer, metadata


def build(args, torch, tokenizer, models, record):
    builder = build_offline if args.algorithm in ("kto", "orpo", "cpo", "reward") else build_online
    return builder(args, torch, tokenizer, models, record)
