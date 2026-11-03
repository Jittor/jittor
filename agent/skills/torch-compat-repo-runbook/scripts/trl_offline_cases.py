"""Deterministic offline examples for official TRL trainer breadth experiments.

Models and their cross-runtime initial-state transfer belong to the caller.
The tN sequences are synthetic tokenizer fixtures, not question-answer quality
benchmarks. Subclasses only control sample order and observe upstream results.
"""


ALGORITHMS = ("sft", "dpo", "kto", "orpo", "cpo", "reward")


def records(algorithm):
    if algorithm not in ALGORITHMS:
        raise ValueError("Unknown offline method: " + algorithm)
    pairs = [{"prompt": "t%d t%d " % (10 + i, 30 + i),
              "chosen": "t%d t%d" % (50 + i, 70 + i),
              "rejected": "t%d t%d" % (90 + i, 110 + i)} for i in range(10)]
    if algorithm == "sft":
        return [{"text": row["prompt"] + row["chosen"]} for row in pairs]
    if algorithm == "reward":
        return [{"chosen": row["prompt"] + row["chosen"],
                 "rejected": row["prompt"] + row["rejected"]} for row in pairs]
    if algorithm == "kto":
        # Both classes occur in each sequential two-example microbatch.
        return [{"prompt": row["prompt"],
                 "completion": row["chosen"] if index % 2 == 0 else row["rejected"],
                 "label": index % 2 == 0} for index, row in enumerate(pairs)]
    return pairs


def build(args, torch, tokenizer, models, record):
    """Return (official-trainer subclass, JSON metadata) for one offline method.

    ``models(role, kind='causal', copy_from=None)`` must return a model whose
    initial parameters/buffers have already been shared with the oracle.
    ``record(name, tensor_or_number, exact=False)`` snapshots observations.
    The caller owns complete gradient and parameter-update observations.
    """
    from datasets import Dataset
    import trl

    algorithm = args.algorithm
    rows = records(algorithm)
    for row in rows:
        for field, text in row.items():
            if field == "label":
                continue
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not tokens or tokenizer.unk_token_id in tokens:
                raise AssertionError("Synthetic data must use known nonempty tokenizer tokens")
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
                         bf16=False, fp16=False, max_length=32)
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
        configuration.update(max_prompt_length=16, max_completion_length=16,
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
        "scope": "Ten synthetic token sequences; verifies trainer numerics, not answer quality or generalization",
        "reference": "Separate frozen copy of initial policy" if algorithm in ("dpo", "kto") else None,
        "kto_batch_note": "KTO requires actual batch size > 1 for cross-pair KL; ten steps consume twenty examples"
                          if algorithm == "kto" else None,
    }
    if len(trainer.train_dataset) != len(rows):
        raise AssertionError("Official preprocessing unexpectedly changed dataset size")
    return trainer, metadata
