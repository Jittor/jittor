"""Small deterministic preference dataset for an observed official TRL DPO run.

The caller owns runtime isolation, identical initial buffers, tensor recording,
and before/after evaluation. Reference probabilities are computed by TRL before
updates; prepare both train and evaluation dataloaders before training so the
evaluation reference also remains the original policy.
"""


def records():
    """Return ten distinct preference pairs with short, unambiguous answers."""
    examples = (
        ("What is 2 + 3? Answer:", " 5", " 8"),
        ("What is 9 - 4? Answer:", " 5", " 2"),
        ("What is 3 times 4? Answer:", " 12", " 7"),
        ("What is 8 divided by 2? Answer:", " 4", " 6"),
        ("What is the capital of France? Answer:", " Paris", " Berlin"),
        ("What is the capital of Japan? Answer:", " Tokyo", " Paris"),
        ("How many days are in a week? Answer:", " 7", " 9"),
        ("How many sides does a triangle have? Answer:", " 3", " 5"),
        ("Which is larger: 10 or 2? Answer:", " 10", " 2"),
        ("What is the opposite of hot? Answer:", " cold", " tall"),
    )
    return [dict(prompt=prompt, chosen=chosen, rejected=rejected)
            for prompt, chosen, rejected in examples]


def build(args, torch, tokenizer, config_kwargs, record):
    """Build a full-parameter DPOTrainer with observation-only loss wrapping.

    ``record(name, value, exact=False)`` receives unmodified tensors or numeric
    scalars. The caller supplies phase/step/call prefixes and records all
    parameter gradients and final trained parameters separately. Full-vocabulary
    logits are measured by the separate post-training probe.
    """
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from trl import DPOConfig, DPOTrainer

    class ObservedDPOTrainer(DPOTrainer):
        def _get_train_sampler(self, train_dataset=None):
            # RNG algorithms differ across frameworks. Specify the actual
            # order, while retaining the official loader and training loop.
            dataset = self.train_dataset if train_dataset is None else train_dataset
            return torch.utils.data.SequentialSampler(dataset)

        def get_batch_loss_metrics(self, model, batch, train_eval="train"):
            for name, value in batch.items():
                if torch.is_tensor(value):
                    record("input::" + name, value,
                           exact=name not in ("ref_chosen_logps", "ref_rejected_logps"))
            loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval=train_eval)
            record("loss", loss)
            for name, value in metrics.items():
                record("metric::" + name, value)
            return loss, metrics

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager")
    if model.config.model_type != "qwen3":
        raise ValueError("This evaluation requires the official Qwen3 checkpoint")
    model.config.use_cache = False
    model.train()
    dataset = Dataset.from_list(records())
    settings = dict(config_kwargs)
    settings.update(
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        bf16=False, fp16=False,
        max_length=64, max_prompt_length=48, max_completion_length=16,
        loss_type="sigmoid", beta=0.1, label_smoothing=0.0,
        reference_free=False, sync_ref_model=False,
        precompute_ref_log_probs=True, precompute_ref_batch_size=1,
        disable_dropout=True, use_liger_loss=False, use_logits_to_keep=False,
        padding_free=False, generate_during_eval=False,
        prediction_loss_only=True,
    )
    trainer = ObservedDPOTrainer(
        model=model, args=DPOConfig(**settings), train_dataset=dataset,
        eval_dataset=dataset, processing_class=tokenizer)
    if trainer.ref_model is not None:
        raise AssertionError("DPO precomputation should use the initial policy without a resident reference")
    metadata = {
        "method": "dpo", "records": records(), "sample_count": len(dataset),
        "training_order": "SequentialSampler; each pair once for ten updates",
        "loss_type": "sigmoid", "beta": 0.1,
        "max_length": 64, "max_prompt_length": 48, "max_completion_length": 16,
        "reference": "official TRL precompute_ref_log_probs on the initial policy",
        "reference_initialization_requirement": (
            "Synchronize initial buffers, then prepare train and evaluation dataloaders before updates"),
        "evaluation": "same ten training pairs before and after; not held-out generalization",
        "upstream_trainer": "trl.trainer.dpo_trainer.DPOTrainer",
        "upstream_methods_unchanged": [
            "train", "compute_loss", "dpo_loss", "concatenated_forward",
            "compute_ref_log_probs", "get_train_dataloader", "get_eval_dataloader",
        ],
        "overrides": {
            "_get_train_sampler": "SequentialSampler for identical cross-runtime example order",
            "get_batch_loss_metrics": "Record inputs and super() return values; return them unchanged",
        },
    }
    return trainer, metadata
