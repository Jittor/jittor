"""Five pinned official text Trainers; input/observation controls only.

Diffusion Trainers and BCO's optional embedding-based distribution matching are
outside this diagnostic. PRM uses the actual Qwen token-classification model.
"""

ALGORITHMS = ("bco", "iterative_sft", "prm", "nash_md", "xpo")


def records(algorithm):
    if algorithm not in ALGORITHMS:
        raise ValueError(algorithm)
    rows = []
    for i in range(10):
        prompt = f"t{10+i} t{30+i} t{50+i}"
        completion = f"t{70+i} t{90+i}"
        if algorithm == "prm":
            row = dict(prompt=prompt, completions=[completion, f"t{110+i} t{130+i}"],
                       labels=[True, False] if i % 2 == 0 else [False, True])
        elif algorithm == "bco":
            row = dict(prompt=prompt, completion=completion, label=i % 2 == 0)
        elif algorithm == "iterative_sft":
            row = dict(text=prompt + " " + completion)
        else:
            row = dict(prompt=prompt)
        rows.append(row)
    return rows


def build(args, torch, tokenizer, models, record):
    from datasets import Dataset
    import trl

    algorithm = args.algorithm
    rows = records(algorithm)
    kind = "token_classification" if algorithm == "prm" else "causal"
    policy = models("policy", kind=kind)
    common = dict(args.training_kwargs)
    common.update(per_device_train_batch_size=2 if algorithm == "bco" else 1,
                  per_device_eval_batch_size=1)
    names = dict(bco="BCO", iterative_sft="IterativeSFT", prm="PRM", nash_md="NashMD", xpo="XPO")
    base = getattr(trl, names[algorithm] + "Trainer")
    config_type = getattr(trl, names[algorithm] + "Config")

    class Observed(base):
        def _get_train_sampler(self, train_dataset=None):
            return torch.utils.data.SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        def compute_loss(self, model, inputs, *positional, **kwargs):
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    record("input/" + key, value, exact=not value.is_floating_point())
                elif key == "label" and isinstance(value, list):
                    record("input/label", value, exact=True)
            result = super().compute_loss(model, inputs, *positional, **kwargs)
            record("loss", result[0] if isinstance(result, tuple) else result)
            return result

        def training_step(self, *positional, **kwargs):
            result = super().training_step(*positional, **kwargs)
            if algorithm in ("nash_md", "xpo"):
                record("loss", result)
            return result

    kwargs = dict(model=policy, processing_class=tokenizer)
    if algorithm != "iterative_sft":
        kwargs["train_dataset"] = Dataset.from_list(rows)
    if algorithm == "bco":
        common.update(max_length=32, max_prompt_length=16, max_completion_length=16,
                      disable_dropout=True, beta=.1, precompute_ref_log_probs=False,
                      generate_during_eval=False, dataset_num_proc=None)
        reference = models("reference", copy_from="policy")
        reference.requires_grad_(False)
        reference.eval()
        kwargs.update(ref_model=reference, embedding_func=None, embedding_tokenizer=None)
    elif algorithm == "prm":
        common.update(max_length=32, max_prompt_length=16, max_completion_length=16,
                      step_separator=" t200", train_on_last_step_only=False,
                      disable_dropout=True, dataset_num_proc=None)
    elif algorithm == "iterative_sft":
        # Its public step() clips whenever max_grad_norm is not None; zero
        # would zero every gradient, unlike Trainer's positive-value check.
        common.update(max_length=32, max_grad_norm=None)
    else:
        common.update(max_length=32, max_new_tokens=4, temperature=1., beta=.1,
                      use_vllm=False)
        if algorithm == "nash_md":
            common["mixture_coef"] = .5
        else:
            common["alpha"] = .005
        reference = models("reference", copy_from="policy")
        reward = models("reward", kind="sequence_classification")
        for model in (reference, reward):
            model.requires_grad_(False)
            model.eval()
        kwargs.update(ref_model=reference, reward_model=reward)
    kwargs["args"] = config_type(**common)
    trainer = Observed(**kwargs)
    if hasattr(trainer, "generation_config"):
        trainer.generation_config.use_cache = False
    metadata = dict(algorithm=algorithm, records=rows, sample_count=10,
                    expected_optimizer_steps=args.steps, model_kind=kind,
                    upstream_trainer=base.__module__ + "." + base.__name__,
                    objective="Unmodified official Trainer implementations via super()",
                    bco_udm=False if algorithm == "bco" else None,
                    training_examples_seen=args.steps * common["per_device_train_batch_size"],
                    controlled_generation="native policy/reference/geometric-mixture generation with token replay"
                    if algorithm in ("nash_md", "xpo") else None,
                    scope="Tiny random Qwen3, ten diagnostic examples; no quality or full-checkpoint claim")
    return trainer, metadata


def train(trainer, args, metadata):
    if args.algorithm == "iterative_sft":
        for index in range(args.steps):
            trainer.step(texts=[metadata["records"][index]["text"]])
    else:
        trainer.train()
