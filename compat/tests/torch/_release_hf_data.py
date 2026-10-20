"""Public Hugging Face model/data construction for release workflow acceptance.

This module does not select a runtime or import torch at module import time.
Model and data paths are fixed by the caller; no network download is permitted.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


TASKS = ("bert", "qwen-lora", "t5")


def array(tensor):
    value = tensor.detach()
    if str(value.dtype) in ("torch.bfloat16", "bfloat16"):
        value = value.float()
    return value.cpu().numpy().copy()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path, limit=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("empty JSONL dataset: %s" % path)
    return rows


def model_class(task):
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoModelForSeq2SeqLM,
    )

    return {
        "bert": AutoModelForSequenceClassification,
        "qwen-lora": AutoModelForCausalLM,
        "t5": AutoModelForSeq2SeqLM,
    }[task]


def configure(config, task, compact=False, dropout=None):
    if compact:
        if task == "bert":
            config.hidden_size = 32
            config.num_hidden_layers = 1
            config.num_attention_heads = 2
            config.intermediate_size = 64
            config.max_position_embeddings = 128
        elif task == "qwen-lora":
            config.hidden_size = 32
            config.num_hidden_layers = 1
            config.num_attention_heads = 2
            config.num_key_value_heads = 1
            config.intermediate_size = 64
            config.head_dim = 16
            config.max_position_embeddings = 256
        else:
            config.d_model = 32
            config.d_ff = 64
            config.num_layers = 1
            config.num_decoder_layers = 1
            config.num_heads = 2
            config.d_kv = 16
    if task == "bert":
        config.num_labels = 2
        config.problem_type = "single_label_classification"
    if dropout is not None:
        for field in (
            "hidden_dropout_prob", "attention_probs_dropout_prob", "dropout_rate",
            "attention_dropout", "classifier_dropout", "hidden_dropout", "activation_dropout",
        ):
            if hasattr(config, field) and getattr(config, field) is not None:
                setattr(config, field, dropout)
    config._attn_implementation = "eager"
    return config


def load_base(torch, task, path, compact=False, dropout=None, random_init=False):
    from transformers import AutoConfig

    config = configure(AutoConfig.from_pretrained(path, local_files_only=True),
                       task, compact, dropout)
    constructor = model_class(task)
    if compact or random_init:
        return constructor.from_config(config)
    return constructor.from_pretrained(
        path, config=config, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    )


def attach_lora(model):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    ))


def initialize(args, torch):
    from transformers import AutoTokenizer

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("initialization destination already exists: %s" % output)
    output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    base = load_base(torch, args.task, args.model_dir, args.compact, args.dropout)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    save_base = args.task != "qwen-lora" or args.compact
    base_dir = output / "base" if save_base else Path(args.model_dir).resolve()
    if save_base:
        base.save_pretrained(str(base_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(base_dir))
    adapter_dir = None
    if args.task == "qwen-lora":
        adapter_dir = output / "adapter"
        attach_lora(base).save_pretrained(str(adapter_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(adapter_dir))
    report = {
        "task": args.task, "seed": args.seed, "compact": args.compact,
        "dropout": args.dropout, "source_model_dir": str(Path(args.model_dir).resolve()),
        "initial_model_dir": str(base_dir.resolve()),
        "initial_adapter_dir": str(adapter_dir.resolve()) if adapter_dir else None,
    }
    (output / "initialization.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def resolve_initial(args):
    if args.initialization_dir:
        report = json.loads((Path(args.initialization_dir) / "initialization.json").read_text())
        if report["task"] != args.task:
            raise ValueError("initialization task does not match --task")
        if args.initial_model_dir or args.initial_adapter_dir:
            raise ValueError("--initialization-dir cannot be combined with explicit initial paths")
        return report["initial_model_dir"], report.get("initial_adapter_dir"), report
    if not args.initial_model_dir:
        raise ValueError("provide --initialization-dir or --initial-model-dir")
    return args.initial_model_dir, args.initial_adapter_dir, None


def load_model(torch, task, base_dir, adapter_dir=None):
    base = load_base(torch, task, base_dir)
    if task == "qwen-lora":
        if not adapter_dir:
            raise ValueError("qwen-lora requires an initialized --initial-adapter-dir")
        from peft import PeftModel

        return PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    return base


def instruction_text(row):
    if "text" in row and str(row["text"]).strip():
        return str(row["text"])
    pieces = [str(row.get("instruction", "")), str(row.get("input", "")),
              str(row.get("output", ""))]
    return "\n".join(piece for piece in pieces if piece.strip())


class EncodedDataset:
    def __init__(self, rows, tokenizer, task, max_length, max_target_length,
                 include_sample_id=False):
        self.items = []
        for index, row in enumerate(rows):
            if task == "bert":
                item = tokenizer(str(row["text"]), truncation=True, max_length=max_length)
                item["labels"] = int(row["label"])
            elif task == "qwen-lora":
                item = tokenizer(instruction_text(row), truncation=True, max_length=max_length)
                if len(item["input_ids"]) < 2:
                    raise ValueError("causal training text must contain at least two tokens")
                item["labels"] = list(item["input_ids"])
            else:
                source = row.get("source", row.get("instruction"))
                target = row.get("target", row.get("output"))
                if source is None or target is None:
                    raise ValueError("T5 requires source/target or instruction/output fields")
                item = tokenizer(str(source), truncation=True, max_length=max_length)
                item["labels"] = tokenizer(text_target=str(target), truncation=True,
                                           max_length=max_target_length)["input_ids"]
            if include_sample_id:
                item["sample_id"] = index
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return dict(self.items[index])


class RecordingCollator:
    def __init__(self, torch, tokenizer, task, model):
        from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding

        self.torch = torch
        self.tokenizer = tokenizer
        self.task = task
        self.delegate = (DataCollatorWithPadding(tokenizer) if task == "bert" else
                         DataCollatorForSeq2Seq(tokenizer, model=model if task == "t5" else None,
                                               label_pad_token_id=-100))

    def __call__(self, features):
        features = [dict(item) for item in features]
        sample_ids = [item.pop("sample_id") for item in features] if "sample_id" in features[0] else None
        batch = self.delegate(features)
        if sample_ids is not None:
            batch["sample_id"] = self.torch.tensor(sample_ids, dtype=self.torch.long)
        return batch


def build_data(args, torch, model, tokenizer):
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = read_jsonl(args.train_file, args.train_samples)
    eval_rows = read_jsonl(args.eval_file, args.eval_samples)
    datasets = [EncodedDataset(rows, tokenizer, args.task, args.max_length,
                               args.max_target_length, include_sample_id=train)
                for rows, train in ((train_rows, True), (eval_rows, False))]
    collator = RecordingCollator(torch, tokenizer, args.task, model)
    return datasets[0], datasets[1], collator


def metrics_for(task, tokenizer):
    def compute(prediction):
        values, labels = prediction.predictions, prediction.label_ids
        values = values[0] if isinstance(values, tuple) else values
        if task == "bert":
            return {"accuracy": float(np.mean(np.asarray(values).argmax(-1) == labels))}
        labels = np.where(np.asarray(labels) == -100, tokenizer.pad_token_id, labels)
        texts = tokenizer.batch_decode(values, skip_special_tokens=True)
        targets = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {"exact_match": float(np.mean([a.strip() == b.strip()
                                              for a, b in zip(texts, targets)]))}

    return compute
