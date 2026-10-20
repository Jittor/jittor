"""Pin the public tokenizer, collator and device handoff used by HF training."""

import os

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast


_DEVICE = os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0]


def _tokenizer():
    raw = Tokenizer(models.WordLevel(
        {"[UNK]": 0, "[PAD]": 1, "alpha": 2, "beta": 3, "gamma": 4},
        unk_token="[UNK]",
    ))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )


def test_fast_tokenizer_collator_and_batchencoding_device_handoff():
    tokenizer = _tokenizer()
    encoded = tokenizer(["alpha beta", "beta"], padding=True, return_tensors="pt")
    assert encoded["input_ids"].tolist() == [[2, 3], [3, 1]]
    assert encoded["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert encoded["input_ids"].dtype == torch.int64

    encoded = encoded.to(_DEVICE)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    batch = collator([
        {"input_ids": [2, 3], "attention_mask": [1, 1], "labels": [3, 2]},
        {"input_ids": [3], "attention_mask": [1], "labels": [4]},
    ])
    batch = {key: value.to(_DEVICE) for key, value in batch.items()}
    assert batch["input_ids"].tolist() == [[2, 3], [3, 1]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["labels"].tolist() == [[3, 2], [4, -100]]

    weights = np.arange(20, dtype=np.float32).reshape(5, 4) / 8
    embedding = torch.nn.Embedding(5, 4).to(_DEVICE)
    with torch.no_grad():
        embedding.weight.copy_(torch.tensor(weights, device=_DEVICE))
    output = embedding(batch["input_ids"])
    expected = weights[np.asarray([[2, 3], [3, 1]])]
    np.testing.assert_array_equal(output.detach().cpu().numpy(), expected)

    assert encoded["input_ids"].device.type == _DEVICE
    assert all(value.device.type == _DEVICE for value in batch.values())
    assert embedding.weight.device.type == output.device.type == _DEVICE
    if hasattr(torch, "_torch_compat_install_context"):
        import jittor as jt

        expected_index = int(torch.cuda.current_device()) if _DEVICE == "cuda" else -1
        for value in [*encoded.values(), *batch.values(), embedding.weight, output]:
            backend, index = jt.core.dispatch_context([value])
            assert backend == _DEVICE
            if _DEVICE == "cuda":
                assert index == expected_index
        assert jt.core.backend_fallback_count() == 0
    elif _DEVICE == "cuda":
        torch.cuda.synchronize()
