import os
import tempfile
import unittest

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_REQUIRE_OPTIONAL_DEPS = os.environ.get("JITTOR_REQUIRE_OPTIONAL_DEPS") == "1"

try:
    import torch
    import jittor as jt
    from peft import (
        AdaLoraConfig,
        BOFTConfig,
        BoneConfig,
        C3AConfig,
        CPTConfig,
        FourierFTConfig,
        HRAConfig,
        IA3Config,
        LNTuningConfig,
        LoHaConfig,
        LoKrConfig,
        LoraConfig,
        MissConfig,
        MultitaskPromptTuningConfig,
        OFTConfig,
        PeftModel,
        PolyConfig,
        PrefixTuningConfig,
        PromptEncoderConfig,
        PromptTuningConfig,
        RandLoraConfig,
        ShiraConfig,
        TaskType,
        TrainableTokensConfig,
        VBLoRAConfig,
        VeraConfig,
        XLoraConfig,
        get_peft_model,
    )
    from transformers import Qwen2Config, Qwen2ForCausalLM

    _HAS_DEPS = True
except Exception:
    if _REQUIRE_OPTIONAL_DEPS:
        raise
    _HAS_DEPS = False


def _qwen2_config():
    return Qwen2Config(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )


def _method_configs():
    lora = {
        "task_type": TaskType.CAUSAL_LM,
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
    }
    return {
        "lora": LoraConfig(**lora),
        "rslora": LoraConfig(use_rslora=True, **lora),
        "dora": LoraConfig(use_dora=True, **lora),
        "adalora": AdaLoraConfig(init_r=2, target_r=1, total_step=2, **lora),
        "ia3": IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
        ),
        "prompt_tuning": PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM, num_virtual_tokens=3
        ),
        "p_tuning": PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=3,
            encoder_hidden_size=8,
        ),
        "prefix_tuning": PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM, num_virtual_tokens=3
        ),
        "multitask_prompt_tuning": MultitaskPromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=3,
            num_tasks=2,
            num_ranks=1,
        ),
        "cpt": CPTConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=3,
                         cpt_token_ids=[1, 2, 3], cpt_mask=[1, 1, 1],
                         cpt_tokens_type_mask=[1, 1, 1]),
        "loha": LoHaConfig(r=2, alpha=2, **{k: v for k, v in lora.items() if k not in {"r", "lora_alpha", "lora_dropout"}}),
        "lokr": LoKrConfig(r=2, alpha=2, **{k: v for k, v in lora.items() if k not in {"r", "lora_alpha", "lora_dropout"}}),
        "oft": OFTConfig(r=2, oft_block_size=0, task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]),
        "boft": BOFTConfig(boft_block_size=2, task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]),
        "poly": PolyConfig(r=2, n_tasks=2, n_skills=2, task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]),
        "ln_tuning": LNTuningConfig(task_type=TaskType.CAUSAL_LM, target_modules=["input_layernorm", "post_attention_layernorm"]),
        "vera": VeraConfig(r=2, save_projection=False, task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]),
        "fourierft": FourierFTConfig(n_frequency=4, task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]),
        "hra": HRAConfig(r=2, task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]),
        "vblora": VBLoRAConfig(r=2, num_vectors=8, vector_length=2, topk=2, task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]),
        "bone": BoneConfig(r=2, task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]),
        "miss": MissConfig(r=2, mini_r=1, task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]),
        "randlora": RandLoraConfig(r=2, randlora_alpha=4, task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]),
        "trainable_tokens": TrainableTokensConfig(task_type=TaskType.CAUSAL_LM, token_indices=[1, 2]),
        "shira": ShiraConfig(r=2, task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]),
        "c3a": C3AConfig(block_size=2, task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]),
    }


@unittest.skipUnless(_HAS_DEPS, "needs torch shim, transformers, and peft")
class TestPeftQwenMethods(unittest.TestCase):
    def setUp(self):
        self._previous_use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 0

    def tearDown(self):
        jt.flags.use_cuda = self._previous_use_cuda

    def test_qwen2_priority_methods_train_and_reload(self):
        ids = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]], dtype=torch.long)
        for name, peft_config in _method_configs().items():
            with self.subTest(method=name):
                torch.manual_seed(1729)
                base = Qwen2ForCausalLM(_qwen2_config())
                base_state = {
                    key: value.detach().clone() for key, value in base.state_dict().items()
                }
                model = get_peft_model(base, peft_config)
                trainable = {
                    key: value
                    for key, value in model.named_parameters()
                    if value.requires_grad
                }
                frozen = {
                    key: value
                    for key, value in model.named_parameters()
                    if not value.requires_grad
                }
                self.assertTrue(trainable)
                with torch.no_grad():
                    for parameter in trainable.values():
                        parameter.add_(0.03125)

                extra = {}
                if name in {"multitask_prompt_tuning", "poly"}:
                    extra["task_ids"] = torch.tensor([0, 1], dtype=torch.long)
                if name == "cpt":
                    extra["input_type_mask"] = torch.full_like(ids, 3)
                output = model(input_ids=ids, labels=ids, **extra)
                self.assertTrue(bool(torch.isfinite(output.loss)))
                output.loss.backward()
                missing = [
                    key
                    for key, parameter in trainable.items()
                    if parameter.grad is None
                ]
                expected_missing = (
                    ["prompt_encoder.default.embedding.weight"] if name == "cpt" else []
                )
                self.assertEqual(missing, expected_missing)
                self.assertTrue(
                    all(
                        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                        for parameter in trainable.values()
                    )
                )

                frozen_before = {
                    key: parameter.detach().clone()
                    for key, parameter in frozen.items()
                }
                optimizer = torch.optim.SGD(trainable.values(), lr=0.01)
                optimizer.step()
                changed_frozen = [
                    key
                    for key, parameter in frozen.items()
                    if not bool(torch.equal(parameter.detach(), frozen_before[key]))
                ]
                self.assertEqual(changed_frozen, [])

                model.eval()
                with torch.no_grad():
                    expected = model(input_ids=ids, **extra).logits.detach().clone()
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory, safe_serialization=True)
                    fresh = Qwen2ForCausalLM(_qwen2_config())
                    fresh.load_state_dict(base_state, strict=True)
                    loaded = PeftModel.from_pretrained(fresh, directory).eval()
                    with torch.no_grad():
                        actual = loaded(input_ids=ids, **extra).logits
                self.assertLess(float((expected - actual).abs().max()), 1e-5)

    def test_qwen2_xlora_uses_immutable_base_fixture(self):
        ids = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as directory:
            adapters = {}
            for index in range(2):
                torch.manual_seed(1729)
                expert = get_peft_model(
                    Qwen2ForCausalLM(_qwen2_config()),
                    LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=2,
                        lora_alpha=4,
                        target_modules=["q_proj", "v_proj"],
                    ),
                )
                with torch.no_grad():
                    for parameter in expert.parameters():
                        if parameter.requires_grad:
                            parameter.add_(0.03125 * (index + 1))
                path = os.path.join(directory, f"expert_{index}")
                expert.save_pretrained(path)
                adapters[str(index)] = path

            torch.manual_seed(1729)
            base = Qwen2ForCausalLM(_qwen2_config())
            base.config.use_cache = False
            base_path = os.path.join(directory, "base.npz")
            np.savez(base_path, **{
                key: value.detach().cpu().numpy().copy()
                for key, value in base.state_dict().items()
            })
            model = get_peft_model(
                base,
                XLoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    hidden_size=16,
                    adapters=adapters,
                    xlora_size=16,
                    xlora_depth=1,
                    xlora_dropout_p=0.0,
                ),
            )
            trainable = {
                key: value for key, value in model.named_parameters() if value.requires_grad
            }
            self.assertEqual(len(trainable), 2)
            model.eval()
            model.base_model.set_scaling_pass_value(0.0)
            zero = model(input_ids=ids).logits.detach()
            model.base_model.set_scaling_pass_value(1.0)
            unit = model(input_ids=ids).logits.detach()
            self.assertGreater(float((zero - unit).abs().max()), 0.0)
            model.base_model.set_scaling_pass_value(None)
            model.train()
            output = model(input_ids=ids, labels=ids)
            output.loss.backward()
            self.assertTrue(all(parameter.grad is not None for parameter in trainable.values()))
            torch.optim.SGD(trainable.values(), lr=0.01).step()
            model.eval()
            model.base_model.set_scaling_pass_value(None)
            with torch.no_grad():
                expected = model(input_ids=ids).logits
            saved = os.path.join(directory, "xlora")
            model.save_pretrained(saved, safe_serialization=True)

            fresh = Qwen2ForCausalLM(_qwen2_config())
            fresh.config.use_cache = False
            arrays = np.load(base_path)
            fresh.load_state_dict({
                key: torch.tensor(arrays[key], dtype=value.dtype)
                for key, value in fresh.state_dict().items()
            }, strict=True)
            loaded = PeftModel.from_pretrained(fresh, saved).eval()
            loaded.base_model.set_scaling_pass_value(None)
            with torch.no_grad():
                actual = loaded(input_ids=ids).logits
            self.assertLess(float((expected - actual).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
