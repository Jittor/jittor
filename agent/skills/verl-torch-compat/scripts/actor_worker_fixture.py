"""Controlled actor-worker protocol; no Torch import or hardware work."""

MODEL_SPEC = {
    "model_type": "qwen3", "vocab_size": 32, "hidden_size": 16,
    "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
    "num_key_value_heads": 2, "head_dim": 8, "attention_bias": False,
    "tie_word_embeddings": False,
}
INPUT_IDS = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 5, 9, 10, 11, 12]]
ADVANTAGES = [[0.7, -0.2, 0.3, 1.1], [-0.6, 0.4, -0.9, 0.2]]
STAGES = ["runtime", "import", "distributed_init", "init", "reset", "infer", "train"]
SCRIPT_NAMES = {
    "probe_actor_worker.py", "actor_worker_fixture.py", "probe_training_worker.py",
    "worker_gradient_observer.py", "worker_buffer_contract.py", "worker_loading_provenance.py",
}
SCOPE = "real in-process single-rank actor TrainingWorker; fixed PPO fixture, not rollout/RLHF"


def training_config(steps):
    return {
        "optimizer": "SGD", "lr": 0.01, "weight_decay": 0.0, "momentum": 0.0,
        "clip_grad": 1.0, "dtype": "float32", "steps": steps,
        "wrap_policy": {"disable": True}, "fsdp_size": 1, "use_orig_params": True,
        "batch_size": 2, "sequence_length": 8, "response_length": 4,
        "micro_batch_size": 2, "engine_seed": 42, "checkpoint_seed": 20260924,
        "mask_rule": "all_valid", "input_ids": INPUT_IDS, "advantages": ADVANTAGES,
        "advantage_source": "fixed synthetic fixture, not reward/GAE/GRPO",
        "old_log_probs_source": "initial worker inference, detached and frozen across updates",
        "temperature": 1.0, "loss_mode": "vanilla", "loss_agg_mode": "token-mean",
        "clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.2,
        "clip_ratio_c": 3.0, "entropy_coeff": 0.0, "use_kl_loss": False,
        "rollout_n": 1, "gradient_capture": True, "training_entry": "train_batch",
    }


def parameter_shapes():
    shapes = {"model.embed_tokens.weight": (32, 16), "model.norm.weight": (16,),
              "lm_head.weight": (32, 16)}
    prefix = "model.layers.0."
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        shapes[prefix + "self_attn." + projection + ".weight"] = (16, 16)
    for norm in ("q_norm", "k_norm"):
        shapes[prefix + "self_attn." + norm + ".weight"] = (8,)
    shapes[prefix + "mlp.gate_proj.weight"] = (32, 16)
    shapes[prefix + "mlp.up_proj.weight"] = (32, 16)
    shapes[prefix + "mlp.down_proj.weight"] = (16, 32)
    for norm in ("input_layernorm", "post_attention_layernorm"):
        shapes[prefix + norm + ".weight"] = (16,)
    return shapes
