"""Fixed two-rank worker protocol; no runtime imports or hardware work."""
from actor_worker_fixture import parameter_shapes as actor_shapes

MODEL_SPEC = {
    "model_type": "qwen3", "vocab_size": 32, "hidden_size": 16,
    "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
    "num_key_value_heads": 2, "head_dim": 8, "attention_bias": False,
    "tie_word_embeddings": True,
}
INPUTS = [
    [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 5, 9, 10, 11, 12]],
    [[1, 13, 14, 15, 16, 17, 18, 19], [1, 13, 14, 20, 21, 22, 23, 24]],
]
TARGETS = {
    "actor": [[[0.7, -0.2, 0.3, 1.1], [-0.6, 0.4, -0.9, 0.2]],
              [[-0.3, 0.8, -0.4, 0.2], [0.9, -0.5, 0.3, -0.7]]],
    "critic": [[[0.4, 0.3, 0.2, 0.1], [-0.1, 0.1, 0.3, 0.5]],
               [[-0.2, 0.5, 0.1, 0.7], [0.6, -0.3, 0.2, -0.1]]],
}
STAGES = ["runtime", "import", "distributed_init", "init", "reset", "infer", "train"]
SCOPE = "real two-rank in-process verl FSDP2 TrainingWorker; fixed fixture, not rollout/RLHF"
WORKFLOW_SCOPE = (
    "real two-rank in-process verl FSDP2 TrainingWorker with verl reward/advantage targets; "
    "fixed responses, not Ray rollout or coupled actor-critic training"
)
SCRIPT_NAMES = {
    "probe_fsdp2_worker.py", "compare_fsdp2_worker.py", "fsdp2_worker_fixture.py",
    "actor_worker_fixture.py", "probe_training_worker.py", "worker_gradient_observer.py",
    "worker_buffer_contract.py", "worker_loading_provenance.py",
    "worker_npu_provenance.py",
}


def parameter_shapes(role):
    shapes = actor_shapes()
    del shapes["lm_head.weight"]
    if role == "critic":
        shapes.update({"score.weight": (1, 16), "score.bias": (1,)})
    elif role != "actor":
        raise ValueError("unknown worker role")
    return shapes


def aliases(role):
    result = {name: [name] for name in parameter_shapes(role)}
    if role == "actor":
        result["model.embed_tokens.weight"].append("lm_head.weight")
    return result


def scope(algorithm="fixed"):
    return SCOPE if algorithm == "fixed" else WORKFLOW_SCOPE


def protocol(role, algorithm="fixed"):
    result = {
        "role": role, "strategy": "fsdp2", "fsdp_size": 2, "mesh": [2],
        "wrap_policy": {"transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"]},
        "dtype": "float32", "offload": False, "forward_prefetch": False,
        "requested_attention": "eager",
        # verl 0.9.1 load_valuehead_model explicitly chooses flash_attention_2
        # after HFModelConfig initialization; the actor respects the request.
        "expected_loaded_attention": "eager" if role == "actor" else "flash_attention_2",
        "reshard_after_forward": True, "sp_size": 1, "steps": 3,
        "per_rank_batch": 2, "global_batch_size": 4, "ppo_mini_batch_size": 4,
        "micro_batch_size": 2, "sequence_length": 8, "response_length": 4,
        "optimizer": "ObservedSGD", "lr": 0.01, "momentum": 0.0,
        "weight_decay": 0.0, "clip_grad": 1.0, "engine_seed": 42,
        "checkpoint_seed": 20260924 if role == "actor" else 20260923,
        "input_ids": INPUTS, "targets": TARGETS[role], "temperature": 1.0,
        "loss_agg_mode": "token-mean", "clip_ratio": 0.2,
        "clip_ratio_low": 0.2, "clip_ratio_high": 0.2, "clip_ratio_c": 3.0,
        "cliprange_value": 0.5,
        "use_kl_loss": False, "entropy_coeff": 0.0,
        "anchor": "initial worker inference, detached and frozen",
        "target_source": "fixed synthetic advantages" if role == "actor" else "fixed synthetic returns",
    }
    if algorithm != "fixed":
        result.update(
            algorithm=algorithm,
            target_source=("verl-computed advantages" if role == "actor" else "verl-computed returns"),
            reward_source={
                "ppo": "deterministic token rule",
                "grpo": "deterministic token rule with per-rank prompt pair",
                "rlhf": "deterministic scalar reward plus explicit reference-logprob fixture",
                "rlvr": "deterministic inverse-token-distance verifier",
            }[algorithm],
            target_limitations=[
                "fixed responses rather than rollout-engine generation",
                "actor and critic role launches are not coupled",
                "RLHF distributed fixture does not instantiate a reward or reference model",
            ],
        )
    return result
