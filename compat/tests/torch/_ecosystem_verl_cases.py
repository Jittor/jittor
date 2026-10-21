"""Tiny real VERL registry components, independent of rollout/Ray trainers.

Builders use upstream public registry getters and configs unchanged. GAE is
intentionally non-differentiable upstream; gradients belong to the policy
branch. The packed k3 case checks both student and teacher input gradients.
"""


def _response_mask(torch, device):
    return torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]],
        dtype=torch.float32,
        device=device,
    )


def _actor_config():
    from verl.workers.config import ActorConfig

    return ActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        ppo_mini_batch_size=4,
        ppo_micro_batch_size_per_gpu=1,
    )


def ppo_gae_vanilla(torch):
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn, get_policy_loss_fn

    class TinyPPO(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = torch.nn.Linear(3, 1)
            self.config = _actor_config()

        def forward(self, features):
            log_prob = self.policy(features).squeeze(-1) * 0.1
            mask = _response_mask(torch, features.device)
            advantages, returns = get_adv_estimator_fn("gae")(
                features[..., 0].detach(),
                features[..., 1].detach(),
                mask,
                0.97,
                0.93,
            )
            if advantages.requires_grad or returns.requires_grad:
                raise RuntimeError("VERL GAE must preserve its upstream no_grad contract")
            loss, _metrics = get_policy_loss_fn("vanilla")(
                torch.zeros_like(log_prob),
                log_prob,
                advantages,
                mask,
                config=self.config,
            )
            return torch.cat((loss.reshape(1), advantages.reshape(-1), returns.reshape(-1)))

    return TinyPPO(), {"features": ("float32", (4, 6, 3), None)}


def distillation_k3(torch):
    from tensordict import TensorDict
    from verl.trainer.distillation.losses import get_distillation_loss_fn
    from verl.workers.config import DistillationConfig, DistillationLossConfig

    class TinyDistillation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.student = torch.nn.Linear(3, 1)
            self.actor_config = _actor_config()
            self.distillation_config = DistillationConfig(
                enabled=False,
                distillation_loss=DistillationLossConfig(loss_mode="k3"),
            )

        def forward(self, features, teacher):
            mask = _response_mask(torch, features.device)
            # Upstream accepts packed token log-probs plus per-sequence
            # metadata. They have different leading dimensions, so this
            # TensorDict has no common batch dimension (not a trainer batch).
            data = TensorDict(
                {
                    "prompts": torch.ones(4, 1, dtype=torch.int64, device=features.device),
                    "responses": torch.ones(4, 6, dtype=torch.int64, device=features.device),
                    "attention_mask": torch.ones(4, 7, dtype=torch.int64, device=features.device),
                    "response_mask": mask,
                    "teacher_logprobs": -torch.nn.functional.softplus(teacher),
                },
                batch_size=[],
            )
            output = {
                "log_probs": -torch.nn.functional.softplus(self.student(features).squeeze(-1))
            }
            losses, _metrics = get_distillation_loss_fn("k3")(
                self.actor_config,
                self.distillation_config,
                output,
                data,
            )
            return losses * mask

    return TinyDistillation(), {
        "features": ("float32", (28, 3), None),
        "teacher": ("float32", (28, 1), None),
    }
