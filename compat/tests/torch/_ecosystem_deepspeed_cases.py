"""Runtime-independent models/configs for the dedicated real Engine gate."""


def model(torch):
    class Residual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up = torch.nn.Linear(8, 16)
            self.down = torch.nn.Linear(16, 8)
            self.norm = torch.nn.LayerNorm(8)
            self.head = torch.nn.Linear(8, 4)
            with torch.no_grad():
                for index, parameter in enumerate(self.parameters()):
                    values = torch.arange(parameter.numel(), dtype=torch.float32)
                    parameter.copy_(values.reshape(parameter.shape) * .001 + .01 * (index + 1))

        def forward(self, inputs):
            hidden = self.down(torch.nn.functional.gelu(self.up(inputs)))
            return self.head(self.norm(inputs + hidden))
    return Residual()


def config(optimizer="AdamW", zero_stage=0):
    return {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "train_batch_size": 4,
        "zero_optimization": {"stage": zero_stage},
        "optimizer": {"type": optimizer, "params": {
            "torch_adam": True, "adam_w_mode": optimizer == "AdamW",
            "lr": .003, "betas": [.8, .93], "eps": 1e-5,
            "weight_decay": .02,
        }},
        "gradient_clipping": 1.0,
        "steps_per_print": 10000,
    }
