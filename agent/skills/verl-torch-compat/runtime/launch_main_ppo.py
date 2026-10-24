#!/usr/bin/env python3
"""Launch verl PPO after selecting the owned Torch API version policy."""
import torch


reporter = getattr(torch, "compat_report_torch_api_version", None)
if callable(reporter):
    reported = reporter(True)
    if reported != getattr(torch, "__torch_version__", None):
        raise RuntimeError("Jittor Torch API version bootstrap reported an inconsistent value")

from verl.trainer.main_ppo import main


if __name__ == "__main__":
    main()
