"""Shared optimizer trajectories for StepLR and LambdaLR contracts."""
import copy

import torch

from _release_test_helpers import step_scheduler


def optimizer(device):
    parameters = [torch.nn.Parameter(torch.tensor([1., 2.], device=device)),
                  torch.nn.Parameter(torch.tensor([3., 4.], device=device))]
    return torch.optim.SGD([{"params": [parameters[0]], "lr": .2},
                            {"params": [parameters[1]], "lr": .4}], momentum=.9)

def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    return copy.deepcopy(value)

def values(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {key: values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [values(item) for item in value]
    return value

def update(opt, scheduler):
    for group in opt.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.full_like(parameter, .125)
    step_scheduler(opt, scheduler)

def snapshot(opt, scheduler):
    return {"parameters": [values(parameter) for group in opt.param_groups
                            for parameter in group["params"]],
            "optimizer": values(opt.state_dict()), "scheduler": scheduler.state_dict()}
