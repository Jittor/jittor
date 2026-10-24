"""Negative control: unrelated backward must leave a live idle FSDP model alone."""
import json,os
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

class _IdleBranch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.used=torch.nn.Linear(2,2)
        self.unused=torch.nn.Linear(2,2)
        self.frozen=torch.nn.Parameter(torch.tensor([.25,.5]),requires_grad=False)
    def forward(self,value):return self.used(value)+self.frozen

def _array(value):
    method=getattr(value,"full_tensor",None)
    return (method() if callable(method) else value).detach().float().cpu().numpy().tolist()

def _snapshot(model):
    result={}
    for name,p in model.named_parameters():
        g=p.grad
        result[name]={"id":id(p),"role":getattr(p,"_jittor_fsdp2_role",None),
            "entry_id":id(getattr(p,"_jittor_fsdp2_entry",None)),"gradient_is_none":g is None,
            "gradient_id":None if g is None else id(g),"gradient_dtype":None if g is None else str(g.dtype),
            "gradient_device":None if g is None else str(g.device),
            "complete_gradient":None if g is None else _array(g)}
    return result

@pytest.mark.parametrize("flat",[False,True])
@pytest.mark.parametrize("idle_optimizer",[False,True])
@pytest.mark.parametrize("active_fsdp",[False,True])
def test_idle_fsdp_state_untouched_by_unrelated_backward(flat,idle_optimizer,active_fsdp,monkeypatch,tmp_path):
    monkeypatch.setenv("JITTOR_FSDP2_FLAT","1" if flat else "0")
    idle=_IdleBranch()
    fully_shard(idle,mesh=init_device_mesh("cpu",(1,)))
    optimizer=torch.optim.SGD(idle.parameters(),lr=.01) if idle_optimizer else None
    old=dict(idle.named_parameters())
    for name,p in old.items():
        p.grad=torch.ones_like(p) if name.startswith("used.") else None
    before=_snapshot(idle)
    before_state={"unsharded":idle._fsdp_state.true_fsdp_unsharded,
        "full_parameter_ids":[None if e.full_param is None else id(e.full_param) for e in idle._fsdp_state.true_fsdp_params]}
    active=torch.nn.Linear(2,2)
    if active_fsdp:fully_shard(active,mesh=init_device_mesh("cpu",(1,)))
    leaf=torch.ones((2,2),requires_grad=True)
    output=active(leaf);output.square().mean().backward()
    after=_snapshot(idle)
    after_state={"unsharded":idle._fsdp_state.true_fsdp_unsharded,
        "full_parameter_ids":[None if e.full_param is None else id(e.full_param) for e in idle._fsdp_state.true_fsdp_params]}
    folder=Path(os.environ.get("FSDP_IDLE_GRAD_AUDIT",str(tmp_path)));folder.mkdir(parents=True,exist_ok=True)
    path=folder/("flat%d-opt%d-activefsdp%d.json"%(flat,idle_optimizer,active_fsdp))
    with path.open("x") as f:
        json.dump({"before":before,"after":after,"before_state":before_state,"after_state":after_state,
            "active_complete_output":_array(output),"active_complete_input_gradient":_array(leaf.grad)},f,indent=2,sort_keys=True)
        f.write("\n");f.flush();os.fsync(f.fileno())
    assert before_state==after_state
    assert before==after
    current=dict(idle.named_parameters())
    assert all(current[name] is parameter for name,parameter in old.items())
