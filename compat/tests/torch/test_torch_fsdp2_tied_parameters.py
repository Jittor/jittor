"""One FSDP state preserves every registration of a tied parameter."""

import numpy as np
import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, set_model_state_dict,
)


class _TiedEmbeddingHead(torch.nn.Module):
    def __init__(self, trainable, root_alias):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 3)
        self.head = torch.nn.Linear(3, 4, bias=False)
        self.head.weight = self.embedding.weight
        self.embedding.weight.requires_grad_(trainable)
        self.adapter = torch.nn.Linear(3, 3, bias=False)
        if root_alias:
            self.shared_weight = self.embedding.weight
        self.observed_ties = []

    def forward(self, indices):
        self.observed_ties.append(self.embedding.weight is self.head.weight)
        return self.head(self.adapter(self.embedding(indices)))


def _assert_tied(model, root_alias):
    assert model.embedding.weight is model.head.weight
    if root_alias:
        assert model.shared_weight is model.embedding.weight
    assert len(list(model.named_parameters())) == 2
    aliases = dict(model.named_parameters(remove_duplicate=False))
    assert aliases["embedding.weight"] is aliases["head.weight"]
    entry = model.embedding.weight._jittor_fsdp2_entry
    assert sum(item is entry for item in model._fsdp_state.true_fsdp_params) == 1
    assert len(model._fsdp_state.true_fsdp_params) == 2


def _array(tensor):
    full = getattr(tensor, "full_tensor", None)
    return (full() if callable(full) else tensor).detach().cpu().numpy()


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("root_alias", [False, True])
def test_same_state_tied_parameter_lifecycle(flat, trainable, root_alias, monkeypatch):
    monkeypatch.setenv("JITTOR_FSDP2_FLAT", "1" if flat else "0")
    torch.manual_seed(719)
    model = _TiedEmbeddingHead(trainable, root_alias)
    reference = _TiedEmbeddingHead(trainable, root_alias)
    reference.load_state_dict(model.state_dict())
    # The optimizer may exist before fully_shard replaces the public holders.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
    assert model._fsdp_state.true_fsdp_initialized
    _assert_tied(model, root_alias)
    assert len(optimizer.param_groups[0]["params"]) == 2
    indices = torch.tensor([0, 1, 1, 3], dtype=torch.int64)
    output, expected = model(indices), reference(indices)
    assert model.observed_ties == [True]
    _assert_tied(model, root_alias)
    np.testing.assert_array_equal(_array(output), _array(expected))
    output.square().sum().backward()
    expected.square().sum().backward()
    _assert_tied(model, root_alias)
    for actual, dense in ((model.embedding.weight, reference.embedding.weight),
                          (model.adapter.weight, reference.adapter.weight)):
        if dense.requires_grad:
            assert actual.grad is not None and dense.grad is not None
            np.testing.assert_allclose(_array(actual.grad), _array(dense.grad), rtol=1e-6, atol=1e-7)
        else:
            assert actual.grad is None
    optimizer.step()
    reference_optimizer.step()
    _assert_tied(model, root_alias)
    np.testing.assert_allclose(_array(model.embedding.weight), _array(reference.embedding.weight), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(_array(model.adapter.weight), _array(reference.adapter.weight), rtol=1e-6, atol=1e-7)
    model.unshard()
    _assert_tied(model, root_alias)
    model.reshard()
    _assert_tied(model, root_alias)
    # Both public checkpoint transformations must preserve the registration tie,
    # including flat loads that construct new shard views.
    for full in (True, False):
        options = StateDictOptions(full_state_dict=full)
        state = get_model_state_dict(model, options=options)
        target = _TiedEmbeddingHead(trainable, root_alias)
        fully_shard(target, mesh=init_device_mesh("cpu", (1,)))
        set_model_state_dict(target, state, options=options)
        _assert_tied(target, root_alias)
        np.testing.assert_array_equal(_array(target.embedding.weight), _array(model.embedding.weight))
        np.testing.assert_array_equal(_array(target.adapter.weight), _array(model.adapter.weight))


@pytest.mark.parametrize("kind", ("list", "dict"))
def test_alias_in_parameter_container_tracks_canonical_holder_replacement(kind):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(3))
    key = 0 if kind == "list" else "tied"
    model.aliases = (torch.nn.ParameterList([model.weight]) if kind == "list"
                     else torch.nn.ParameterDict({"tied": model.weight}))
    fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
    for action in (model.unshard, model.reshard, model.unshard, model.reshard):
        action()
        assert model.weight is model.aliases[key]
        np.testing.assert_array_equal(_array(model.weight), np.ones(3))
    assert model._fsdp_state.true_fsdp_initialized
