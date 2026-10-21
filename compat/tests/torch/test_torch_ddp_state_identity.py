"""DDP metadata has stable identity when native optimizers register it."""

import pytest
import jittor as jt
from jittor.compat.torch.tensor_state import compatibility_owner


def _wrapped():
    torch = compatibility_owner(jt)
    raw = torch.nn.Linear(2, 1).to("cpu")
    return torch, raw, torch.nn.parallel.DistributedDataParallel(raw)


@pytest.mark.parametrize("algorithm", ("Adam", "AdamW"))
def test_zero_grad_registers_a_wrapper_created_after_the_optimizer(algorithm):
    torch = compatibility_owner(jt)
    raw = torch.nn.Linear(2, 1).to("cpu")
    optimizer = getattr(torch.optim, algorithm)(raw.parameters(), lr=.01)
    wrapper = torch.nn.parallel.DistributedDataParallel(raw)
    state = wrapper._jittor_ddp_state

    optimizer.zero_grad(set_to_none=True)

    assert list(optimizer._jittor_ddp_states) == [state]
    assert all(parameter._jittor_ddp_state is state
               for parameter in raw.parameters())
    assert state.has_unsynced_grads is False


@pytest.mark.parametrize("algorithm", ("Adam", "AdamW"))
def test_constructor_and_late_groups_keep_equal_field_states_distinct(algorithm):
    torch, first_raw, first = _wrapped()
    _, second_raw, second = _wrapped()
    first_state, second_state = first._jittor_ddp_state, second._jittor_ddp_state
    assert vars(first_state) == vars(second_state)
    assert first_state is not second_state

    optimizer = getattr(torch.optim, algorithm)(first_raw.parameters(), lr=.01)
    optimizer.add_param_group({"params": list(second_raw.parameters())})
    optimizer.zero_grad(set_to_none=True)

    assert first_state != second_state
    assert len(optimizer._jittor_ddp_states) == 2
    assert first_state in optimizer._jittor_ddp_states
    assert second_state in optimizer._jittor_ddp_states


@pytest.mark.parametrize("algorithm", ("Adam", "AdamW"))
def test_no_sync_and_dynamic_fields_preserve_the_registered_state_key(algorithm):
    torch, _, wrapper = _wrapped()
    optimizer = getattr(torch.optim, algorithm)(wrapper.parameters(), lr=.01)
    optimizer.zero_grad(set_to_none=True)
    state = wrapper._jittor_ddp_state
    state_hash = hash(state)

    with wrapper.no_sync():
        assert wrapper._jittor_ddp_state is state
        assert state.sync_enabled is False
        assert wrapper.require_backward_grad_sync is False
        state.annotation = {"source": "identity-regression"}
        assert state in optimizer._jittor_ddp_states
        assert hash(state) == state_hash

    assert wrapper._jittor_ddp_state is state
    assert state.sync_enabled is True
    assert wrapper.require_backward_grad_sync is True
    state.has_unsynced_grads = True
    optimizer.zero_grad(set_to_none=True)
    assert state.has_unsynced_grads is False
    assert list(optimizer._jittor_ddp_states) == [state]
    assert hash(state) == state_hash
