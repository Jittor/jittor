"""Adam options must affect arithmetic or fail before updating parameters."""
import numpy as np
import pytest
import torch
from _device_assertions import require_residency


@pytest.mark.parametrize('optimizer_name', ['Adam', 'AdamW'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('group_override', [False, True])
def test_adam_maximize_uses_ascent_and_preserves_grad(optimizer_name, device, group_override):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    parameter = torch.tensor([1., -2.], device=device, requires_grad=True)
    params = [{'params': [parameter], 'maximize': True}] if group_override else [parameter]
    optimizer = getattr(torch.optim, optimizer_name)(params, lr=.1, betas=(0., 0.),
                                                   weight_decay=.2, maximize=not group_override)
    expected = np.array([1., -2.], dtype=np.float32)
    for values in ([1., -1.], [-2., 3.]):
        gradient = np.array(values, dtype=np.float32)
        parameter.grad = torch.tensor(values, device=device)
        update = -gradient
        if optimizer_name == 'Adam':
            update = update + .2 * expected
        else:
            expected = expected * .98
        expected = expected - .1 * update / (np.abs(update) + 1e-8)
        optimizer.step()
        require_residency(parameter, device)
        np.testing.assert_allclose(parameter.detach().cpu().numpy(), expected, atol=2e-6)
        np.testing.assert_array_equal(parameter.grad.cpu().numpy(), gradient)


@pytest.mark.parametrize('optimizer_name', ['Adam', 'AdamW'])
@pytest.mark.parametrize('option', ['amsgrad', 'differentiable', 'capturable'])
@pytest.mark.parametrize('source', ['keyword', 'group', 'after_init'])
def test_unsupported_adam_options_fail_without_update(optimizer_name, option, source):
    parameter = torch.tensor([1.], requires_grad=True)
    optimizer_type = getattr(torch.optim, optimizer_name)
    with pytest.raises(NotImplementedError, match=option):
        if source == 'keyword':
            optimizer_type([parameter], **{option: True})
        elif source == 'group':
            optimizer_type([{'params': [parameter], option: True}])
        else:
            optimizer = optimizer_type([parameter])
            optimizer.param_groups[0][option] = True
            parameter.grad = torch.tensor([1.])
            optimizer.step()
    assert parameter.item() == 1.
