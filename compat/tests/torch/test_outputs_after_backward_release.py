"""Outputs stay non-leaf after a non-retained backward releases their graph.

``running_loss += loss`` after ``loss.backward()`` -- the training loop of
most PyTorch tutorials -- raised "a leaf
Variable that requires grad is being used in an in-place operation": once the
backward freed the producer, nothing remembered that the loss had one. Torch
keeps grad_fn and leaves detach()/no_grad outputs as leaves. Expectations from
PyTorch 2.9.0. (A second backward through the released graph still follows
Jittor's missing-gradient policy rather than raising like torch.)
"""
import pytest
import torch


def test_training_loop_accumulates_released_losses():
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)
    running = 0
    for _ in range(3):
        opt.zero_grad()
        loss = (w * 3).sum()
        loss.backward()
        assert loss.is_leaf is False and loss.grad_fn is not None and loss.requires_grad
        opt.step()
        running += loss
    assert running.is_leaf is False
    assert float(running.detach()) == pytest.approx(21.6, rel=1e-6)


def test_retained_graph_and_leaves_are_unchanged():
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    x = w * 2
    x.sum().backward(retain_graph=True)
    x.sum().backward()
    assert w.grad.tolist() == [4.0, 4.0]
    detached = (w * 2).detach()
    assert detached.is_leaf and not detached.requires_grad
    with torch.no_grad():
        constant = w * 2
    assert constant.is_leaf and not constant.requires_grad
