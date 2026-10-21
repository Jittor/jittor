"""Optimizer-style writes preserve physical slices and detached storage aliases."""
import gc
import weakref

import numpy as np
import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestTensorInplaceStorage:
    def test_low_precision_slice_mutations_preserve_aliases(self, device):
        for dtype in (torch.float16, torch.bfloat16):
            for operation, expected in ((lambda x: x.copy_(torch.tensor([6., 8.], device=device)), [6., 8.]),
                                        (lambda x: x.div_(2.0), [1., 2.]),
                                        (lambda x: x.mul_(3.0), [6., 12.]),
                                        (lambda x: x.zero_(), [0., 0.])):
                root = torch.tensor([1., 2., 4., 8.], dtype=dtype, device=device)
                target = root.narrow(0, 1, 2)
                alias = target.detach()
                pointers = [x.data_ptr() for x in (root, target, alias)]
                assert operation(target) is target
                assert pointers == [x.data_ptr() for x in (root, target, alias)]
                assert target.dtype == alias.dtype == dtype
                assert not target.requires_grad and not alias.requires_grad
                assert target.device.type == alias.device.type == device
                np.testing.assert_array_equal(alias.float().cpu().numpy(), expected)
                if hasattr(torch, "_torch_compat_install_context"):
                    target.sync()
                    assert target.location() == ("device" if device == "cuda" else "cpu")

    def test_python_float_division_does_not_create_a_trainable_constant(self, device):
        def exercise(divide):
            for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
                for scalar in (2, 2.0, np.float64(2)):
                    for reflected in (False, True):
                        value = torch.tensor([2., 4.], dtype=dtype, device=device)
                        quotient = divide(value, scalar, reflected)
                        assert not quotient.requires_grad and quotient.dtype == dtype
                        np.testing.assert_array_equal(quotient.float().cpu().numpy(),
                                                      [1., .5] if reflected else [1., 2.])
                        leaf = torch.tensor([2., 4.], dtype=dtype, device=device, requires_grad=True)
                        divide(leaf, scalar, reflected).float().sum().backward()
                        np.testing.assert_array_equal(leaf.grad.float().cpu().numpy(),
                                                      [-.5, -.125] if reflected else [.5, .5])
                special = torch.tensor([0., float("inf"), float("nan")], dtype=dtype, device=device)
                quotient = divide(special, 0.0, False)
                assert not quotient.requires_grad and quotient.dtype == dtype
                actual = quotient.float().cpu().numpy()
                assert np.isnan(actual[[0, 2]]).all() and np.isposinf(actual[1])

        exercise(lambda value, scalar, reflected: scalar / value if reflected else value / scalar)
        if hasattr(torch, "_torch_compat_install_context"):
            from jittor.compat.torch.installers.tensor import method_api
            exercise(lambda value, scalar, reflected: method_api._true_division(
                value, scalar, "__rtruediv__" if reflected else "__truediv__"))

    def test_slice_mutations_keep_aliases_and_pending_snapshots(self, device):
        for operation, expected in (
                (lambda x: x.copy_(torch.tensor([6., 8.], device=device)), [6., 8.]),
                (lambda x: x.div_(2), [1., 2.]),
                (lambda x: x.mul_(3), [6., 12.]),
                (lambda x: x.zero_(), [0., 0.])):
            root = torch.tensor([1., 2., 4., 8.], device=device)
            target = root.narrow(0, 1, 2)
            alias = target.detach()
            snapshot = root.clone()
            pointers = [x.data_ptr() for x in (root, target, alias)]
            assert operation(target) is target
            assert pointers == [x.data_ptr() for x in (root, target, alias)]
            for value in (target, alias):
                np.testing.assert_array_equal(value.cpu().numpy(), expected)
                assert value.device.type == device and not value.requires_grad
            np.testing.assert_array_equal(root.cpu().numpy(), [1., *expected, 8.])
            np.testing.assert_array_equal(snapshot.cpu().numpy(), [1., 2., 4., 8.])

    def test_temporary_base_can_expire_without_losing_storage_alias(self, device):
        for operation, expected in ((lambda x: x.copy_(torch.tensor([9., 10.], device=device)), [9., 10.]),
                                    (lambda x: x.div_(2), [1., 2.]),
                                    (lambda x: x.mul_(3), [6., 12.]),
                                    (lambda x: x.zero_(), [0., 0.])):
            root = torch.tensor([1., 2., 4., 8.], device=device)
            target = root.view(2, 2).narrow(0, 0, 1).view(2)
            # Target [1,2] is rewritten to [2,4] before measuring its aliases.
            target.copy_(torch.tensor([2., 4.], device=device))
            alias = target.detach()
            reference = weakref.ref(root)
            del root
            gc.collect()
            # Torch retains its tensor base; Jittor's provenance is weak.
            if hasattr(torch, "_torch_compat_install_context"):
                assert reference() is None
            pointers = [x.data_ptr() for x in (target, alias)]
            assert operation(target) is target
            assert pointers == [x.data_ptr() for x in (target, alias)]
            np.testing.assert_array_equal(alias.cpu().numpy(), expected)

    def test_parameter_data_and_detached_aliases_see_no_grad_writes(self, device):
        parameter = torch.nn.Parameter(torch.tensor([2., 4.], device=device))
        data = parameter.data
        alias = data.detach()
        pointers = [x.data_ptr() for x in (parameter, data, alias)]
        with torch.no_grad():
            parameter.copy_(torch.tensor([6., 8.], device=device))
            data.div_(2)
            alias.mul_(2)
        assert pointers == [x.data_ptr() for x in (parameter, data, alias)]
        for value in (parameter, data, alias):
            np.testing.assert_array_equal(value.detach().cpu().numpy(), [6., 8.])
        assert parameter.is_leaf and parameter.requires_grad

    def test_copy_broadcast_and_shape_failure_preserve_target(self, device):
        target = torch.tensor([[1., 2.], [3., 4.]], device=device)
        alias = target.detach()
        pointer = target.data_ptr()
        target.copy_(torch.tensor([7., 8.], device=device))
        assert target.data_ptr() == pointer
        np.testing.assert_array_equal(alias.cpu().numpy(), [[7., 8.], [7., 8.]])
        with pytest.raises(RuntimeError):
            target.copy_(torch.tensor([1., 2., 3., 4.], device=device))
        np.testing.assert_array_equal(alias.cpu().numpy(), [[7., 8.], [7., 8.]])

    def test_copy_partial_overlap_rejected_and_self_copy_allowed(self, device):
        root = torch.tensor([1., 2., 3., 4.], device=device)
        with pytest.raises(RuntimeError, match="overlap|single memory"):
            root.narrow(0, 1, 3).copy_(root.narrow(0, 0, 3))
        np.testing.assert_array_equal(root.cpu().numpy(), [1., 2., 3., 4.])
        assert root.copy_(root) is root
        np.testing.assert_array_equal(root.cpu().numpy(), [1., 2., 3., 4.])

    def test_leaf_grad_domain_rejection_and_no_grad_mutation(self, device):
        for operation in (lambda x: x.copy_(torch.tensor([3., 4.], device=device)),
                          lambda x: x.div_(2), lambda x: x.mul_(2), lambda x: x.zero_()):
            leaf = torch.tensor([1., 2.], device=device, requires_grad=True)
            with pytest.raises(RuntimeError, match="leaf Variable"):
                operation(leaf)
            np.testing.assert_array_equal(leaf.detach().cpu().numpy(), [1., 2.])
            pointer = leaf.data_ptr()
            with torch.no_grad():
                operation(leaf)
            assert leaf.requires_grad and leaf.is_leaf and leaf.data_ptr() == pointer

    def test_differentiable_mul_div_copy_retain_rhs_edges(self, device):
        for operation, expected, expected_rhs in (
                (lambda x, rhs: x.mul_(rhs), [6., 6.], [3., 6.]),
                (lambda x, rhs: x.div_(rhs), [1.5, 1.5], [-.75, -1.5]),
                (lambda x, rhs: x.copy_(rhs), [0., 0.], [1., 1.])):
            source = torch.tensor([1., 2.], device=device, requires_grad=True)
            rhs = torch.tensor([2., 2.], device=device, requires_grad=True)
            target = source * 3
            assert operation(target, rhs) is target
            target.sum().backward()
            # copy_ disconnects the destination's old graph (zero source grad).
            actual = np.zeros(2) if source.grad is None else source.grad.cpu().numpy()
            np.testing.assert_array_equal(actual, expected)
            assert rhs.grad is not None
            np.testing.assert_array_equal(rhs.grad.cpu().numpy(), expected_rhs)

    def test_foreach_contiguous_slice_preserves_root_and_alias(self, device):
        root = torch.tensor([1., 2., 3., 4.], device=device)
        target = root.narrow(0, 1, 2)
        alias = target.detach()
        pointers = [x.data_ptr() for x in (root, target, alias)]
        values = [target]
        assert torch._foreach_zero_(values) is values
        assert pointers == [x.data_ptr() for x in (root, target, alias)]
        np.testing.assert_array_equal(root.cpu().numpy(), [1., 0., 0., 4.])
        np.testing.assert_array_equal(alias.cpu().numpy(), [0., 0.])

    def test_differentiable_zero_keeps_zero_gradient_edge(self, device):
        source = torch.tensor([1., 2., float("inf"), float("nan")], device=device, requires_grad=True)
        target = source * 3
        assert target.zero_() is target
        assert target.requires_grad and not target.is_leaf
        np.testing.assert_array_equal(target.detach().cpu().numpy(), [0., 0., 0., 0.])
        target.sum().backward()
        assert source.grad is not None
        np.testing.assert_array_equal(source.grad.cpu().numpy(), [0., 0., 0., 0.])

    def test_transpose_writeback_retains_logical_axis_order(self, device):
        root = torch.tensor([[1., 2.], [3., 4.]], device=device)
        target = root.transpose(0, 1)
        if hasattr(torch, "_torch_compat_install_context"):
            assert not target._storage_view_is_physical()
        target.copy_(torch.tensor([[5., 6.], [7., 8.]], device=device))
        np.testing.assert_array_equal(root.cpu().numpy(), [[5., 7.], [6., 8.]])


instantiate_device_type_tests(TestTensorInplaceStorage, globals())


def test_physical_provenance_distinguishes_pending_and_materialized_transpose():
    root = torch.tensor([[1., 2.], [3., 4.]])
    physical = root.view(4).narrow(0, 1, 2)
    transpose = root.transpose(0, 1)
    assert physical._storage_view_is_physical()
    assert not transpose._storage_view_is_physical()
    physical.sync()
    transpose.sync()
    assert physical._storage_view_is_physical()
    assert not transpose._storage_view_is_physical()
    transpose.copy_(torch.tensor([[5., 6.], [7., 8.]]))
    np.testing.assert_array_equal(root.cpu().numpy(), [[5., 7.], [6., 8.]])


class _EmbeddingSum(torch.nn.Module):
    """BERT's embedding sum: an out-of-place add, then an in-place one."""

    def __init__(self):
        super().__init__()
        self.position = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10)

    def forward(self, words, types):
        embeddings = words + types
        embeddings += self.position
        return embeddings * 2


def test_an_inplace_add_inside_a_captured_module_call():
    """`embeddings += position_embeddings` while the call is being captured.

    The in-place add writes into the tensor's existing storage when it can, so
    that detached aliases see the write; inside a capture the tensor is still
    being built and has no storage yet, and the write raised `_copy_into needs
    an allocated tensor` -- every `bert_base_infer` step of the torch_compat
    benchmark, on its first captured call. Such a tensor has nothing an alias
    could read, so the add rebinds it as it did before the storage path.
    """
    import jittor as jt

    rng = np.random.RandomState(0)
    feeds = [(rng.rand(3, 4).astype("float32"), rng.rand(3, 4).astype("float32")) for _ in range(5)]
    model = _EmbeddingSum().eval()
    position = model.position.detach().cpu().numpy()
    flag = jt.flags.auto_graph_replay
    jt.flags.auto_graph_replay = 1
    try:
        with torch.no_grad():
            got = [model(torch.tensor(w), torch.tensor(t)).cpu().numpy() for w, t in feeds]
        state = model.__dict__.get("_auto_graph_replay")
    finally:
        jt.flags.auto_graph_replay = flag
    for (w, t), out in zip(feeds, got):
        np.testing.assert_allclose(out, (w + t + position) * 2, rtol=1e-6)
    assert state is not None and state.replay is not None, "the call was never captured"
    assert state.replay.refused is None
