"""Direct child registry writes must change the live Module, never a snapshot."""
import copy
import gc
import pickle
import weakref

import numpy as np
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestModuleChildStorage:
    def test_native_child_adoption_rebinds_an_existing_view(self, device):
        import jittor as jt
        from jittor.compat.torch.context import get_install_context
        from jittor.compat.torch.frontend import tensor_frontend
        from jittor.compat.torch.nn_adoption import adopt_owned_children
        context = get_install_context(jt)
        owner = context.state["nn_frontend_owner"]
        with tensor_frontend(owner.tensor_type):
            original = context.native_backend.nn.Linear(2, 1)
        old_view = original._modules
        root = torch.nn.Module()
        root.child = original
        adopt_owned_children(owner, root, set())
        assert root.child is not original
        assert root.child._modules is root.child.__dict__["_modules"]
        assert root.child._modules is not old_view
        root.to(device)
        root.child.weight.data.fill_(2)
        root.child.bias.data.zero_()
        value = torch.tensor([[1., 1.]], device=device, requires_grad=True)
        output = root.child(value)
        output.sum().backward()
        assert output.device.type == device
        np.testing.assert_array_equal(output.detach().cpu().numpy(), [[4.]])
        np.testing.assert_array_equal(value.grad.cpu().numpy(), [[2., 2.]])
        assert set(root.state_dict()) == {"child.weight", "child.bias"}

    def test_direct_storage_changes_forward_gradients_and_checkpoint_keys(self, device):
        root = torch.nn.Module()
        storage = root.__dict__["_modules"]
        old = torch.nn.Linear(2, 1, bias=False).to(device)
        new = torch.nn.Linear(2, 1, bias=False).to(device)
        with torch.no_grad():
            old.weight.fill_(2)
            new.weight.fill_(3)
        storage["child"] = old
        assert root.child is old and root._modules is storage
        assert list(root.children()) == [old]
        storage["child"] = new
        value = torch.tensor([[1., 4.]], device=device, requires_grad=True)
        output = root.child(value)
        assert output.device.type == device
        output.sum().backward()
        np.testing.assert_array_equal(output.detach().cpu().numpy(), [[15.]])
        np.testing.assert_array_equal(value.grad.cpu().numpy(), [[3., 3.]])
        np.testing.assert_array_equal(new.weight.grad.cpu().numpy(), [[1., 4.]])
        assert old.weight.grad is None
        assert list(root.parameters()) == [new.weight]
        assert set(root.state_dict()) == {"child.weight"}
        root.extra = old
        assert storage["extra"] is old
        del root.extra
        assert "extra" not in storage
        del storage["child"]
        assert not hasattr(root, "child") and not list(root.parameters())

    def test_none_aliases_and_held_views(self, device):
        root = torch.nn.Module()
        storage = root.__dict__["_modules"]
        child = torch.nn.Linear(2, 1).to(device)
        storage["first"] = child
        storage["alias"] = child
        assert list(root.children()) == [child]
        assert [n for n, _ in root.named_modules()] == ["", "first"]
        assert [n for n, _ in root.named_modules(remove_duplicate=False)] == ["", "first", "alias"]
        assert set(root.state_dict()) == {"first.weight", "first.bias", "alias.weight", "alias.bias"}
        storage["first"] = None
        assert root.first is None and "first" in storage
        assert [n for n, _ in root.named_modules()] == ["", "alias"]
        root.alias = None
        assert storage["alias"] is None and not list(root.children())

    def test_pickle_and_copy_rebind_child_storage(self, device):
        root = torch.nn.Module()
        root.child = torch.nn.Linear(2, 1).to(device)
        root._modules["empty"] = None
        for restored in (copy.deepcopy(root), pickle.loads(pickle.dumps(root))):
            assert restored._modules is restored.__dict__["_modules"]
            assert restored.child is not root.child
            assert restored.empty is None
            replacement = torch.nn.Linear(2, 1).to(device)
            restored.__dict__["_modules"]["child"] = replacement
            assert restored.child is replacement and root.child is not replacement

    def test_registry_does_not_keep_parent_alive(self, device):
        root = torch.nn.Module()
        root.child = torch.nn.Linear(2, 1).to(device)
        storage = root.__dict__["_modules"]
        reference = weakref.ref(root)
        del root
        gc.collect()
        assert reference() is None
        del storage

    def test_detached_registry_copy_and_pickle_remain_usable(self, device):
        root = torch.nn.Module()
        root.child = torch.nn.Linear(2, 1).to(device)
        root._modules["empty"] = None
        for snapshot in (copy.copy(root._modules), copy.deepcopy(root._modules),
                         pickle.loads(pickle.dumps(root._modules)),
                         copy.deepcopy(root.__dict__)["_modules"]):
            assert set(snapshot) == {"child", "empty"}
            assert snapshot["empty"] is None
            output = snapshot["child"](torch.ones((1, 2), device=device))
            assert output.device.type == device and output.shape == (1, 1)
        # An actual reference cycle must not hang a module-name traversal.
        root._modules["cycle"] = root
        assert [n for n, _ in root.named_modules()] == ["", "child"]
        del root._modules["cycle"]


instantiate_device_type_tests(TestModuleChildStorage, globals(), only_for=("cpu", "cuda"))
