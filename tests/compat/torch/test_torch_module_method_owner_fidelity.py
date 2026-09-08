"""``nn.Module`` methods are module-level owners with recorded fidelity.

``_install_module_methods`` used to define all forty of its Module methods as
closures inside the installer, so none of them could be imported or exercised
without running a full install first, and none could carry fidelity metadata.
This file pins the promotion: every method the installer binds is a module-level
function in ``jittor.compat.torch.installers.nn``, the installer itself contains
no nested ``def``/``class``/``lambda`` at all, and the behaviour of each method
matches torch (probed against torch 2.12.1) rather than merely "not crashing".
"""

import ast
import contextlib
import inspect
import textwrap

import numpy as np
import pytest

import jittor as jt
import torch
import torch.nn as nn

from jittor.compat.torch import fidelity as fidelity_mod
from jittor.compat.torch.installers.nn import module_methods as nn_installer

from _helpers.device_types import instantiate_device_type_tests


# --------------------------------------------------------------------------
# structure: the installer binds, it does not define
# --------------------------------------------------------------------------

def test_install_module_methods_defines_nothing():
    """The installer body holds zero nested def/class/lambda."""
    source = textwrap.dedent(inspect.getsource(nn_installer._install_module_methods))
    tree = ast.parse(source)
    installer = tree.body[0]
    nested = [
        node for node in ast.walk(installer)
        if node is not installer
        and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef, ast.Lambda))
    ]
    assert nested == [], (
        "_install_module_methods must only bind module-level objects; found "
        + ", ".join(getattr(n, "name", "<lambda>") for n in nested))


# Every (Module attribute, owning module-level function) the installer binds
# unconditionally. Guarded bindings are checked separately because jittor may
# already own the native attribute.
_UNCONDITIONAL_OWNERS = (
    ("execute", "_execute"),
    ("_dispatch_call", "_call"),
    ("named_parameters", "_named_parameters"),
    ("named_buffers", "_named_buffers"),
    ("named_modules", "_named_modules"),
    ("load_state_dict", "_load_state_dict"),
    ("parameters", "_parameters"),
    ("train", "_train"),
    ("eval", "_eval"),
    ("to", "_module_to"),
    ("to_empty", "_module_to_empty"),
    ("cuda", "_module_cuda"),
    ("npu", "_module_npu"),
    ("cpu", "_module_cpu"),
    ("zero_grad", "_zero_grad"),
)


@pytest.mark.parametrize("attr,owner_name", _UNCONDITIONAL_OWNERS)
def test_module_method_is_the_module_level_object(attr, owner_name):
    """``nn.Module.foo`` is the module-level function, not a fresh closure."""
    owner = getattr(nn_installer, owner_name)
    bound = nn.Module.__dict__.get(attr)
    assert bound is owner, (
        "nn.Module.%s is %r, expected the module-level %s"
        % (attr, bound, owner_name))


@pytest.mark.parametrize("attr,owner_name", _UNCONDITIONAL_OWNERS)
def test_module_method_identity_is_stable(attr, owner_name):
    """Repeated attribute access yields one object (no per-access wrapper)."""
    assert nn.Module.__dict__.get(attr) is nn.Module.__dict__.get(attr)


@pytest.mark.parametrize("attr,owner_name", _UNCONDITIONAL_OWNERS)
def test_module_method_is_documented(attr, owner_name):
    """A promoted owner is importable *and* says what it does."""
    owner = getattr(nn_installer, owner_name)
    assert inspect.isfunction(owner)
    assert (owner.__doc__ or "").strip(), "%s needs a docstring" % owner_name


def test_reinstall_keeps_identity():
    """A second install rebinds the same objects (no closure churn)."""
    before = {attr: nn.Module.__dict__.get(attr)
              for attr, _ in _UNCONDITIONAL_OWNERS}
    nn_installer._install_module_methods(nn)
    for attr, expected in before.items():
        assert nn.Module.__dict__.get(attr) is expected, attr


# The seven natives the wrappers delegate to. Promotion moved their capture from
# install time to import time, which is only safe while no earlier installer has
# already replaced them: capturing a wrapper here would make the wrapper delegate
# to itself. That recurses on the very first call, so it is loud rather than
# silent -- but it would be loud at import of a user's model, not here.
_CAPTURED_NATIVES = (
    ("_ORIG_MODULE_EXECUTE", "execute"),
    ("_ORIG_MODULE_DISPATCH_CALL", "_dispatch_call"),
    ("_ORIG_MODULE_NAMED_PARAMETERS", "named_parameters"),
    ("_ORIG_MODULE_NAMED_BUFFERS", "named_buffers"),
    ("_ORIG_MODULE_NAMED_MODULES", "named_modules"),
    ("_ORIG_MODULE_LOAD_STATE_DICT", "load_state_dict"),
    ("_ORIG_MODULE_PARAMETERS", "parameters"),
)


@pytest.mark.parametrize("handle_name,attr", _CAPTURED_NATIVES)
def test_the_captured_module_methods_are_still_native(handle_name, attr):
    """An ``_ORIG_MODULE_*`` handle holds Jittor's method, not a compat wrapper.

    Pins the assumption the import-time capture rests on. Without this, the
    capture order is merely asserted in a comment: the file's own promoted
    wrappers are the objects that would be captured if some earlier installer
    had already patched ``nn.Module``, and a wrapper delegating to itself
    recurses instead of reaching Jittor.
    """
    captured = getattr(nn_installer, handle_name)
    promoted = {
        id(value) for name, value in vars(nn_installer).items()
        if name.startswith("_") and callable(value) and inspect.isfunction(value)
        and value.__module__ == nn_installer.__name__
    }
    assert id(captured) not in promoted, (
        "%s captured this file's own %s instead of Jittor's native %s"
        % (handle_name, getattr(captured, "__name__", captured), attr))
    assert not getattr(captured, "__module__", "").startswith(
        "jittor.compat.torch"), (
        "%s must hold a native Jittor method, got one owned by %s"
        % (handle_name, captured.__module__))


# --------------------------------------------------------------------------
# fidelity metadata
# --------------------------------------------------------------------------

_FIDELITY_APIS = (
    "torch.nn.Module.execute",
    "torch.nn.Module.named_parameters",
    "torch.nn.Module.named_buffers",
    "torch.nn.Module.named_modules",
    "torch.nn.Module.load_state_dict",
    "torch.nn.Module.parameters",
    "torch.nn.Module.train",
    "torch.nn.Module.eval",
    "torch.nn.Module.to",
    "torch.nn.Module.to_empty",
    "torch.nn.Module.cuda",
    "torch.nn.Module.zero_grad",
)


@pytest.mark.parametrize("api", _FIDELITY_APIS)
def test_fidelity_recorded(api):
    """Each promoted Module method carries a queryable fidelity record."""
    record = fidelity_mod.fidelity_of(api)
    assert record.level in (fidelity_mod.Fidelity.EXACT,
                            fidelity_mod.Fidelity.APPROXIMATE)
    assert len(record.detail.strip()) > 20, "detail must state the limitation"


def test_fidelity_points_at_the_bound_object():
    """The record's implementation is what ``nn.Module`` actually uses."""
    assert (fidelity_mod.fidelity_of("torch.nn.Module.zero_grad").implementation
            is nn.Module.__dict__["zero_grad"])
    assert (fidelity_mod.fidelity_of("torch.nn.Module.to").implementation
            is nn.Module.__dict__["to"])


def test_module_methods_appear_in_the_report():
    """The coverage report can enumerate this cohort by prefix."""
    reported = {r.api for r in fidelity_mod.fidelity_report("torch.nn.Module.")}
    assert set(_FIDELITY_APIS) <= reported


# --------------------------------------------------------------------------
# behaviour: probed against torch 2.12.1
# --------------------------------------------------------------------------

@contextlib.contextmanager
def unbridged_grad():
    """Run a backward-dependent block on the unbridged autograd path.

    Two pieces of process-global state decide whether ``backward()`` populates
    ``.grad`` at all, and neither belongs to this file:

    ``jt.flags.no_grad`` -- an earlier file can fail *inside* a ``no_grad`` block
    and leave grad disabled for the rest of the process.

    ``jt._active_optimizers`` -- while *any* optimizer object is alive anywhere in
    the process, the backward path routes gradients into it and ``p.grad`` stays
    None; dropping the last reference to that optimizer brings ``.grad`` back.
    Real torch 2.12.1 populates ``.grad`` regardless of whether an optimizer
    exists, so this is a genuine divergence, but it lives in the
    optimizer/autograd bridge rather than in ``_install_module_methods``, so it
    is recorded for that owner instead of being patched from here. Neutralizing
    it locally is what makes a zero_grad assertion fail only when zero_grad is
    actually wrong.
    """
    previous_no_grad = bool(jt.flags.no_grad)
    previous_current = getattr(jt, "_current_optimizer", None)
    previous_active = list(getattr(jt, "_active_optimizers", []) or [])
    jt.flags.no_grad = 0
    jt._current_optimizer = None
    if hasattr(jt, "_active_optimizers"):
        jt._active_optimizers[:] = []
    try:
        yield
    finally:
        jt.flags.no_grad = 1 if previous_no_grad else 0
        jt._current_optimizer = previous_current
        if hasattr(jt, "_active_optimizers"):
            jt._active_optimizers[:] = previous_active


@pytest.fixture
def grad_enabled():
    """Fixture form of :func:`unbridged_grad` for the plain pytest functions.

    The device-parameterized class below cannot take fixtures -- the device-type
    templates are unittest classes -- so it uses the context manager directly.
    """
    with unbridged_grad():
        yield


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.register_buffer("keep", torch.ones(3))
        self.register_buffer("drop", torch.zeros(2), persistent=False)

    def forward(self, x):
        return self.lin(x)


def test_zero_grad_set_to_none_false_leaves_zero_tensors(grad_enabled):
    """torch's ``zero_grad(set_to_none=False)`` zeroes grads, it does not drop them.

    Real torch 2.12.1 leaves ``p.grad`` as an all-zero tensor here. Returning
    None instead is silently wrong: gradient clipping and accumulation code is
    written as ``if p.grad is not None``, so every parameter is skipped without
    any error being raised.
    """
    m = nn.Linear(3, 2)
    m(torch.randn(4, 3)).sum().backward()
    assert m.weight.grad is not None, "backward must populate .grad"

    m.zero_grad(set_to_none=False)

    assert m.weight.grad is not None, (
        "zero_grad(set_to_none=False) must keep .grad as a tensor")
    np.testing.assert_array_equal(
        m.weight.grad.numpy(), np.zeros((2, 3), dtype="float32"))
    assert tuple(m.weight.grad.shape) == tuple(m.weight.shape)
    assert str(m.weight.grad.dtype) == str(m.weight.dtype)


def test_zero_grad_set_to_none_true_drops_grads(grad_enabled):
    """The default (and explicit True) still clears to None, as torch does."""
    m = nn.Linear(3, 2)
    m(torch.randn(4, 3)).sum().backward()
    m.zero_grad(set_to_none=True)
    assert m.weight.grad is None
    m(torch.randn(4, 3)).sum().backward()
    m.zero_grad()
    assert m.weight.grad is None


def test_zero_grad_returns_none():
    """torch's zero_grad returns None, not self."""
    m = nn.Linear(3, 2)
    assert m.zero_grad() is None


def test_named_parameters_and_buffers_partition_cleanly():
    """A buffer is never a parameter and vice versa, as in torch."""
    net = _Net()
    params = {k for k, _ in net.named_parameters()}
    buffers = {k for k, _ in net.named_buffers()}
    assert params == {"lin.weight", "lin.bias"}
    assert buffers == {"keep", "drop"}
    assert not (params & buffers)


def test_named_modules_includes_self_under_empty_name():
    """torch yields ('', self) first, then children."""
    net = _Net()
    names = [k for k, _ in net.named_modules()]
    assert names[0] == ""
    assert "lin" in names
    mods = dict(net.named_modules())
    assert mods[""] is net
    assert mods["lin"] is net.lin


def test_parameters_matches_named_parameters():
    net = _Net()
    assert [id(p) for p in net.parameters()] == \
           [id(p) for _, p in net.named_parameters()]


def test_non_persistent_buffer_is_excluded_from_state_dict():
    """persistent=False buffers are reported but not serialized."""
    net = _Net()
    assert net._non_persistent_buffers_set == {"drop"}
    keys = set(net.state_dict().keys())
    assert "keep" in keys
    assert "drop" not in keys


def test_load_state_dict_roundtrip_is_exact():
    src, dst = _Net(), _Net()
    result = dst.load_state_dict(src.state_dict())
    assert list(result.missing_keys) == []
    assert list(result.unexpected_keys) == []
    np.testing.assert_array_equal(
        dst.lin.weight.numpy(), src.lin.weight.numpy())


def test_load_state_dict_reports_key_diff():
    """Unexpected/missing keys are reported rather than silently ignored."""
    net = _Net()
    sd = dict(net.state_dict())
    sd["nope.weight"] = torch.zeros(2)
    result = net.load_state_dict(sd, strict=False)
    assert "nope.weight" in set(result.unexpected_keys)


def test_load_state_dict_preserves_target_dtype():
    """Loading a float32 checkpoint into a float16 module keeps float16."""
    dst = _Net()
    src_sd = {k: v.float() for k, v in _Net().state_dict().items()}
    dst.lin.weight.assign(dst.lin.weight.float16())
    before = str(dst.lin.weight.dtype)
    dst.load_state_dict(src_sd, strict=False)
    assert str(dst.lin.weight.dtype) == before


def test_train_eval_toggle_is_recursive_and_returns_self():
    net = _Net()
    assert net.train() is net
    assert net.is_train and net.lin.is_train
    assert net.eval() is net
    assert not net.is_train and not net.lin.is_train
    assert net.train(False) is net
    assert not net.is_train


def test_to_dtype_casts_float_params_only():
    """``to(float64)`` must not touch integer buffers (torch 2.12.1 does not)."""
    net = _Net()
    net.register_buffer("ids", torch.tensor([1, 2, 3]))
    net.to(torch.float64)
    assert str(net.lin.weight.dtype) == "float64"
    assert str(net.ids.dtype) in ("int32", "int64"), \
        "integer buffers must survive to(float64)"


def test_to_returns_self_and_preserves_parameter_identity():
    """torch's Module.to is in place: parameter objects keep identity."""
    net = _Net()
    w = net.lin.weight
    assert net.to("cpu") is net
    assert net.lin.weight is w


def test_to_is_a_noop_without_device_or_dtype():
    net = _Net()
    w_before = net.lin.weight.numpy().copy()
    assert net.to() is net
    np.testing.assert_array_equal(net.lin.weight.numpy(), w_before)


def test_float_double_half_roundtrip():
    net = _Net()
    net.double()
    assert str(net.lin.weight.dtype) == "float64"
    net.float()
    assert str(net.lin.weight.dtype) == "float32"


def test_get_parameter_rejects_a_buffer():
    """torch raises AttributeError rather than returning the buffer."""
    net = _Net()
    assert tuple(net.get_parameter("lin.weight").shape) == (3, 4)
    with pytest.raises(AttributeError):
        net.get_parameter("keep")


def test_get_buffer_rejects_a_parameter():
    net = _Net()
    assert tuple(net.get_buffer("keep").shape) == (3,)
    with pytest.raises(AttributeError):
        net.get_buffer("lin.weight")


def test_get_submodule_resolves_dotted_path():
    net = _Net()
    assert net.get_submodule("lin") is net.lin
    assert net.get_submodule("") is net


def test_register_parameter_registers_the_name():
    net = _Net()
    p = torch.ones(2)
    net.register_parameter("extra", p)
    assert net.extra is p
    assert dict(net.named_parameters())["extra"] is p


def test_forward_result_matches_manual_matmul():
    """Numerical parity for the dispatch path, pinned against NumPy."""
    net = _Net()
    x = torch.randn(5, 4)
    got = net(x).numpy()
    w = net.lin.weight.numpy()
    b = net.lin.bias.numpy()
    np.testing.assert_allclose(got, x.numpy() @ w.T + b, rtol=1e-5, atol=1e-5)


def test_execute_and_forward_agree():
    """``execute`` and ``forward`` are the same computation."""
    net = _Net()
    x = torch.randn(3, 4)
    np.testing.assert_array_equal(net.execute(x).numpy(), net.forward(x).numpy())


# --------------------------------------------------------------------------
# pipelining knob: module-level state, not an install-time closure cell
# --------------------------------------------------------------------------

def test_execution_pipelining_roundtrip():
    previous = nn.Module.get_execution_pipelining()
    try:
        nn.Module.set_execution_pipelining(8)
        assert nn.Module.get_execution_pipelining() == 8
        nn.Module.set_execution_pipelining(0)
        assert nn.Module.get_execution_pipelining() == 0
    finally:
        nn.Module.set_execution_pipelining(previous)


def test_execution_pipelining_state_is_module_level():
    """The threshold lives in a module dict, so it is inspectable and resettable."""
    previous = nn.Module.get_execution_pipelining()
    try:
        nn.Module.set_execution_pipelining(4)
        assert nn_installer._pipeline_state["threshold"] == 4
    finally:
        nn.Module.set_execution_pipelining(previous)


def test_pipelining_does_not_change_results():
    """Turning the knob on must not perturb the numbers."""
    net = _Net()
    x = torch.randn(4, 4)
    previous = nn.Module.get_execution_pipelining()
    try:
        nn.Module.set_execution_pipelining(0)
        base = net(x).numpy().copy()
        nn.Module.set_execution_pipelining(1)
        np.testing.assert_allclose(net(x).numpy(), base, rtol=1e-6, atol=1e-6)
    finally:
        nn.Module.set_execution_pipelining(previous)


# --------------------------------------------------------------------------
# CPU and accelerator: these methods *are* the residency-migration surface,
# so CPU-only evidence would not cover what they exist to do.
# --------------------------------------------------------------------------

class TestModuleMethodsAcrossDevices:
    """Runs on every buildable device via instantiate_device_type_tests."""

    def test_forward_matches_numpy_on_this_device(self, device):
        net = _Net()
        x = torch.randn(6, 4)
        got = net(x).numpy()
        expected = x.numpy() @ net.lin.weight.numpy().T + net.lin.bias.numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_forward_agrees_with_the_cpu_path(self, device):
        """The same weights and input give the same answer on either device."""
        net = _Net()
        x_np = np.arange(24, dtype="float32").reshape(6, 4) / 24.0
        w = net.lin.weight.numpy().copy()
        b = net.lin.bias.numpy().copy()
        got = net(torch.tensor(x_np)).numpy()
        np.testing.assert_allclose(got, x_np @ w.T + b, rtol=1e-4, atol=1e-4)

    def test_to_preserves_parameter_identity_on_this_device(self, device):
        """Migration is in place: the Parameter object survives, only storage moves."""
        net = _Net()
        w = net.lin.weight
        net.to(device)
        assert net.lin.weight is w

    def test_to_dtype_leaves_integer_buffers_alone(self, device):
        net = _Net()
        net.register_buffer("ids", torch.tensor([1, 2, 3]))
        net.to(torch.float64)
        assert str(net.lin.weight.dtype) == "float64"
        assert str(net.ids.dtype) in ("int32", "int64")

    def test_round_trip_migration_preserves_values(self, device):
        """to(device) then to('cpu') is value-preserving, bit for bit."""
        net = _Net()
        before = net.lin.weight.numpy().copy()
        net.to(device)
        net.to("cpu")
        np.testing.assert_array_equal(net.lin.weight.numpy(), before)

    def test_named_parameters_are_stable_across_devices(self, device):
        net = _Net()
        before = [k for k, _ in net.named_parameters()]
        net.to(device)
        assert [k for k, _ in net.named_parameters()] == before

    def test_state_dict_round_trip_on_this_device(self, device):
        src, dst = _Net(), _Net()
        src.to(device)
        dst.to(device)
        result = dst.load_state_dict(src.state_dict())
        assert list(result.missing_keys) == []
        np.testing.assert_allclose(
            dst.lin.weight.numpy(), src.lin.weight.numpy(), rtol=0, atol=0)

    def test_zero_grad_set_to_none_false_zeroes_on_this_device(self, device):
        """The silent-wrong fix must hold on the accelerator too, not just CPU."""
        with unbridged_grad():
            m = nn.Linear(3, 2)
            m.to(device)
            m(torch.randn(4, 3)).sum().backward()
            assert m.weight.grad is not None
            m.zero_grad(set_to_none=False)
            assert m.weight.grad is not None
            np.testing.assert_array_equal(
                m.weight.grad.numpy(), np.zeros((2, 3), dtype="float32"))

    def test_zero_grad_set_to_none_true_drops_on_this_device(self, device):
        with unbridged_grad():
            m = nn.Linear(3, 2)
            m.to(device)
            m(torch.randn(4, 3)).sum().backward()
            m.zero_grad(set_to_none=True)
            assert m.weight.grad is None

    def test_train_eval_toggle_on_this_device(self, device):
        net = _Net()
        net.to(device)
        net.train()
        assert net.is_train and net.lin.is_train
        net.eval()
        assert not net.is_train and not net.lin.is_train


instantiate_device_type_tests(TestModuleMethodsAcrossDevices, globals())
