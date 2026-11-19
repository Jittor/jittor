"""torch.ops.load_library fails like a missing extension, not a missing name."""
import pytest
import torch


def test_load_library_raises_import_error():
    with pytest.raises(ImportError, match="libtorch C\\+\\+ extension ABI"):
        torch.ops.load_library("/nonexistent/libtorch_extension.so")


def test_optional_extension_probe_still_registers_ops():
    # torchvision-style probe: a `try` around the extension load, then register
    # the library's ops anyway. The failure must be an ImportError so the probe
    # can carry on; an empty namespace returned by __getattr__ is not one.
    try:
        torch.ops.load_library("/nonexistent/libtorch_extension.so")
    except ImportError:
        pass
    library = torch.library.Library("optional_extension_probe", "DEF")
    library.define("identity(Tensor x) -> Tensor")
    library.impl("identity", lambda x: x, "CPU")
    x = torch.tensor([1.0, 2.0])
    assert torch.ops.optional_extension_probe.identity(x).tolist() == [1.0, 2.0]
