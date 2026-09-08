"""Stable policy APIs and the explicit owner of the Pillow bridge."""
import weakref
import jittor as jt
from .context import get_install_context
from .api_delegates import bind_delegates
from .fidelity import Fidelity, register_fidelity
from ..stub_policy import allow_stub, set_allow_stub
from ..diagnostics import EXPECTED, swallowed
from ..transaction import set_attr


def compat_allow_stub(value=None):
    if value is not None:
        set_allow_stub(value)
    return allow_stub()


def compat_report_torch_api_version(enable=True):
    context = get_install_context(jt)
    target = context.target_namespace
    native = getattr(target, "__jittor_version__", None) or context.state["core_install_api"]["native_version"]
    api = getattr(target, "__torch_version__", None)
    set_attr(target, "__version__", api if enable and api is not None else native, context=context)
    return target.__version__


def pillow_fromarray(obj, mode=None, *args, **kwargs):
    from PIL import Image
    reference = vars(Image).get("_jittor_fromarray_context")
    context = reference() if reference is not None else None
    if context is None:
        raise RuntimeError("Pillow compatibility owner is no longer active")
    original = context.state["core_install_api"]["pil_fromarray"]
    if mode in ("RGB", "RGBA", "L") and getattr(obj, "dtype", None) is not None:
        try:
            import numpy as np
            if obj.dtype == np.int8:
                obj = obj.view(np.uint8)
        except EXPECTED as error:
            swallowed("Pillow int8 image reinterpretation", error)
    return original(obj, mode=mode, *args, **kwargs)


setattr(pillow_fromarray, "_jittor_torch_compat", True)


def bind_core_install_api(context):
    target = context.target_namespace
    delegates = dict(context.state.get("core_install_api", {}))
    delegates.setdefault("native_version", getattr(target, "__version__", None))
    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image is not None:
        original = Image.fromarray
        if original is pillow_fromarray:
            reference = vars(Image).get("_jittor_fromarray_context")
            previous = reference() if reference is not None else None
            if previous is None:
                raise RuntimeError("Pillow adapter has no live delegate owner")
            original = previous.state["core_install_api"]["pil_fromarray"]
        delegates["pil_fromarray"] = original
    bind_delegates(context, "core_install_api", delegates)
    if Image is not None:
        set_attr(Image, "_jittor_fromarray_context", weakref.ref(context), context=context)
        set_attr(Image, "fromarray", pillow_fromarray, context=context)
    target.compat_allow_stub = compat_allow_stub
    target.compat_report_torch_api_version = compat_report_torch_api_version


for name, function, detail in (
    ("torch.compat_allow_stub", compat_allow_stub, "Explicit compatibility escape-hatch policy"),
    ("torch.compat_report_torch_api_version", compat_report_torch_api_version,
     "Reports the active frontend API version; writes are transaction-aware"),
    ("PIL.Image.fromarray", pillow_fromarray,
     "Only int8 RGB/RGBA/L arrays are reinterpreted as uint8; other inputs delegate to Pillow"),
):
    register_fidelity(name, function, Fidelity.APPROXIMATE, detail)
