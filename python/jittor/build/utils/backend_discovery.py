"""Select a backend provider without importing or probing unselected backends."""

import os
import inspect
import sys
from typing import TYPE_CHECKING, Optional, Tuple

if sys.version_info >= (3, 8):
    from typing import Protocol
else:  # Python 3.7, declared in pyproject.toml.
    from typing_extensions import Protocol

if TYPE_CHECKING:
    from jittor_utils.build_config import BuildConfig, BuildContext


class Backend(Protocol):
    """A selected build provider; discovery never calls these hooks.

    configure returns a BuildConfig value without mutating its input/compiler.
    install_extern publishes libraries through BuildContext's services and
    returns whether its library setup handled the selected provider.
    These two hooks are required because bootstrap invokes them unconditionally.
    A requested provider's failure propagates; it must not pretend to be CPU.
    This Python build contract is distinct from the native BackendOps ABI.
    """

    def configure(self, context: "BuildContext") -> "BuildConfig": ...
    def install_extern(self, context: "BuildContext") -> bool: ...


class BackendPostProcess(Protocol):
    """Optional hook binding runtime operators after core construction."""

    def post_process(self, context: "BuildContext") -> object: ...


def validate_backend_provider(provider: Backend, name: str) -> None:
    """Check all bootstrap entry points before any provider compilation."""
    methods: Tuple[str, ...] = ("configure", "install_extern")
    if hasattr(provider, "post_process"):
        methods += ("post_process",)
    for method in methods:
        callback = getattr(provider, method, None)
        if not callable(callback):
            raise TypeError("backend provider %r must define %s(context)" % (name, method))
        try:
            inspect.signature(callback).bind(object())
        except (TypeError, ValueError) as error:
            raise TypeError("backend provider %r: %s must accept one BuildContext argument"
                            % (name, method)) from error


ENTRY_POINT_GROUP = "jittor.backends"
BUILTIN_PROVIDERS = {
    "acl": "jittor.backends.acl",
    "rocm": "jittor.backends.rocm",
    "corex": "jittor.backends.corex",
}


def requested_backend(environ=None, *, is_file=os.path.isfile):
    environ = os.environ if environ is None else environ
    explicit = environ.get("JT_BACKEND", "").strip().lower()
    if explicit:
        return {"npu": "acl", "hip": "rocm"}.get(explicit, explicit)
    hints = {
        "acl": ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "tikcc_path"),
        "rocm": ("ROCM_HOME", "ROCM_PATH", "HIP_PATH", "hipcc_path"),
        "corex": ("COREX_HOME",),
    }
    selected = [name for name, keys in hints.items()
                if any(environ.get(key) for key in keys)]
    if not selected:
        candidates = {
            "acl": "/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin/ccec",
            "rocm": "/opt/rocm/bin/hipcc",
            "corex": "/usr/local/corex/bin/clang++",
        }
        selected = [name for name, path in candidates.items() if is_file(path)]
    if len(selected) > 1:
        raise RuntimeError(
            "multiple backend SDKs are configured: %s; set JT_BACKEND explicitly"
            % ", ".join(sorted(selected)))
    return selected[0] if selected else None


def backend_entry_point(name, *, entries=None):
    try:
        from importlib import metadata
    except ImportError:
        import importlib_metadata as metadata
    if entries is None:
        entries = metadata.entry_points()
        if hasattr(entries, "select"):
            entries = entries.select(group=ENTRY_POINT_GROUP)
        else:
            entries = entries.get(ENTRY_POINT_GROUP, ())
    matches = [entry for entry in entries
               if entry.name == name and entry.group == ENTRY_POINT_GROUP]
    if len(matches) > 1:
        raise RuntimeError("duplicate backend entry point: " + name)
    if matches:
        return matches[0]
    # A source checkout may run without installed distribution metadata.
    if name in BUILTIN_PROVIDERS:
        return metadata.EntryPoint(
            name=name, value=BUILTIN_PROVIDERS[name], group=ENTRY_POINT_GROUP)
    raise RuntimeError("backend entry point is not installed: " + name)


def load_backend_provider(name, *, entries=None) -> Optional[Backend]:
    if name in (None, "cpu", "cuda"):
        return None
    provider = backend_entry_point(name, entries=entries).load()
    validate_backend_provider(provider, name)
    return provider
