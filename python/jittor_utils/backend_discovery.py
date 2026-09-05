"""Select a backend provider without importing or probing unselected backends."""

import os


ENTRY_POINT_GROUP = "jittor.backends"
BUILTIN_PROVIDERS = {
    "acl": "jittor.extern.acl.acl_compiler",
    "rocm": "jittor.extern.rocm.rocm_compiler",
    "corex": "jittor.extern.corex.corex_compiler",
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


def load_backend_provider(name, *, entries=None):
    if name in (None, "cpu", "cuda"):
        return None
    provider = backend_entry_point(name, entries=entries).load()
    if not callable(getattr(provider, "configure", None)):
        raise TypeError("backend provider must define configure(context): " + name)
    return provider
