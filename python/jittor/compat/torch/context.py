"""Deterministic state and module publication for the Torch compatibility install."""

from __future__ import absolute_import

import importlib
import importlib.machinery
import sys
import types
from collections.abc import MutableMapping
from dataclasses import dataclass, field

from .._aliases import _is_deployed_torch_placeholder
from ..diagnostics import EXPECTED, swallowed
from .namespace import TorchNamespace


def _native_backend_for(target):
    """Resolve explicit namespace ownership without delegated attribute reads."""
    return target.owner if isinstance(target, TorchNamespace) else target


class TransformGetItemToIndex:
    """Scope the vmap getitem lowering hint for one Jittor owner.

    The compatibility surface historically exposed the depth as a private
    attribute on ``jittor``.  Keeping the owner explicit here prevents the
    compiler installer and numerical runtime from growing separate state
    machines, while the attribute remains observable for old callers.
    """

    DEPTH_ATTR = "_transform_getitem_to_index_depth"

    def __init__(self, owner):
        self.owner = owner
        self._previous_depth = 0

    def __enter__(self):
        self._previous_depth = int(getattr(self.owner, self.DEPTH_ATTR, 0))
        setattr(self.owner, self.DEPTH_ATTR, self._previous_depth + 1)
        return self

    def __exit__(self, *exc):
        # Always restore the exact entry value, including when the body raises
        # or when contexts are nested.
        setattr(self.owner, self.DEPTH_ATTR, self._previous_depth)
        return False


def getitem_transform_active(owner):
    """Return whether getitem-to-index lowering is active for ``owner``."""

    return bool(getattr(owner, TransformGetItemToIndex.DEPTH_ATTR, 0))


class InstallStepError(RuntimeError):
    """A required compatibility installer failed."""

    def __init__(self, step, error):
        self.step = step
        self.error = error
        super().__init__(
            "torch compatibility install step %r failed: %s" % (step, error)
        )


@dataclass(frozen=True)
class InstallReport:
    step: str
    required: bool
    status: str
    error: str = ""


class _RegistryModuleMap(MutableMapping):
    """Mapping facade that routes legacy assignment syntax through a registry."""

    def __init__(self, registry):
        self.registry = registry

    def __getitem__(self, name):
        return self.registry._modules[name]

    def __setitem__(self, name, module):
        self.registry.publish(name, module, replace=True)

    def __delitem__(self, name):
        del self.registry._modules[name]

    def __iter__(self):
        return iter(self.registry._modules)

    def __len__(self):
        return len(self.registry._modules)

    def setdefault(self, name, default=None):
        module = self.registry._modules.get(name)
        if module is not None:
            return module
        return self.registry.publish(name, default, replace=True)

    def pop(self, name, *default):
        return self.registry._modules.pop(name, *default)


class ModuleRegistry:
    """Create and publish ``torch.*`` modules without changing object identity."""

    def __init__(self, root_module, modules=None, *, native_backend=None):
        self.root_module = root_module
        self.native_backend = (
            _native_backend_for(root_module) if native_backend is None else native_backend
        )
        if isinstance(root_module, TorchNamespace) and self.native_backend is not root_module.owner:
            raise ValueError("registry native backend differs from namespace owner")
        self._modules = sys.modules if modules is None else modules
        self._published = {}
        self.module_map = _RegistryModuleMap(self)

    @property
    def target_namespace(self):
        """The publication target; root_module remains its legacy spelling."""
        return self.root_module

    @staticmethod
    def _ensure_import_metadata(name, module):
        """Give synthetic modules the metadata a real distribution provides.

        Installers build most ``torch.*`` nodes with ``ModuleType``.  Such
        nodes normally have an empty ``__package__`` and no ``ModuleSpec``;
        that is enough for direct attribute access but not for importlib's
        package discovery on a standalone wheel.  Preserve metadata from
        genuinely imported modules and fill only the missing synthetic
        fields.
        """

        if not isinstance(name, str) or not hasattr(module, "__name__"):
            return module
        parent = name.rsplit(".", 1)[0] if "." in name else ""
        if not getattr(module, "__package__", None):
            module.__package__ = name if hasattr(module, "__path__") else parent
        if getattr(module, "__spec__", None) is None:
            is_package = hasattr(module, "__path__")
            module.__spec__ = importlib.machinery.ModuleSpec(
                name, loader=None, is_package=is_package
            )
            if is_package and getattr(module, "__path__", None) is None:
                module.__path__ = []
        return module

    def get(self, name):
        return self._modules.get(name)

    def ensure(self, name, factory=None, package=False):
        module = self._modules.get(name)
        if module is not None:
            if package and not hasattr(module, "__path__"):
                module.__path__ = []
            self._ensure_import_metadata(name, module)
            return self.publish(name, module)
        module = factory(name) if factory is not None else types.ModuleType(name)
        if package and not hasattr(module, "__path__"):
            module.__path__ = []
        self._ensure_import_metadata(name, module)
        return self.publish(name, module)

    def publish(self, name, module, bind_parent=True, replace=False):
        self._ensure_import_metadata(name, module)
        current = self._modules.get(name)
        if current is not None and current is not module:
            root_replacement = name == "torch" and (
                _is_deployed_torch_placeholder(current)
            )
            if (name == "torch" and not root_replacement) or (
                name != "torch" and not replace
            ):
                raise RuntimeError(
                    "module %r already published with a different object" % name
                )
        self._modules[name] = module
        self._published[name] = module
        if bind_parent and "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = self._modules.get(parent_name)
            if parent is not None:
                # Compatibility installers deliberately replace placeholder
                # attributes (often SimpleNamespace objects) with the one module
                # object published in sys.modules.
                setattr(parent, attr, module)
        return module

    def alias(self, alias_name, canonical):
        module = (
            importlib.import_module(canonical)
            if isinstance(canonical, str)
            else canonical
        )
        return self.publish(alias_name, module)


@dataclass
class InstallContext:
    jittor_module: object
    registry: ModuleRegistry
    strict: bool = True
    reports: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    native_backend: object = None

    MARKERS_ATTR = "_torch_compat_install_steps"
    CONTEXT_ATTR = "_torch_compat_install_context"
    COMPLETE_ATTR = "_torch_compat_install_complete"

    def __post_init__(self):
        if self.native_backend is None:
            self.native_backend = _native_backend_for(self.target_namespace)
        if self.registry.target_namespace is not self.target_namespace:
            raise ValueError("install context and registry target namespaces differ")
        if self.registry.native_backend is not self.native_backend:
            raise ValueError("install context and registry native backends differ")
        if self.MARKERS_ATTR not in vars(self.target_namespace):
            setattr(self.target_namespace, self.MARKERS_ATTR, {})

    @property
    def target_namespace(self):
        """Installer write target, preserving the legacy jittor_module field."""
        return self.jittor_module

    @classmethod
    def for_module(cls, jittor_module, strict=True, *, native_backend=None):
        backend = _native_backend_for(jittor_module) if native_backend is None else native_backend
        existing = vars(jittor_module).get(cls.CONTEXT_ATTR)
        if isinstance(existing, cls):
            if (existing.target_namespace is not jittor_module
                    or existing.registry.target_namespace is not jittor_module):
                raise ValueError("existing install context belongs to a different target namespace")
            if (existing.native_backend is not backend
                    or existing.registry.native_backend is not backend):
                raise ValueError("existing install context belongs to a different native backend")
            existing.strict = bool(strict)
            return existing
        context = cls(
            jittor_module=jittor_module,
            registry=ModuleRegistry(jittor_module, native_backend=backend),
            strict=bool(strict),
            native_backend=backend,
        )
        setattr(jittor_module, cls.CONTEXT_ATTR, context)
        return context

    @property
    def markers(self):
        return vars(self.target_namespace)[self.MARKERS_ATTR]

    @property
    def complete(self):
        return bool(vars(self.target_namespace).get(self.COMPLETE_ATTR, False))

    def _record(self, step, required, status, error=""):
        report = InstallReport(step, required, status, error)
        self.reports.append(report)
        return report

    def run_required(self, step, installer):
        if self.markers.get(step) == "complete":
            self._record(step, True, "skipped")
            return None
        try:
            result = installer(self)
        except EXPECTED as error:
            swallowed("torch/context.py run_required: result = installer(self)", error)
            self._record(step, True, "failed", repr(error))
            raise InstallStepError(step, error) from error
        self.markers[step] = "complete"
        self._record(step, True, "complete")
        return result

    def run_optional(self, step, installer):
        if self.markers.get(step) == "complete":
            self._record(step, False, "skipped")
            return None
        try:
            result = installer(self)
        except EXPECTED as error:
            warned = self.state.setdefault("_optional_warned_steps", set())
            if step not in warned:
                swallowed("torch/context.py run_optional: result = installer(self)", error)
                warned.add(step)
            self.markers[step] = "failed"
            self._record(step, False, "failed", repr(error))
            return None
        self.markers[step] = "complete"
        self._record(step, False, "complete")
        return result

    def optional_failures(self):
        """Return the latest failure report for each optional step, if any."""
        latest = {}
        for report in self.reports:
            if not report.required and report.status == "failed":
                latest[report.step] = report
        return tuple(latest[name] for name in sorted(latest))

    def mark_complete(self):
        setattr(self.target_namespace, self.COMPLETE_ATTR, True)


def registry_for(jittor_module, registry=None):
    """Return the active registry, including for legacy one-argument helpers."""

    if isinstance(registry, ModuleRegistry):
        return registry
    context = vars(jittor_module).get(InstallContext.CONTEXT_ATTR)
    if isinstance(context, InstallContext):
        if (context.target_namespace is not jittor_module
                or context.registry.target_namespace is not jittor_module):
            raise ValueError("install registry belongs to a different target namespace")
        if context.registry.native_backend is not context.native_backend:
            raise ValueError("install registry belongs to a different native backend")
        return context.registry
    return ModuleRegistry(jittor_module)
