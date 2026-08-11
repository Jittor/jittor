"""Discovery and loading machinery for optional Jittor extension backends."""

from __future__ import annotations

import glob
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import pathlib
import sys
import threading
from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - exercised on Python 3.7 CI
    import importlib_metadata  # type: ignore[no-redef]


EXTERNAL_BACKEND_ENTRY_POINT = "jittor.external_backends"
_BACKENDS = {}
_BACKEND_HINTS = {}
_ENTRY_POINTS_LOADED = set()
_REGISTRY_LOCK = threading.RLock()
_SOURCE_IMPORT_LOCK = threading.RLock()


def _names(values: Sequence[str]) -> Tuple[str, ...]:
    return (values,) if isinstance(values, str) else tuple(values)


def _split_env_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    items = []
    for item in value.replace(",", os.pathsep).split(os.pathsep):
        item = item.strip()
        if item and item not in items:
            items.append(item)
    return items


def _expand_paths(root: pathlib.Path, items: Sequence[str]) -> List[str]:
    paths = []
    for raw in items:
        path = pathlib.Path(raw).expanduser()
        if not path.is_absolute():
            path = root / path
        text = os.fspath(path)
        if any(char in text for char in "*?[]"):
            paths.extend(
                os.fspath(pathlib.Path(hit).resolve())
                for hit in sorted(glob.glob(text, recursive=True))
            )
        else:
            paths.append(os.fspath(path.resolve()))
    return list(dict.fromkeys(paths))


def _default_project_roots(environment_names: Sequence[str]) -> List[pathlib.Path]:
    roots = []
    for name in environment_names:
        value = os.environ.get(name)
        if value:
            roots.append(pathlib.Path(value).expanduser())
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and argv0 not in ("-c", "-m"):
        entry = pathlib.Path(argv0).expanduser()
        if entry.suffix == ".py":
            roots.append(entry.parent)
    try:
        roots.append(pathlib.Path.cwd())
    except OSError:
        pass
    runtime = os.environ.get("JITTOR_TORCH_RUNTIME_ROOT")
    if runtime:
        path = pathlib.Path(runtime).expanduser()
        if path.name == "jittor_torch" and path.parent.name == ".cache":
            roots.append(path.parent.parent)
    output = []
    seen = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        key = os.fspath(resolved)
        if key not in seen:
            seen.add(key)
            output.append(resolved)
    return output


@dataclass(frozen=True)
class ExternalBackendSpec:
    """All discovery policy that differs between external backends."""

    name: str
    public_functions: Tuple[str, ...]
    source_envs: Tuple[str, ...] = ()
    module_env: Optional[str] = None
    module_names: Tuple[str, ...] = ()
    hook_names: Tuple[str, ...] = ()
    manifest_names: Tuple[str, ...] = ()
    relative_source_dirs: Tuple[str, ...] = ()
    source_root_names: Tuple[str, ...] = ()
    project_root_envs: Tuple[str, ...] = ()
    submodule_attrs: Tuple[str, ...] = ()
    environment_names: Tuple[str, ...] = ()
    build_script: str = "build_jittor.py"
    default_module_name: Optional[str] = None
    build_namespace: Optional[str] = None
    force_build_env: Optional[str] = None
    source_predicates: Tuple[Callable[[pathlib.Path], bool], ...] = field(
        default=(), compare=False
    )

    def __post_init__(self) -> None:
        if not self.name or not self.public_functions:
            raise ValueError("external backend needs a name and at least one public function")


@dataclass(frozen=True)
class BackendAttempt:
    source: str
    status: str
    detail: Optional[str] = None


@dataclass(frozen=True)
class BackendReport:
    name: str
    attempts: Tuple[BackendAttempt, ...]
    generation: int

    @property
    def failures(self) -> Tuple[BackendAttempt, ...]:
        return tuple(item for item in self.attempts if item.status == "failed")


@dataclass(frozen=True)
class BackendEntryPointResult:
    name: str
    value: str
    status: str
    detail: Optional[str] = None


class ExternalBackend:
    """Thread-safe environment/module/source/manifest backend resolver."""

    def __init__(
        self,
        spec: ExternalBackendSpec,
        *,
        log: Optional[Callable[[str], None]] = None,
        verbose: Optional[Callable[[], bool]] = None,
        build_root: Optional[Callable[..., str]] = None,
        extension_loader: Optional[Callable[..., ModuleType]] = None,
        setup_builder: Optional[Callable[[pathlib.Path], bool]] = None,
        special_source_loader: Optional[Callable[[pathlib.Path], Optional[ModuleType]]] = None,
        capability_miss: Optional[Callable[[Optional[ModuleType], object], Optional[str]]] = None,
        prepare_capability: Optional[Callable[[object], None]] = None,
    ):
        self.spec = spec
        self._log_callback = log
        self._verbose_callback = verbose
        self._build_root_callback = build_root
        self._extension_loader = extension_loader
        self._setup_builder = setup_builder
        self._special_source_loader = special_source_loader
        self._capability_miss = capability_miss
        self._prepare_capability = prepare_capability
        self._cache: Dict[Tuple[object, ...], Optional[ModuleType]] = {}
        self._lock = threading.RLock()
        self._generation = 0
        self._last_report = BackendReport(spec.name, (), 0)
        self._source_env_hints: List[str] = []
        self._project_root_env_hints: List[str] = []
        self._relative_source_dir_hints: List[str] = []
        self._environment_name_hints: List[str] = []

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def last_report(self) -> BackendReport:
        return self._last_report

    @staticmethod
    def _merged(primary: Sequence[str], hints: Sequence[str]) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(tuple(primary) + tuple(hints)))

    def extend_discovery(
        self,
        *,
        source_envs: Sequence[str] = (),
        project_root_envs: Sequence[str] = (),
        relative_source_dirs: Sequence[str] = (),
        environment_names: Sequence[str] = (),
    ) -> None:
        """Add project-specific discovery hints without replacing the resolver."""

        with self._lock:
            for target, values in (
                (self._source_env_hints, source_envs),
                (self._project_root_env_hints, project_root_envs),
                (self._relative_source_dir_hints, relative_source_dirs),
                (self._environment_name_hints, environment_names),
            ):
                for value in _names(values):
                    if value and value not in target:
                        target.append(value)
            self._cache.clear()

    def _log(self, message: str) -> None:
        if self._log_callback is not None:
            self._log_callback(message)

    def _verbose(self) -> bool:
        return bool(self._verbose_callback and self._verbose_callback())

    def module_names(self) -> List[str]:
        names = _split_env_list(
            os.environ.get(self.spec.module_env) if self.spec.module_env else None
        )
        for name in self.spec.module_names:
            if name not in names:
                names.append(name)
        return names

    def project_roots(self) -> List[pathlib.Path]:
        names = self._merged(self.spec.project_root_envs, self._project_root_env_hints)
        return _default_project_roots(names)

    def environment_names(self) -> Tuple[str, ...]:
        names = (
            self.spec.environment_names
            + self.spec.source_envs
            + self.spec.project_root_envs
        )
        if self.spec.module_env:
            names += (self.spec.module_env,)
        hints = (
            self._environment_name_hints
            + self._source_env_hints
            + self._project_root_env_hints
        )
        return self._merged(names, hints)

    def environment_key(self) -> Tuple[Tuple[str, Optional[str]], ...]:
        return tuple((name, os.environ.get(name)) for name in self.environment_names())

    def looks_like_source_root(self, root: pathlib.Path, explicit: bool = False) -> bool:
        for predicate in self.spec.source_predicates:
            try:
                if predicate(root):
                    return True
            except (OSError, ValueError):
                continue
        if any((root / name).is_file() for name in self.spec.manifest_names):
            return True
        if explicit and (root / "setup.py").is_file():
            return True
        if self.spec.build_script and (root / self.spec.build_script).is_file():
            return True
        root_names = set(self.spec.source_root_names) | set(self.spec.module_names)
        if root.name in root_names and (root / "__init__.py").is_file():
            return True
        for name in self.module_names():
            if (root / name / "__init__.py").is_file():
                return True
        return root.name in root_names and (root / "setup.py").is_file()

    def source_roots(self, explicit_only: bool = False) -> List[str]:
        candidates = []
        source_envs = self._merged(self.spec.source_envs, self._source_env_hints)
        relative_dirs = self._merged(
            self.spec.relative_source_dirs, self._relative_source_dir_hints
        )
        for name in source_envs:
            for raw in _split_env_list(os.environ.get(name)):
                candidates.append((pathlib.Path(raw).expanduser(), True))
        if not explicit_only:
            for base in self.project_roots():
                candidates.append((base, False))
                candidates.extend((base / relative, False) for relative in relative_dirs)
        output = []
        seen = set()
        for root, explicit in candidates:
            try:
                resolved = root.resolve()
            except OSError:
                continue
            key = os.fspath(resolved)
            if key in seen or not resolved.is_dir():
                continue
            if self.looks_like_source_root(resolved, explicit=explicit):
                seen.add(key)
                output.append(key)
        return output

    def has_public_api(self, module: object) -> bool:
        return any(callable(getattr(module, name, None)) for name in self.spec.public_functions)

    def _call_hook(self, hook):
        kwargs = {"build_root": self.default_build_root("hooks"), "verbose": self._verbose()}
        try:
            signature = inspect.signature(hook)
        except (TypeError, ValueError):
            signature = None
        if signature is None:
            try:
                return hook(**kwargs)
            except TypeError:
                return hook()
        parameters = signature.parameters
        has_kwargs = any(item.kind == item.VAR_KEYWORD for item in parameters.values())
        accepted = {key: value for key, value in kwargs.items() if has_kwargs or key in parameters}
        return hook(**accepted)

    def select_backend(self, module: ModuleType, allow_hooks: bool = True) -> Optional[ModuleType]:
        if self.has_public_api(module):
            return module
        for attribute in self.spec.submodule_attrs:
            candidate = getattr(module, attribute, None)
            if isinstance(candidate, ModuleType) and self.has_public_api(candidate):
                return candidate
        if allow_hooks:
            for name in self.spec.hook_names:
                hook = getattr(module, name, None)
                if callable(hook):
                    candidate = self._call_hook(hook)
                    if isinstance(candidate, ModuleType):
                        selected = self.select_backend(candidate, allow_hooks=False)
                        if selected is not None:
                            return selected
        return None

    def try_import(self, name: str) -> Optional[ModuleType]:
        module = importlib.import_module(name)
        return self.select_backend(module)

    def import_installed(self) -> Optional[ModuleType]:
        for name in self.module_names():
            try:
                module = self.try_import(name)
            except Exception as exc:
                self._log("import %s failed: %s" % (name, exc))
                continue
            if module is not None:
                return module
            self._log("module %s has no %s entry points" % (name, self.spec.name))
        return None

    def _inside_root(self, module: ModuleType, root: pathlib.Path) -> bool:
        raw = getattr(module, "__file__", None)
        if not raw:
            return False
        try:
            pathlib.Path(raw).resolve().relative_to(root.resolve())
            return True
        except (OSError, ValueError):
            return False

    def import_local(self, root: pathlib.Path) -> Optional[ModuleType]:
        for name in self.module_names():
            if not (
                (root.name == name and (root / "__init__.py").is_file())
                or (root / name / "__init__.py").is_file()
            ):
                continue
            existing = sys.modules.get(name)
            displaced = {}
            if isinstance(existing, ModuleType) and not self._inside_root(existing, root):
                for key in list(sys.modules):
                    if key == name or key.startswith(name + "."):
                        displaced[key] = sys.modules.pop(key)
                importlib.invalidate_caches()
            try:
                imported = importlib.import_module(name)
                if not self._inside_root(imported, root):
                    raise ImportError("module %s resolved outside %s" % (name, root))
                selected = self.select_backend(imported)
                if selected is not None:
                    return selected
                raise ImportError("module %s has no %s entry points" % (name, self.spec.name))
            except Exception as exc:
                self._log("import local %s from %s failed: %s" % (name, root, exc))
                for key in list(sys.modules):
                    if key == name or key.startswith(name + "."):
                        sys.modules.pop(key, None)
                sys.modules.update(displaced)
        return None

    def manifest_paths(self, root: pathlib.Path) -> List[pathlib.Path]:
        return [root / name for name in self.spec.manifest_names if (root / name).is_file()]

    def default_build_root(self, *parts: str) -> str:
        if self._build_root_callback is not None:
            return self._build_root_callback(*parts)
        root = os.environ.get("JITTOR_TORCH_EXTENSIONS_DIR")
        if not root:
            runtime = os.environ.get("JITTOR_TORCH_RUNTIME_ROOT")
            root = os.path.join(runtime, "torch_extensions") if runtime else os.path.join(
                os.path.expanduser("~"), ".cache", "jittor_torch_extensions"
            )
        path = os.path.join(root, self.spec.build_namespace or self.spec.name, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    def _load_extension(self, **kwargs) -> ModuleType:
        if self._extension_loader is not None:
            return self._extension_loader(**kwargs)
        from jittor.compat.shim.cpp_extension.torch_utils import load

        return load(**kwargs)

    def load_manifest(self, root: pathlib.Path, manifest: pathlib.Path) -> Optional[ModuleType]:
        try:
            with manifest.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as exc:
            self._log("read manifest %s failed: %s" % (manifest, exc))
            return None
        sources = data.get("sources") or data.get("source_files")
        if not sources:
            self._log("manifest %s has no sources" % manifest)
            return None
        if isinstance(sources, str):
            sources = [sources]
        module_name = data.get("module") or data.get("name")
        if not module_name and self.spec.module_env:
            module_name = os.environ.get(self.spec.module_env)
        module_name = module_name or self.spec.default_module_name or (self.spec.name + "_jittor")
        if not isinstance(module_name, str) or not module_name:
            self._log("manifest %s has invalid module name" % manifest)
            return None
        includes = data.get("include_dirs") or data.get("extra_include_paths") or []
        if isinstance(includes, str):
            includes = [includes]
        include_dirs = [
            os.fspath((root / relative).resolve())
            for relative in ("include", "csrc", "src")
            if (root / relative).is_dir()
        ]
        include_dirs.extend(_expand_paths(root, includes))
        build_dir = data.get("build_directory") or data.get("build_dir")
        if build_dir:
            build_path = pathlib.Path(build_dir).expanduser()
            if not build_path.is_absolute():
                build_path = root / build_path
            build_dir = os.fspath(build_path.resolve())
            os.makedirs(build_dir, exist_ok=True)
        else:
            digest = hashlib.sha256(
                (os.fspath(root.resolve()) + "|" + os.fspath(manifest.resolve())).encode("utf-8")
            ).hexdigest()[:16]
            build_dir = self.default_build_root(module_name.replace(".", "_"), digest)
        try:
            module = self._load_extension(
                name=module_name.split(".")[-1],
                sources=_expand_paths(root, sources),
                extra_include_paths=list(dict.fromkeys(include_dirs)),
                extra_cflags=list(data.get("extra_cflags") or data.get("cflags") or []),
                extra_cuda_cflags=list(
                    data.get("extra_cuda_cflags") or data.get("cuda_cflags") or []
                ),
                extra_ldflags=list(data.get("extra_ldflags") or data.get("ldflags") or []),
                build_directory=build_dir,
                verbose=self._verbose(),
            )
        except Exception as exc:
            self._log("compile manifest %s failed: %s" % (manifest, exc))
            return None
        selected = self.select_backend(module)
        if selected is None:
            self._log("compiled module %s has no %s entry points" % (module_name, self.spec.name))
        return selected

    def load_build_script(self, root: pathlib.Path) -> Optional[ModuleType]:
        if not self.spec.build_script:
            return None
        path = root / self.spec.build_script
        if not path.is_file():
            return None
        digest = hashlib.sha256(os.fspath(root).encode("utf-8")).hexdigest()[:16]
        name = "_jittor_%s_build_%s" % (self.spec.name.replace("-", "_"), digest)
        try:
            spec = importlib.util.spec_from_file_location(name, os.fspath(path))
            if spec is None or spec.loader is None:
                raise RuntimeError("cannot load %s" % path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return self.select_backend(module)
        except Exception as exc:
            self._log("load %s failed: %s" % (path, exc))
            return None

    def build_setup(self, root: pathlib.Path) -> bool:
        if not (root / "setup.py").is_file():
            return False
        if self._setup_builder is not None:
            return bool(self._setup_builder(root))
        try:
            from jittor.compat.shim import bootstrap

            force = bool(
                self.spec.force_build_env
                and os.environ.get(self.spec.force_build_env, "").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            bootstrap.build_extension_dirs(
                [os.fspath(root)], force=force, verbose=self._verbose()
            )
            return True
        except Exception as exc:
            self._log("build setup.py %s failed: %s" % (root, exc))
            return False

    @staticmethod
    def _add_source_to_sys_path(root: pathlib.Path) -> None:
        for path in (root, root.parent):
            text = os.fspath(path)
            if text in sys.path:
                sys.path.remove(text)
            sys.path.insert(0, text)

    @staticmethod
    def _capture_source_import_state():
        return tuple(sys.path), dict(sys.modules)

    @staticmethod
    def _restore_source_import_state(state) -> None:
        source_path, source_modules = state
        sys.path[:] = source_path
        for name in set(sys.modules).difference(source_modules):
            sys.modules.pop(name, None)
        for name, module in source_modules.items():
            if name not in sys.modules or sys.modules[name] is not module:
                sys.modules[name] = module
        importlib.invalidate_caches()

    def load_source_root(self, raw_root: str) -> Optional[ModuleType]:
        root = pathlib.Path(raw_root).expanduser().resolve()
        self._add_source_to_sys_path(root)
        if self._special_source_loader is not None:
            try:
                claimed = any(predicate(root) for predicate in self.spec.source_predicates)
            except (OSError, ValueError):
                claimed = False
            if claimed:
                return self._special_source_loader(root)
        for manifest in self.manifest_paths(root):
            module = self.load_manifest(root, manifest)
            if module is not None:
                return module
        module = self.import_local(root) or self.load_build_script(root)
        if module is not None:
            return module
        if self.build_setup(root):
            return self.import_installed()
        return None

    def configuration_key(self, capability_key: object = None) -> Tuple[object, ...]:
        return (
            self.environment_key(),
            tuple(self.source_roots()),
            capability_key,
        )

    def _candidate_capability_miss(self, backend, capability_key):
        if (
            backend is not None
            and self._capability_miss is not None
            and capability_key is not None
        ):
            return self._capability_miss(backend, capability_key)
        return None

    def _load_candidate(self, source, capability_key):
        if source is None:
            backend = self.import_installed()
            return backend, self._candidate_capability_miss(backend, capability_key)

        # Source discovery changes process-global import state. Serialize these
        # transactions across resolvers and commit only a usable candidate.
        with _SOURCE_IMPORT_LOCK:
            state = self._capture_source_import_state()
            try:
                backend = self.load_source_root(source)
                miss = self._candidate_capability_miss(backend, capability_key)
            except BaseException:
                self._restore_source_import_state(state)
                raise
            if backend is None or miss is not None:
                self._restore_source_import_state(state)
            return backend, miss

    def load(self, capability_key: object = None, force: bool = False) -> Optional[ModuleType]:
        with self._lock:
            if self._prepare_capability is not None and capability_key is not None:
                self._prepare_capability(capability_key)
            key = self.configuration_key(capability_key)
            if not force and key in self._cache:
                return self._cache[key]
            self._generation += 1
            attempts = []
            explicit = self.source_roots(explicit_only=True)
            candidates = [("explicit", root) for root in explicit]
            candidates.append(("installed", None))
            candidates.extend(
                ("source", root) for root in self.source_roots() if root not in explicit
            )
            backend = None
            for kind, source in candidates:
                try:
                    backend, miss = self._load_candidate(source, capability_key)
                except Exception as exc:
                    attempts.append(BackendAttempt(str(source or kind), "failed", repr(exc)))
                    continue
                if backend is None:
                    attempts.append(BackendAttempt(str(source or kind), "unavailable"))
                    continue
                if miss is not None:
                    attempts.append(BackendAttempt(str(source or kind), "capability_miss", miss))
                    backend = None
                    continue
                attempts.append(BackendAttempt(str(source or kind), "loaded"))
                break
            self._cache[key] = backend
            self._last_report = BackendReport(self.spec.name, tuple(attempts), self._generation)
            return backend

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()


def register_external_backend(backend) -> ExternalBackend:
    """Register an ``ExternalBackend`` or ``ExternalBackendSpec`` by name."""

    if isinstance(backend, ExternalBackendSpec):
        backend = ExternalBackend(backend)
    if not isinstance(backend, ExternalBackend):
        raise TypeError("external backend registration requires a spec or resolver")
    with _REGISTRY_LOCK:
        existing = _BACKENDS.get(backend.spec.name)
        if existing is not None and existing is not backend:
            existing_policy = dict(vars(existing.spec))
            backend_policy = dict(vars(backend.spec))
            existing_policy.pop("source_predicates", None)
            backend_policy.pop("source_predicates", None)
            if existing_policy != backend_policy:
                raise ValueError("external backend %r is already registered" % backend.spec.name)
            return existing
        _BACKENDS[backend.spec.name] = backend
        hints = _BACKEND_HINTS.get(backend.spec.name, {})
        if hints:
            backend.extend_discovery(**hints)
    return backend


def register_external_backend_hint(
    name: str,
    *,
    source_envs: Sequence[str] = (),
    project_root_envs: Sequence[str] = (),
    relative_source_dirs: Sequence[str] = (),
    environment_names: Sequence[str] = (),
) -> None:
    """Extend a resolver's discovery policy, including before it is registered."""

    if not isinstance(name, str) or not name:
        raise ValueError("external backend hint needs a backend name")
    additions = {
        "source_envs": _names(source_envs),
        "project_root_envs": _names(project_root_envs),
        "relative_source_dirs": _names(relative_source_dirs),
        "environment_names": _names(environment_names),
    }
    with _REGISTRY_LOCK:
        current = _BACKEND_HINTS.setdefault(name, {})
        for key, values in additions.items():
            current[key] = tuple(dict.fromkeys(tuple(current.get(key, ())) + values))
        backend = _BACKENDS.get(name)
        if backend is not None:
            backend.extend_discovery(**additions)


def registered_external_backends() -> Mapping[str, ExternalBackend]:
    with _REGISTRY_LOCK:
        return dict(_BACKENDS)


def external_backend_for_source_root(root) -> Optional[ExternalBackend]:
    """Return the resolver claiming *root* or one of its source-tree parents."""

    path = pathlib.Path(root).expanduser().resolve()
    with _REGISTRY_LOCK:
        backends = tuple(_BACKENDS.values())
    for backend in backends:
        for candidate in (path,) + tuple(path.parents):
            if backend.looks_like_source_root(candidate):
                return backend
    return None


def _entry_points(group: str):
    discovered = importlib_metadata.entry_points()
    select = getattr(discovered, "select", None)
    if callable(select):
        return list(select(group=group))
    return list(discovered.get(group, ()))


def load_external_backend_entry_points() -> Tuple[BackendEntryPointResult, ...]:
    """Load installed backend providers independently and exactly once."""

    results = []
    try:
        entry_points = _entry_points(EXTERNAL_BACKEND_ENTRY_POINT)
    except Exception as exc:
        return (BackendEntryPointResult("discovery", EXTERNAL_BACKEND_ENTRY_POINT, "failed", repr(exc)),)
    with _REGISTRY_LOCK:
        for entry_point in entry_points:
            key = (getattr(entry_point, "name", ""), getattr(entry_point, "value", repr(entry_point)))
            if key in _ENTRY_POINTS_LOADED:
                results.append(BackendEntryPointResult(key[0], key[1], "already_loaded"))
                continue
            try:
                value = entry_point.load()
                if callable(value) and not isinstance(value, type):
                    value = value()
                if value is None:
                    pass
                elif isinstance(value, (ExternalBackend, ExternalBackendSpec)):
                    register_external_backend(value)
                elif isinstance(value, (tuple, list)):
                    for backend in value:
                        register_external_backend(backend)
                else:
                    raise TypeError("entry point did not return an external backend")
            except Exception as exc:
                results.append(BackendEntryPointResult(key[0], key[1], "failed", repr(exc)))
                continue
            _ENTRY_POINTS_LOADED.add(key)
            results.append(BackendEntryPointResult(key[0], key[1], "loaded"))
    return tuple(results)


__all__ = [
    "EXTERNAL_BACKEND_ENTRY_POINT",
    "BackendAttempt",
    "BackendEntryPointResult",
    "BackendReport",
    "ExternalBackend",
    "ExternalBackendSpec",
    "external_backend_for_source_root",
    "load_external_backend_entry_points",
    "register_external_backend",
    "register_external_backend_hint",
    "registered_external_backends",
]
