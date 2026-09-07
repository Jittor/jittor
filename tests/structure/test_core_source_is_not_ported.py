"""4.12 acceptance: the core sources are no longer the input to a port.

The deleted machinery (``process_jittor_source`` / ``process_acl``) worked by
copying the whole checkout into ``<cache>/<backend>_jittor``, running a textual
substitution over every ``.h/.cc/.cu/.cuh`` on the way, and then pointing the
build's source root at that copy.  Deleting the two functions is not enough to
keep it gone: the shape can come back under any other name, and the old guard
in ``test_backend_conversion_boundary`` could not have seen it -- that one
matches *names* (``transform_sources(``, ``process_acl``) and only under
``python/jittor``, which is why the definition survived in ``jittor_utils``
with a caller spelled ``transform_sources=jit_utils.process_jittor_source``
(an assignment, no call parentheses) for the whole migration.

So this file matches the *shape* instead, in three places where a port has to
show itself:

* the provider contract cannot offer a source-rewriting service,
* no production function may mirror a tree and rewrite native sources in it,
* nothing may redirect the source root away from the checkout.

A backend that needs different code gets a registered implementation of its
own; it does not get a rewritten copy of everybody else's.
"""

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

#: Production trees.  Tests legitimately build fake checkouts in ``tmp_path``.
PRODUCTION = ("python/jittor", "backends")

#: Only the bootstrap may say where the sources are.
SOURCE_ROOT_OWNER = "python/jittor/build/compiler.py"

#: A rewriting port has to enumerate a tree...
_WALKS = {"walk", "rglob", "iglob", "glob"}
#: ...carry the files it does not rewrite across verbatim...
_MIRRORS = {"copy", "copy2", "copyfile", "copytree", "move"}
#: ...and write the ones it does.
_TEXT_WRITES = {"write_text", "write_bytes"}
#: Native sources are what a port exists to rewrite.
_NATIVE_SUFFIXES = {".cc", ".cu", ".cuh", ".h"}

#: Field names on the provider contract that would hand a provider the tree.
_REWRITING_SERVICE = re.compile(
    r"(transform|convert|rewrite|translate|port)_?sources?"
    r"|sources?_?(transform|convert|rewrite|translate|port)"
    r"|process_(jittor_)?source",
    re.I,
)


def _production_modules():
    for base in PRODUCTION:
        for path in sorted((ROOT / base).rglob("*.py")):
            relative = path.relative_to(ROOT)
            if "tests" in relative.parts or path.name.startswith("test_"):
                continue
            yield relative, ast.parse(path.read_text(encoding="utf8"),
                                      filename=str(path))


def _called(node):
    if not isinstance(node, ast.Call):
        return ""
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return getattr(node.func, "id", "")


def _opens_for_writing(node):
    if _called(node) != "open":
        return False
    modes = list(node.args[1:2]) + [key.value for key in node.keywords
                                    if key.arg == "mode"]
    return any(isinstance(mode, ast.Constant)
               and isinstance(mode.value, str)
               and set("wax") & set(mode.value) for mode in modes)


def _port_signals(function):
    """Which marks of a mirror-and-rewrite port this function carries."""
    signals = set()
    for node in ast.walk(function):
        name = _called(node)
        if name in _WALKS:
            signals.add("enumerates a tree")
        if name in _MIRRORS:
            signals.add("copies files verbatim")
        if name in _TEXT_WRITES or _opens_for_writing(node):
            signals.add("writes file contents")
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value in _NATIVE_SUFFIXES):
            signals.add("selects native sources")
    return signals


def test_the_provider_contract_offers_no_source_rewriting_service():
    """``BuildContext`` is the public surface a backend provider is handed.

    While it carried ``transform_sources``, "rewrite the shared tree" was a
    documented, supported thing for a provider to do.  Removing the field is
    what makes the port unavailable rather than merely unused.
    """
    source = (ROOT / "python/jittor/build/utils/build_config.py").read_text(encoding="utf8")
    context = next(node for node in ast.parse(source).body
                   if isinstance(node, ast.ClassDef) and node.name == "BuildContext")
    fields = [node.target.id for node in context.body
              if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)]
    offered = [name for name in fields if _REWRITING_SERVICE.search(name)]
    assert offered == [], (
        "BuildContext hands providers a source-rewriting service again: %s. "
        "A backend that needs different code registers its own implementation."
        % offered)
    assert "compile_module" in fields, "the field scan stopped matching anything"


def test_no_production_code_mirrors_a_tree_while_rewriting_native_sources():
    """The port shape, matched by shape rather than by name.

    All four marks together are what a port is.  Three are not enough and must
    not be: the pyjt binding generator enumerates ``.cc`` files and writes
    generated ones, but it copies nothing across, because it produces new
    artifacts instead of a rewritten duplicate of the tree.
    """
    ports = []
    for relative, tree in _production_modules():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            signals = _port_signals(node)
            if len(signals) == 4:
                ports.append("%s:%d %s()" % (relative, node.lineno, node.name))
    assert ports == [], (
        "these functions copy a source tree and rewrite the native files in "
        "it, which is what 4.12 removed: %s" % ports)


def test_nothing_redirects_the_build_source_root_to_a_derived_copy():
    """``jittor_path`` is established once and never re-pointed.

    The final move of the old port was ``config.evolve(jittor_path=<copy>)``:
    every later compile then read the rewritten duplicate instead of the
    checkout.  Reading the field stays free; binding it is the bootstrap's.
    """
    redirects = []
    for relative, tree in _production_modules():
        if str(relative) == SOURCE_ROOT_OWNER:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for key in node.keywords:
                    if key.arg == "jittor_path":
                        redirects.append("%s:%d %s(jittor_path=...)"
                                         % (relative, node.lineno, _called(node)))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr == "jittor_path":
                        redirects.append("%s:%d assigns .jittor_path"
                                         % (relative, node.lineno))
    assert redirects == [], (
        "only the bootstrap in %s may say where the sources are; these "
        "re-point it: %s" % (SOURCE_ROOT_OWNER, redirects))
