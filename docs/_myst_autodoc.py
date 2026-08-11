"""Bridge Sphinx's RST-emitting autodoc directives into MyST documents."""

from __future__ import print_function

from docutils import nodes
from docutils.parsers.rst import directives
import importlib
import inspect
import json
from myst_parser.mdit_to_docutils.base import make_document
from myst_parser.mocking import MockRSTParser
from pathlib import Path
from sphinx.ext.autodoc.directive import DummyOptionSpec
from sphinx.util.docutils import SphinxDirective


_AUTODOC_NAMES = ("automodule", "autoclass", "autodata", "autofunction")
_ALIAS_PREFIX = "jittor-rst-"


def _directive_alias(name, directive_class):
    class RstDirectiveAlias(directive_class):
        def run(self):
            self.name = name
            return super(RstDirectiveAlias, self).run()

    RstDirectiveAlias.__name__ = "Rst{}Alias".format(name.title())
    return RstDirectiveAlias


class RstDirectiveBridge(SphinxDirective):
    """Reconstruct one directive and parse it with MyST's isolated RST parser."""

    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    has_content = True
    option_spec = DummyOptionSpec()

    def _source(self):
        argument = " " + self.arguments[0] if self.arguments else ""
        lines = [".. {}{}::{}".format(_ALIAS_PREFIX, self.name, argument)]
        for key, value in self.options.items():
            option = "   :{}:".format(key)
            if value is not None:
                option += " {}".format(value)
            lines.append(option)
        if self.content:
            lines.append("")
            lines.extend("   " + line for line in self.content)
        return "\n".join(lines) + "\n"

    def run(self):
        document = self.state.document
        nested = make_document()
        nested["source"] = document["source"]
        nested.settings = document.settings
        nested.reporter = document.reporter
        source = "\n" * max(self.lineno - 1, 0) + self._source()
        MockRSTParser().parse(source, nested)
        for node in nested:
            if node.get("names"):
                document.note_explicit_target(node, node)
        return list(nested.children)


def _public_members(module_name):
    module = importlib.import_module(module_name)
    declared = getattr(module, "__all__", None)
    names = declared if declared is not None else vars(module)
    members = []
    for name in names:
        if declared is None and name.startswith("_"):
            continue
        value = getattr(module, name)
        origin = getattr(value, "__module__", "") or ""
        type_origin = getattr(type(value), "__module__", "") or ""
        module_origin = getattr(value, "__name__", "") if inspect.ismodule(value) else ""
        jittor_owned = any(
            candidate in ("jittor", "jittor_core")
            or candidate.startswith(("jittor.", "jittor_core."))
            for candidate in (origin, type_origin, module_origin)
        )
        if declared is None and not jittor_owned:
            continue
        if inspect.isclass(value):
            directive = "autoclass"
        elif inspect.isroutine(value):
            directive = "autofunction"
        else:
            directive = "autodata"
        members.append((name, value, directive))
    return sorted(members, key=lambda item: (item[0].lower(), item[0]))


class PublicAutomoduleBridge(RstDirectiveBridge):
    """Render stable API objects richly and every other public export literally."""

    required_arguments = 1

    def _rich_objects(self, module_name):
        inventory_path = Path(self.env.srcdir) / "api" / "inventory.json"
        self.env.note_dependency(str(inventory_path))
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        for page in inventory["pages"]:
            if page["docname"] == self.env.docname:
                return {
                    name for name in page["objects"] if name.rsplit(".", 1)[0] == module_name
                }
        raise RuntimeError("API inventory has no page for {}".format(self.env.docname))

    def _source(self):
        module_name = self.arguments[0]
        members = _public_members(module_name)
        if not members:
            raise RuntimeError("no public API members found for {}".format(module_name))
        rich_objects = self._rich_objects(module_name)
        discovered = {module_name + "." + member for member, _value, _directive in members}
        missing = sorted(rich_objects - discovered)
        if missing:
            raise RuntimeError("stable API objects are not publicly exported: {}".format(missing))
        lines = [".. py:module:: {}".format(module_name), ""]
        for member, _value, directive in members:
            qualified = module_name + "." + member
            if qualified not in rich_objects:
                continue
            lines.append(
                ".. {}{}:: {}".format(_ALIAS_PREFIX, directive, qualified)
            )
            lines.append("")
        return "\n".join(lines) + "\n"

    def run(self):
        module_name = self.arguments[0]
        rich_objects = self._rich_objects(module_name)
        result = super(PublicAutomoduleBridge, self).run()
        for member, value, _directive in _public_members(module_name):
            qualified = module_name + "." + member
            if qualified in rich_objects:
                continue
            signature = ""
            if callable(value):
                try:
                    signature = str(inspect.signature(value))
                except (TypeError, ValueError):
                    signature = "(...)"
            container = nodes.container(classes=["api-public-export"])
            container["ids"].append("public-api-" + qualified)
            heading = nodes.paragraph()
            heading += nodes.literal(text=qualified + signature)
            heading["translatable"] = False
            container += heading
            docstring = inspect.getdoc(value)
            if docstring:
                literal = nodes.literal_block(text=docstring)
                literal["language"] = "text"
                # Legacy raw docs stay in their canonical language until promoted
                # into the machine-readable stable API inventory.
                literal["translatable"] = False
                container += literal
            result.append(container)
        return result


def _register_bridges(app, config):
    del app, config
    for name in _AUTODOC_NAMES + ("autosummary",):
        directive_class = directives._directives.get(name)
        if directive_class is None:
            raise RuntimeError("Sphinx did not register the {} directive".format(name))
        alias = _directive_alias(name, directive_class)
        directives.register_directive(_ALIAS_PREFIX + name, alias)
        directives.register_directive(name, RstDirectiveBridge)
    directives.register_directive("autopublicmodule", PublicAutomoduleBridge)


def setup(app):
    app.connect("config-inited", _register_bridges, priority=900)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
