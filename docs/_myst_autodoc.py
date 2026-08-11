"""Bridge Sphinx's RST-emitting autodoc directives into MyST documents."""

from __future__ import print_function

from docutils.parsers.rst import directives
from myst_parser.mdit_to_docutils.base import make_document
from myst_parser.mocking import MockRSTParser
from sphinx.ext.autodoc.directive import DummyOptionSpec
from sphinx.util.docutils import SphinxDirective


_AUTODOC_NAMES = ("automodule", "autoclass", "autofunction")
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


def _register_bridges(app, config):
    del app, config
    for name in _AUTODOC_NAMES + ("autosummary",):
        directive_class = directives._directives.get(name)
        if directive_class is None:
            raise RuntimeError("Sphinx did not register the {} directive".format(name))
        alias = _directive_alias(name, directive_class)
        directives.register_directive(_ALIAS_PREFIX + name, alias)
        directives.register_directive(name, RstDirectiveBridge)


def setup(app):
    app.connect("config-inited", _register_bridges, priority=900)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
