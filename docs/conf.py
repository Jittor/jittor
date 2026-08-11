"""Sphinx configuration for the canonical Jittor documentation tree."""

try:
    from importlib.metadata import version
except ImportError:  # pragma: no cover - documentation tooling uses Python 3.11
    from importlib_metadata import version
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


project = "Jittor"
copyright = "2026, Jittor contributors"
author = "Jittor contributors"
release = version("jittor")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
]

source_suffix = {".md": "markdown"}
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "substitution"]
autosummary_generate = False
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autosectionlabel_prefix_document = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pytorch": ("https://pytorch.org/docs/stable/", None),
}
intersphinx_cache_limit = 7
intersphinx_timeout = 15

locale_dirs = ["locales/"]
gettext_compact = False
gettext_uuid = True
gettext_location = False

nitpicky = True
nitpick_ignore = [
    ("py:class", "None."),
    ("py:class", "callable"),
    ("py:class", "compute_uv"),
    ("py:class", "dim"),
    ("py:class", "jittor array"),
    ("py:class", "jittor type-cast function"),
    ("py:class", "jittor.Var."),
    ("py:class", "jittor.dataset.dataset.VarDataset"),
    ("py:class", "jittor.jittor_core.Var"),
    ("py:class", "keepdim"),
    ("py:class", "number"),
    ("py:class", "optional"),
    ("py:class", "string"),
    ("py:func", "matrix_norm"),
    ("py:func", "svdvals"),
    ("py:func", "vector_norm"),
]

html_theme = "furo"
html_title = "Jittor documentation"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/Jittor/jittor/",
    "source_branch": "master",
    "source_directory": "docs/",
}


def _sanitize_autodoc_docstring(app, what, name, obj, options, lines):
    del app, what, obj, options
    if name == "jittor.array":
        lines[:] = [line for line in lines if line.strip() != "----------------"]
    elif name == "jittor.linalg.det":
        lines[:] = [line.replace("|x|", r"\|x\|") for line in lines]
    elif name == "jittor.transform.ColorJitter":
        lines[:] = [
            "Randomly change the brightness, contrast, saturation, and hue of an image.",
            "",
            "Each parameter accepts either a non-negative scalar or a two-value range.",
        ]

def setup(app):
    adapter_path = Path(__file__).with_name("_myst_autodoc.py")
    spec = spec_from_file_location("_jittor_myst_autodoc", str(adapter_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load MyST autodoc adapter: {}".format(adapter_path))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    metadata = module.setup(app)
    app.connect("autodoc-process-docstring", _sanitize_autodoc_docstring)
    return metadata
