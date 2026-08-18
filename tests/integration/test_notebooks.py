"""Contracts for canonical MyST/Jupytext notebook sources."""

from __future__ import print_function

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


_TOPICS = (
    "ConditionGAN",
    "LSGAN",
    "basics",
    "custom_op",
    "diffusion",
    "example",
    "lora",
    "meta_op",
    "profiler",
    "transformer",
    "60分钟快速入门Jittor/计图入门教程 0 --- 介绍与安装",
    "60分钟快速入门Jittor/计图入门教程 1 --- 基本概念",
    "60分钟快速入门Jittor/计图入门教程 2 --- 如何训练一个简单线性回归",
    "60分钟快速入门Jittor/计图入门教程 3 --- 尝试解决一个实际问题",
)

_SMOKE_TOPICS = (
    "basics", "example", "meta_op", "custom_op", "profiler",
    "transformer", "diffusion", "lora",
)

_MACHINE_PATHS = (
    "/home/",
    "/Users/",
    "site-packages",
    ".cache/jittor",
    "python/jittor/notebook",
    "jittor/notebook",
)


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _notebook_root():
    return _repo_root() / "examples" / "notebooks"


def _topic_path(topic):
    return _notebook_root() / (topic + ".md")


def _markdown_cells(path):
    cells = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].startswith("```{code-cell}"):
            index += 1
            continue
        tags = set()
        source = []
        index += 1
        while index < len(lines) and lines[index] != "```":
            line = lines[index]
            match = re.match(r"^:tags:\s*\[([^]]*)\]\s*$", line)
            if match:
                tags.update(tag.strip() for tag in match.group(1).split(",") if tag.strip())
            elif not line.startswith(":"):
                source.append(line)
            index += 1
        cells.append({"tags": tags, "source": "\n".join(source)})
        index += 1
    return cells


def _materialize(topic, destination):
    pytest.importorskip("jupytext")
    output = destination / (topic + ".ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "jupytext",
            "--to",
            "ipynb",
            "--output",
            str(output),
            str(_topic_path(topic)),
        ),
        cwd=str(destination),
        env=os.environ.copy(),
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert output.is_file()
    return output


@pytest.mark.structure
def test_notebook_sources_are_complete_clean_and_portable():
    root = _notebook_root()
    expected = set(_TOPICS)
    markdown_topics = {
        path.relative_to(root).with_suffix("").as_posix()
        for path in root.rglob("*.md")
        if path.name != "README.md"
    }
    assert markdown_topics == expected
    assert not list(root.rglob("*.ipynb"))
    assert not list(root.rglob("*.src.md"))
    assert not (_repo_root() / "python" / "jittor" / "notebook").exists()

    tracked = subprocess.run(
        ("git", "ls-files", "--", "*.ipynb"),
        cwd=str(_repo_root()),
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert tracked.stdout == ""

    for topic in _TOPICS:
        markdown_path = _topic_path(topic)
        markdown = markdown_path.read_text(encoding="utf-8")
        assert "formats: md:myst,ipynb" in markdown
        assert "name: python3" in markdown
        for marker in _MACHINE_PATHS:
            assert marker not in markdown

    assert (root / "figs" / "mop.svg").is_file()
    assert (root / "60分钟快速入门Jittor" / "mnist.png").is_file()
    assert (root / "60分钟快速入门Jittor" / "jittor-star.png").is_file()


@pytest.mark.structure
def test_myst_fences_are_explicit_and_balanced():
    opener = re.compile(r"^```\{(?:code-cell|code-block)\}(?: ipython3| [\w+-]+)$")
    for topic in _TOPICS:
        path = _topic_path(topic)
        inside_fence = False
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not inside_fence and line.startswith("```"):
                assert opener.fullmatch(line), "{}:{}: {}".format(path, line_number, line)
                inside_fence = True
            elif inside_fence and line == "```":
                inside_fence = False
        assert not inside_fence, "unterminated fence in {}".format(path)


@pytest.mark.structure
def test_expensive_and_external_cells_are_tagged():
    all_tags = set()
    for topic in _TOPICS:
        code_cells = _markdown_cells(_topic_path(topic))
        for cell in code_cells:
            tags = cell["tags"]
            all_tags.update(tags)
            if tags.intersection({"network", "cuda", "long-running"}):
                assert "skip-execution" in tags
        if topic in {"ConditionGAN", "LSGAN"}:
            assert code_cells
            assert all({"gan", "skip-execution"}.issubset(cell["tags"]) for cell in code_cells)

    assert {"gan", "network", "cuda", "long-running", "skip-execution"}.issubset(all_tags)
    assert _markdown_cells(_topic_path("basics"))
    assert all(not cell["tags"] for cell in _markdown_cells(_topic_path("basics")))
    for topic in _SMOKE_TOPICS:
        assert any(
            "skip-execution" not in cell["tags"] for cell in _markdown_cells(_topic_path(topic))
        )


@pytest.mark.structure
def test_jupytext_materializes_clean_notebooks_outside_the_checkout(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    for topic in _TOPICS:
        generated = _materialize(topic, tmp_path)
        notebook = nbformat.read(generated, as_version=4)
        assert notebook.metadata["jupytext"]["formats"] == "md:myst,ipynb"
        assert notebook.metadata["kernelspec"]["name"] == "python3"
        assert len([cell for cell in notebook.cells if cell.cell_type == "code"]) == len(
            _markdown_cells(_topic_path(topic))
        )
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs", []) == []
    assert not list(_notebook_root().rglob("*.ipynb"))


def _offline_guard():
    return r"""import socket
_original_connect = socket.socket.connect
_original_create_connection = socket.create_connection

def _is_local(address):
    return isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1", "localhost")

def _offline_connect(sock, address):
    if not _is_local(address):
        raise RuntimeError("network access is disabled for the notebook smoke test")
    return _original_connect(sock, address)

def _offline_create_connection(address, *args, **kwargs):
    if not _is_local(address):
        raise RuntimeError("network access is disabled for the notebook smoke test")
    return _original_create_connection(address, *args, **kwargs)

socket.socket.connect = _offline_connect
socket.create_connection = _offline_create_connection
import jittor as jt
jt.flags.use_cuda = 0
"""


@pytest.mark.cpu
def test_notebook_smokes_execute_offline_on_cpu(tmp_path, monkeypatch):
    for relative in ("home", "jittor-cache", "xdg-cache"):
        (tmp_path / relative).mkdir()
    python_path = str(_repo_root() / "python")
    if os.environ.get("PYTHONPATH"):
        python_path += os.pathsep + os.environ["PYTHONPATH"]
    monkeypatch.setenv("PYTHONPATH", python_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("nvcc_path", "")
    monkeypatch.setenv("use_cuda", "0")
    monkeypatch.setenv("use_nccl", "0")
    monkeypatch.setenv("use_mpi", "0")
    monkeypatch.setenv("use_mkl", "0")
    monkeypatch.setenv("use_cutt", "0")
    monkeypatch.setenv("use_cutlass", "0")
    # A Jupyter kernel is a threaded process, and Jittor's parallel operator
    # compiler forks from it during the first cell. See KI-COMPILER-001.
    monkeypatch.setenv("use_parallel_op_compiler", "0")
    python_config = shutil.which("python3.{}-config".format(sys.version_info[1]))
    if python_config:
        monkeypatch.setenv("python_config_path", python_config)
    monkeypatch.setenv("JITTOR_HOME", str(tmp_path / "jittor-cache"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    try:
        import nbclient
        import nbformat
    except ImportError:
        nbclient = None
        nbformat = None

    executed_topics = []
    for topic in _SMOKE_TOPICS:
        generated = _materialize(topic, tmp_path / "generated")
        if nbformat is None:
            notebook = json.loads(generated.read_text(encoding="utf-8"))
            namespace = {"__name__": "__main__"}
            exec(compile(_offline_guard(), "offline_guard.py", "exec"), namespace)
            eligible = [
                cell
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
                and "skip-execution" not in cell.get("metadata", {}).get("tags", [])
            ]
            assert eligible
            for index, cell in enumerate(eligible):
                source = cell["source"]
                if isinstance(source, list):
                    source = "".join(source)
                exec(compile(source, "{}#cell-{}".format(generated, index), "exec"), namespace)
        else:
            notebook = nbformat.read(generated, as_version=4)
            notebook.cells = [
                cell
                for cell in notebook.cells
                if cell.cell_type != "code"
                or "skip-execution" not in cell.get("metadata", {}).get("tags", [])
            ]
            eligible = [cell for cell in notebook.cells if cell.cell_type == "code"]
            assert eligible
            notebook.cells.insert(0, nbformat.v4.new_code_cell(_offline_guard()))
            client = nbclient.NotebookClient(
                notebook,
                timeout=600,
                kernel_name="python3",
                resources={"metadata": {"path": str(tmp_path)}},
            )
            executed = client.execute()
            assert all(
                cell.get("execution_count") is not None
                for cell in executed.cells
                if cell.cell_type == "code"
            )
        executed_topics.append(topic)

    assert tuple(executed_topics) == _SMOKE_TOPICS
