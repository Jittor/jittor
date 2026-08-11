"""Contracts for the tracked MyST/Jupytext notebook pairs."""

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
    "example",
    "meta_op",
    "profiler",
    "60分钟快速入门Jittor/计图入门教程 0 --- 介绍与安装",
    "60分钟快速入门Jittor/计图入门教程 1 --- 基本概念",
    "60分钟快速入门Jittor/计图入门教程 2 --- 如何训练一个简单线性回归",
    "60分钟快速入门Jittor/计图入门教程 3 --- 尝试解决一个实际问题",
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


def _topic_path(topic, suffix):
    return _notebook_root() / (topic + suffix)


def _load_notebook(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_notebook(path, nbformat):
    notebook = nbformat.read(path, as_version=4)
    return {
        "kernelspec": dict(notebook.metadata.get("kernelspec", {})),
        "formats": notebook.metadata.get("jupytext", {}).get("formats"),
        "cells": [
            {
                "cell_type": cell.cell_type,
                "source": cell.source,
                "tags": list(cell.metadata.get("tags", [])),
                "execution_count": cell.get("execution_count"),
                "outputs": json.loads(json.dumps(cell.get("outputs", []))),
            }
            for cell in notebook.cells
        ],
    }


@pytest.mark.structure
def test_notebook_pairs_are_complete_clean_and_portable():
    root = _notebook_root()
    expected = set(_TOPICS)
    markdown_topics = {
        path.relative_to(root).with_suffix("").as_posix()
        for path in root.rglob("*.md")
        if path.name != "README.md"
    }
    notebook_topics = {
        path.relative_to(root).with_suffix("").as_posix() for path in root.rglob("*.ipynb")
    }
    assert markdown_topics == expected
    assert notebook_topics == expected
    assert not list(root.rglob("*.src.md"))
    assert not (_repo_root() / "python" / "jittor" / "notebook").exists()

    for topic in _TOPICS:
        markdown_path = _topic_path(topic, ".md")
        notebook_path = _topic_path(topic, ".ipynb")
        markdown = markdown_path.read_text(encoding="utf-8")
        notebook_text = notebook_path.read_text(encoding="utf-8")
        notebook = json.loads(notebook_text)

        assert "formats: md:myst,ipynb" in markdown
        assert notebook["metadata"]["jupytext"]["formats"] == "md:myst,ipynb"
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        for cell in notebook["cells"]:
            assert not cell.get("attachments")
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs", []) == []
        for marker in _MACHINE_PATHS:
            assert marker not in markdown
            assert marker not in notebook_text

    assert (root / "figs" / "mop.svg").is_file()
    assert (root / "60分钟快速入门Jittor" / "mnist.png").is_file()
    assert (root / "60分钟快速入门Jittor" / "jittor-star.png").is_file()


@pytest.mark.structure
def test_myst_fences_are_explicit_and_balanced():
    opener = re.compile(r"^```\{(?:code-cell|code-block)\}(?: ipython3| [\w+-]+)$")
    for topic in _TOPICS:
        path = _topic_path(topic, ".md")
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
        notebook = _load_notebook(_topic_path(topic, ".ipynb"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for cell in code_cells:
            tags = set(cell.get("metadata", {}).get("tags", []))
            all_tags.update(tags)
            if tags.intersection({"network", "cuda", "long-running"}):
                assert "skip-execution" in tags
        if topic in {"ConditionGAN", "LSGAN"}:
            assert code_cells
            assert all(
                {"gan", "skip-execution"}.issubset(set(cell.get("metadata", {}).get("tags", [])))
                for cell in code_cells
            )

    assert {"gan", "network", "cuda", "long-running", "skip-execution"}.issubset(all_tags)
    basics = _load_notebook(_topic_path("basics", ".ipynb"))
    basics_code = [cell for cell in basics["cells"] if cell["cell_type"] == "code"]
    assert basics_code
    assert all(not cell.get("metadata", {}).get("tags") for cell in basics_code)


@pytest.mark.structure
def test_jupytext_sync_reproduces_committed_pairs(tmp_path):
    jupytext = pytest.importorskip("jupytext")
    nbformat = pytest.importorskip("nbformat")
    del jupytext

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for topic in _TOPICS:
        source = _topic_path(topic, ".md")
        temporary_source = tmp_path / (topic + ".md")
        temporary_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, temporary_source)
        result = subprocess.run(
            [sys.executable, "-m", "jupytext", "--sync", str(temporary_source)],
            cwd=str(tmp_path),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert result.returncode == 0, result.stdout
        generated = temporary_source.with_suffix(".ipynb")
        assert generated.is_file()
        assert _semantic_notebook(generated, nbformat) == _semantic_notebook(
            _topic_path(topic, ".ipynb"), nbformat
        )


@pytest.mark.cpu
@pytest.mark.slow
def test_basics_notebook_executes_offline_on_cpu(tmp_path, monkeypatch):
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
    python_config = shutil.which("python3.{}-config".format(sys.version_info[1]))
    if python_config:
        monkeypatch.setenv("python_config_path", python_config)
    monkeypatch.setenv("JITTOR_HOME", str(tmp_path / "jittor-cache"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    offline_guard = """import socket
_original_connect = socket.socket.connect
_original_create_connection = socket.create_connection

def _is_local(address):
    return isinstance(address, tuple) and address[0] in (\"127.0.0.1\", \"::1\", \"localhost\")

def _offline_connect(sock, address):
    if not _is_local(address):
        raise RuntimeError(\"network access is disabled for the notebook smoke test\")
    return _original_connect(sock, address)

def _offline_create_connection(address, *args, **kwargs):
    if not _is_local(address):
        raise RuntimeError(\"network access is disabled for the notebook smoke test\")
    return _original_create_connection(address, *args, **kwargs)

socket.socket.connect = _offline_connect
socket.create_connection = _offline_create_connection
import jittor as jt
jt.flags.use_cuda = 0
"""

    try:
        import nbclient
        import nbformat
    except ImportError:
        runner = (
            offline_guard
            + """
import json
import sys

notebook_path = sys.argv[1]
with open(notebook_path, encoding=\"utf-8\") as stream:
    notebook = json.load(stream)
namespace = {\"__name__\": \"__main__\"}
for index, cell in enumerate(notebook[\"cells\"]):
    if cell[\"cell_type\"] == \"code\":
        filename = \"{}#cell-{}\".format(notebook_path, index)
        source = cell[\"source\"]
        if isinstance(source, list):
            source = \"\".join(source)
        exec(compile(source, filename, \"exec\"), namespace)
"""
        )
        result = subprocess.run(
            [sys.executable, "-c", runner, str(_topic_path("basics", ".ipynb"))],
            cwd=str(tmp_path),
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
        assert result.returncode == 0, result.stdout
    else:
        notebook = nbformat.read(_topic_path("basics", ".ipynb"), as_version=4)
        notebook.cells.insert(0, nbformat.v4.new_code_cell(offline_guard))
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
