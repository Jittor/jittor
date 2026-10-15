"""A real Git pathspec regression; no tensor runtime or downstream imports."""

import hashlib
import importlib.util
from pathlib import Path
import subprocess


def test_runtime_fingerprint_includes_root_and_nested_python(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = root / 'compat/tests/torch/_ecosystem_trellis_runner.py'
    spec = importlib.util.spec_from_file_location('trellis_source_fingerprint', script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    subprocess.run(['git', 'init', str(tmp_path)], check=True, capture_output=True)
    files = {'trellis2/__init__.py': b'ROOT = True\n',
             'trellis2/models/model.py': b'class Model: pass\n',
             'trellis2/README.md': b'not executable source\n'}
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    subprocess.run(['git', '-C', str(tmp_path), 'add', *files], check=True, capture_output=True)
    actual = runner._source_hashes(tmp_path)
    assert actual == {relative: hashlib.sha256(content).hexdigest()
                      for relative, content in files.items() if relative.endswith('.py')}
