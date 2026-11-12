"""Environment startup hooks run once in each fresh Python interpreter."""
import os
from pathlib import Path

from _helpers.child_process import run_child_script
from jittor.compat.shim.build import _write_build_sitecustomize


def _sites(tmp_path):
    shim, hook = tmp_path / 'shim', tmp_path / 'hook'
    shim.mkdir()
    hook.mkdir()
    _write_build_sitecustomize(shim)
    return shim, hook


def test_environment_hook_runs_in_parent_and_spawned_child(tmp_path):
    shim, hook = _sites(tmp_path)
    (hook / 'sitecustomize.py').write_text(
        'import os\nprint("ENV_HOOK=" + str(os.getpid()), flush=True)\n')
    tests = Path(__file__).resolve().parents[3] / 'tests'
    script = '''
import sys
sys.path.insert(0, %r)
from _helpers.child_process import run_child_script
result = run_child_script('print("CHILD_READY")', text=True)
print(result.stdout)
assert result.returncode == 0, result.stderr
''' % str(tests)
    result = run_child_script(script, env={'PYTHONPATH': os.pathsep.join((str(shim), str(hook)))},
                              text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('ENV_HOOK=') == 2, result.stdout
    assert 'CHILD_READY' in result.stdout


def test_reentrant_hook_runs_once_and_does_not_export_a_disable_flag(tmp_path):
    shim, hook = _sites(tmp_path)
    (hook / 'sitecustomize.py').write_text(
        'import runpy\nprint("ENV_HOOK")\nrunpy.run_path(%r)\n' % str(shim / 'sitecustomize.py'))
    script = '''
import os, runpy
runpy.run_path(%r)
assert '_JT_ENV_SITECUSTOMIZE_CHAIN' not in os.environ
''' % str(shim / 'sitecustomize.py')
    result = run_child_script(script, env={'PYTHONPATH': os.pathsep.join((str(shim), str(hook)))},
                              text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('ENV_HOOK') == 1, result.stdout
