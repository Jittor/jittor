"""A compile_module cache must track headers of the generated extension TU."""

import json
import os

from _helpers.child_process import run_child_script


def test_compile_module_rebuilds_after_only_an_included_header_changes(tmp_path):
    header = tmp_path / "module_dependency.h"
    header.write_text("#pragma once\n#define MODULE_DEPENDENCY_VALUE 17\n", encoding="utf8")
    initial_stat = header.stat()
    source = """
    #include "common.h"
    #include HEADER_PATH
    namespace jittor {
    // @pyjt(module_dependency_value)
    int module_dependency_value() { return MODULE_DEPENDENCY_VALUE; }
    }
    """.replace("HEADER_PATH", json.dumps(header.as_posix()))
    child_source = """
import json
import jittor as jt
import jittor_utils
module = jittor_utils.compile_module(SOURCE, jt.compiler.cc_flags)
print("MODULE_DEPENDENCY_RESULT=" + json.dumps({
    "value": module.module_dependency_value(),
    "name": module.__name__,
    "path": module.__file__,
}))
""".replace("SOURCE", repr(source))

    def execute():
        result = run_child_script(child_source, directory=tmp_path,
                                  name="module_dependency", timeout=180,
                                  without_torch_mode=True, text=True)
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        rows = [line.partition("=")[2]
                for line in result.stdout.splitlines()
                if line.startswith("MODULE_DEPENDENCY_RESULT=")]
        assert len(rows) == 1, result.stdout
        return json.loads(rows[0])

    first = execute()
    assert first["value"] == 17
    header.write_text("#pragma once\n#define MODULE_DEPENDENCY_VALUE 29\n", encoding="utf8")
    # The dependency cache promises content hashing, not size/mtime heuristics.
    os.utime(header, ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns))
    second = execute()
    assert second["name"] == first["name"]
    assert second["path"] == first["path"]
    assert second["value"] == 29
