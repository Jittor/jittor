from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_executor_var_holder_dependency_is_one_way():
    var_holder_h = (ROOT / "python/jittor/src/var_holder.h").read_text()
    var_holder_cc = (ROOT / "python/jittor/src/var_holder.cc").read_text()
    executor_cc = (ROOT / "python/jittor/src/executor.cc").read_text()

    assert '#include "executor.h"' not in var_holder_h
    assert "exe." not in var_holder_h
    assert '#include "executor.h"' in var_holder_cc
    assert '#include "var_holder.h"' in executor_cc

    for method in ("migrate_to_cpu_", "data", "raw_ptr", "set_data"):
        assert "VarHolder::" + method in var_holder_cc
