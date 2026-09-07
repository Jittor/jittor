from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_executor_var_holder_dependency_is_one_way():
    var_holder_h = (ROOT / "src/core/var_holder.h").read_text()
    var_holder_cc = (ROOT / "src/core/var_holder.cc").read_text()
    executor_cc = (ROOT / "src/core/executor.cc").read_text()

    assert '#include "core/executor.h"' not in var_holder_h
    assert "exe." not in var_holder_h
    assert '#include "core/executor.h"' in var_holder_cc
    assert '#include "core/var_holder.h"' in executor_cc

    for method in ("migrate_to_cpu_", "data", "raw_ptr", "set_data"):
        assert "VarHolder::" + method in var_holder_cc
