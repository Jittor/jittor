from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_memory_diagnostics_load_atomic_counters_before_streaming():
    source = (ROOT / "src" / "mem" / "mem_info.cc").read_text(encoding="utf-8")

    assert '<< "lived_vars:" << Var::number_of_lived_vars.load()' in source
    assert '<< "lived_ops:" << Op::number_of_lived_ops.load()' in source


def test_exec_plan_check_loads_atomic_counter_before_formatting_failure():
    source = (
        ROOT / "src" / "tests" / "test_exec_plan_properties.cc"
    ).read_text(encoding="utf-8")

    assert "CHECKop(Op::number_of_lived_ops.load(),==,ops_before);" in source
