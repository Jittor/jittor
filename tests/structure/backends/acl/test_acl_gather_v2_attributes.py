"""Single-axis GatherV2 has its own schema instead of changing Gather."""
from _helpers.acl_recording_pipeline import build_pipeline


def test_gather_v2_dimension_uses_standard_typed_data(monkeypatch):
    load, Tensor, calls = build_pipeline(monkeypatch)
    program = load("_code").acl_program("GatherV2", 2, 1, attributes={"dim": 1})
    output = load("_code").acl_emit(program, [Tensor((2, 7, 3), "float16"), Tensor((4,), "int32")],
                                   ["float16"], [(2, 4, 3)])[0]
    assert output.shape == (2, 4, 3) and output.dtype == "float16"
    assert calls[-1]["data"] == load("_attributes").attribute_data("GatherV2", {"dim": 1})
    assert "GatherV2OpRunner" in calls[-1]["cuda_src"]
    assert not calls[-1].get("cuda_grad_src")
    assert load("_attributes").runner_for_alias("gather") == "Gather"
    assert load("_attributes").runner_for_alias("gather_v2") == "GatherV2"
