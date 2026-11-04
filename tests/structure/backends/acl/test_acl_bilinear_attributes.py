"""Bilinear alignment reaches the typed data channel without source changes."""

from _helpers.acl_recording_pipeline import build_pipeline


def test_bilinear_alignment_and_geometry_are_runtime_data(monkeypatch):
    load, Tensor, calls = build_pipeline(monkeypatch)
    module = load("upsample_op")
    monkeypatch.setattr(module.jt.flags, "no_grad", 1)
    x = Tensor((1, 2, 3, 4), "float16")
    for alignment in (False, True, False):
        output = module.UpsampleBilinear2dACL()(x, (5, 7), alignment)
        assert output.shape == (1, 2, 5, 7) and output.dtype == "float16"
    assert calls[0]["cuda_src"] == calls[1]["cuda_src"] == calls[2]["cuda_src"]
    assert calls[0]["data"] == calls[2]["data"] != calls[1]["data"]
    assert calls[1]["data"] == load("_attributes").attribute_data("UpsampleBilinear2d", {
        "outputSize": [5, 7], "inputSize": [1, 2, 3, 4], "alignCorners": True,
    }, prefix="acl_payload.UpsampleBilinear2d_op.")
    assert all(call["backend"] == "acl" and not call.get("cuda_grad_src") for call in calls)
