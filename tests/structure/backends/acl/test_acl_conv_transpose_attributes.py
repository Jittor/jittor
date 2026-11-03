"""Forward convolution mode defaults preserve existing data-channel callers."""

from _helpers.acl_recording_pipeline import build_pipeline


def test_omitted_transpose_mode_matches_false_and_true_has_distinct_data(monkeypatch):
    load, _, _ = build_pipeline(monkeypatch)
    attributes = load("_attributes")
    data = attributes.attribute_data
    original = {
        "convStrides": [2, 2], "convPads": [1, 1], "convDilations": [1, 1],
        "group": 1, "convOutPads": [0, 0], "cube_math_type": 1,
    }
    omitted = data("Conv2d", original)
    assert omitted == data("Conv2d", dict(original, is_transposed=False))
    assert omitted != data("Conv2d", dict(original, is_transposed=True))
    # Transpose is an inference-only extension of the forward schema. Existing
    # backward programs keep their original schema and cannot silently opt in.
    assert "is_transposed" not in attributes.SCHEMAS["Conv2dBackward"]
