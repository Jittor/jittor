"""Missing optional libraries must not become a successful required gate."""
from _helpers.pytest_policy import _environment_explains


def test_optional_deepspeed_absence_is_visible_environment_skip(monkeypatch):
    monkeypatch.delenv("JITTOR_REQUIRE_DEEPSPEED", raising=False)
    assert _environment_explains(["deepspeed optional dependency is not installed"])


def test_required_deepspeed_cannot_explain_empty_execution(monkeypatch):
    monkeypatch.setenv("JITTOR_REQUIRE_DEEPSPEED", "1")
    assert not _environment_explains(["deepspeed optional dependency is not installed"])


def test_required_oracle_stays_required_when_deepspeed_is_optional(monkeypatch):
    monkeypatch.setenv("JITTOR_REQUIRE_REAL_TORCH", "1")
    monkeypatch.delenv("JITTOR_REQUIRE_DEEPSPEED", raising=False)
    assert not _environment_explains(["real_torch_python is not configured"])
    assert _environment_explains(["deepspeed optional dependency is not installed"])


def test_nightly_requires_l0_and_pins_original_source():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/deepspeed-l0.yml").read_text(encoding="utf-8")
    nox = (root / "noxfile.py").read_text(encoding="utf-8")
    assert "schedule:" in workflow
    assert "nox -s deepspeed_l0" in workflow
    assert "b3318064ee5798e8a27d201ea8b888f0439973c4eac9af9ab381dd1862ebdf45" in workflow
    gate = nox.split("def deepspeed_l0(session):", 1)[1].split("@nox.session", 1)[0]
    for required in ("JITTOR_REQUIRE_DEEPSPEED", "JITTOR_REQUIRE_REAL_TORCH", "JITTOR_TEST_REQUIRE_EXECUTION"):
        assert required + '=\"1\"' in gate
    assert 'compat/tests/torch/test_deepspeed_l0.py' in gate

    assert 'str(REPO_ROOT / "adapters")' in gate
    assert '"pillow==11.0.0"' in gate
