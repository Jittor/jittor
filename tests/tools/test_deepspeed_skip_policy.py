"""DeepSpeed absence is optional only when the gate did not promise the library."""
import pytest

from _helpers import pytest_policy


@pytest.mark.parametrize("setting", ("1", "true", "yes", "on"))
def test_required_deepspeed_withdraws_environment_skip(monkeypatch, setting):
    reason = {"deepspeed is not installed"}
    monkeypatch.delenv("JITTOR_REQUIRE_DEEPSPEED", raising=False)
    assert pytest_policy._environment_explains(reason)
    monkeypatch.setenv("JITTOR_REQUIRE_DEEPSPEED", setting)
    assert not pytest_policy._environment_explains(reason)
    assert pytest_policy._environment_explains({"nvcc is not installed"})
