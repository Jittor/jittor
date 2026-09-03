import pytest

from _helpers.retry import RetryWarning, retry


def test_retry_reports_recovered_failure_count():
    attempts = []

    @retry(3)
    def flaky(*, value):
        attempts.append(value)
        if len(attempts) < 3:
            raise RuntimeError("transient")
        return value

    with pytest.warns(RetryWarning, match=r"passed after 2 retries \(3 attempts\)"):
        assert flaky(value=7) == 7

    assert attempts == [7, 7, 7]
    assert flaky.call_count == 1
    assert flaky.last_retries == 2
    assert flaky.total_retries == 2


def test_retry_reports_count_and_preserves_final_failure():
    attempts = []

    @retry(2)
    def always_fails():
        attempts.append(None)
        raise LookupError("permanent")

    with pytest.warns(RetryWarning, match=r"failed after 1 retries \(2 attempts\)"):
        with pytest.raises(LookupError, match="permanent"):
            always_fails()

    assert len(attempts) == 2
    assert always_fails.call_count == 1
    assert always_fails.last_retries == 1
    assert always_fails.total_retries == 1
