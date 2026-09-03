import pytest

from _helpers.state_leaks import assert_rss_growth_bounded


def test_rss_bound_rejects_an_intentional_retained_allocation():
    retained = []

    def leak_one_mebibyte():
        retained.append(bytearray(1 << 20))

    with pytest.raises(AssertionError, match="RSS grew"):
        assert_rss_growth_bounded(
            leak_one_mebibyte,
            warmup=0,
            iterations=8,
            max_growth_bytes=4 << 20,
        )
