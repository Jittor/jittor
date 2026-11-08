"""Native deterministic policy: default disabled, explicit provider enforcement."""

import pytest
import jittor as jt
import jittor_utils


_PROBE_SOURCE = r"""
#include "runtime/backend.h"
#include <stdexcept>
namespace jittor {
namespace {
BackendOps saved_policy_provider;
BackendOps* policy_provider = nullptr;
bool policy_enabled = false, ignore_policy_write = false, fail_policy_read = false;
int policy_writes = 0, policy_reads = 0;
void observed_policy_set(bool enabled) {
    ++policy_writes;
    if (!ignore_policy_write) policy_enabled = enabled;
}
bool observed_policy_get() {
    ++policy_reads;
    if (fail_policy_read) throw std::runtime_error("observed policy query failed");
    return policy_enabled;
}
}
// @pyjt(deterministic_probe_install)
void deterministic_probe_install(int kind, bool ignore_write, bool fail_read) {
    if (policy_provider) throw std::runtime_error("policy probe already installed");
    policy_provider = &const_cast<BackendOps&>(backend_ops(accelerator_backend_id()));
    saved_policy_provider = *policy_provider;
    policy_enabled = ignore_policy_write = fail_policy_read = false;
    policy_writes = policy_reads = 0;
    ignore_policy_write = ignore_write;
    fail_policy_read = fail_read;
    policy_provider->set_deterministic_algorithms =
        kind == 1 || kind == 2 ? observed_policy_set : nullptr;
    policy_provider->get_deterministic_algorithms =
        kind == 1 || kind == 3 ? observed_policy_get : nullptr;
}
// @pyjt(deterministic_probe_restore)
void deterministic_probe_restore() {
    if (!policy_provider) return;
    *policy_provider = saved_policy_provider;
    policy_provider = nullptr;
}
// @pyjt(deterministic_probe_counts)
vector<int> deterministic_probe_counts() {
    return {int(policy_enabled), policy_writes, policy_reads};
}
}
"""


@pytest.fixture(scope="module")
def policy_probe():
    if jt.compiler.build_config.backend != "cpu":
        pytest.skip("CPU-only provider instrumentation is unavailable in an accelerator build")
    return jittor_utils.compile_module(_PROBE_SOURCE, jt.compiler.cc_flags)


@pytest.fixture
def provider(policy_probe):
    try:
        yield policy_probe
    finally:
        policy_probe.deterministic_probe_restore()


def test_cpu_default_disabled_policy_needs_no_accelerator():
    if jt.compiler.build_config.backend != "cpu":
        pytest.skip("CPU-only default policy assertion is unavailable in an accelerator build")
    with jt.flag_scope(use_cuda=0):
        assert jt.core.backend_get_deterministic_algorithms() is False
        for _ in range(3):
            jt.core.backend_set_deterministic_algorithms(False)
            assert jt.core.backend_get_deterministic_algorithms() is False
        with pytest.raises(RuntimeError, match="does not support deterministic"):
            jt.core.backend_set_deterministic_algorithms(True)
        assert jt.core.backend_get_deterministic_algorithms() is False


def test_complete_provider_owns_policy_and_readback(provider):
    provider.deterministic_probe_install(1, False, False)
    for enabled in (True, False, True):
        jt.core.backend_set_deterministic_algorithms(enabled)
        assert jt.core.backend_get_deterministic_algorithms() is enabled
    state, writes, reads = provider.deterministic_probe_counts()
    assert state == 1 and writes == 3 and reads >= 3


@pytest.mark.parametrize("kind", [2, 3])
def test_partial_provider_is_refused_before_any_callback(provider, kind):
    provider.deterministic_probe_install(kind, False, False)
    for enabled in (False, True):
        with pytest.raises(RuntimeError, match="does not support deterministic"):
            jt.core.backend_set_deterministic_algorithms(enabled)
    with pytest.raises(RuntimeError, match="does not support deterministic"):
        jt.core.backend_get_deterministic_algorithms()
    assert provider.deterministic_probe_counts() == [0, 0, 0]


def test_runtime_ignoring_requested_policy_is_refused(provider):
    provider.deterministic_probe_install(1, True, False)
    with pytest.raises(RuntimeError, match="readback mismatch"):
        jt.core.backend_set_deterministic_algorithms(True)
    assert jt.core.backend_get_deterministic_algorithms() is False


def test_provider_query_errors_propagate(provider):
    provider.deterministic_probe_install(1, False, True)
    with pytest.raises(RuntimeError, match="observed policy query failed"):
        jt.core.backend_set_deterministic_algorithms(True)
    with pytest.raises(RuntimeError, match="observed policy query failed"):
        jt.core.backend_get_deterministic_algorithms()
