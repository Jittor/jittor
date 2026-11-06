"""The OpenMP pool setter and getter operate on the native runtime."""

import zlib

import jittor as jt
import numpy as np
import pytest


def test_thread_pool_runtime_round_trip():
    original = jt.core.get_num_threads()
    try:
        jt.core.set_num_threads(2)
        assert jt.core.get_num_threads() == 2
    finally:
        jt.core.set_num_threads(original)


_THREAD_TEAM_PROBE = r"""
#pragma once
#include "core/op.h"
#include "ops/op_register.h"
#include <stdexcept>
namespace jittor {
namespace {
OpDef thread_team_original;
bool thread_team_installed = false;
int thread_team_compiles = 0;
void thread_team_codegen(Op* op, string& source) {
    ++thread_team_compiles;
    auto original = thread_team_original.implementations.at(BackendId::Cpu).codegen.optimize;
    if (original) original(op, source);
    const string needle = "omp_get_thread_num();";
    size_t at = 0;
    int region = 0;
    while ((at = source.find(needle, at)) != string::npos) {
        string call = "jittor_test_record_team(" + std::to_string(region++) + ");";
        at += needle.size();
        source.insert(at, call);
        at += call.size();
    }
    size_t cursor = 0;
    while ((cursor = source.find("#pragma omp parallel for", cursor)) != string::npos) {
        size_t line_end = source.find('\n', cursor);
        size_t collapse = source.find("collapse(", cursor);
        int depth = collapse < line_end ? std::stoi(source.substr(collapse + 9)) : 1;
        size_t body = cursor;
        for (int index = 0; index < depth; ++index) {
            body = source.find('{', body + 1);
            if (body == string::npos) break;
        }
        if (body == string::npos) break;
        string call = "jittor_test_record_team(" + std::to_string(region++) + ");";
        source.insert(body + 1, call);
        cursor = body + 1 + call.size();
    }
    source.insert(0, R"PROBE(
#include <cstdio>
#include <cstdlib>
#include <string>
#include <set>
#include <omp.h>
inline static void jittor_test_record_team(int region) {
    if (omp_get_thread_num() != 0) return;
    const char* path = std::getenv("JITTOR_TEST_THREAD_TEAM_LOG");
    if (!path) return;
    static thread_local std::string previous;
    static thread_local std::set<int> recorded;
    if (previous != path) { previous = path; recorded.clear(); }
    if (!recorded.insert(region).second) return;
    FILE* stream = std::fopen(path, "a");
    if (stream) {
        std::fprintf(stream, "%d\n", omp_get_num_threads());
        std::fclose(stream);
    }
}
)PROBE");
}
}
// @pyjt(thread_team_install)
void thread_team_install() {
    if (thread_team_installed) throw std::runtime_error("thread team probe installed twice");
    thread_team_original = get_op_info("fused");
    OpDef replacement = thread_team_original;
    replacement.implementations.at(BackendId::Cpu).codegen.optimize = thread_team_codegen;
    thread_team_compiles = 0;
    op_registe(replacement);
    thread_team_installed = true;
}
// @pyjt(thread_team_restore)
void thread_team_restore() {
    if (thread_team_installed) op_registe(thread_team_original);
    thread_team_installed = false;
}
// @pyjt(thread_team_compile_count)
int thread_team_compile_count() { return thread_team_compiles; }
}
"""


@pytest.fixture
def kernel_team_probe(tmp_path, monkeypatch):
    import jittor_utils

    probe = jittor_utils.compile_module(_THREAD_TEAM_PROBE, jt.compiler.cc_flags)
    original = jt.core.get_num_threads()
    sequence = [0]
    probe.thread_team_install()

    def observe(build):
        path = tmp_path / ("team-%d.txt" % sequence[0])
        sequence[0] += 1
        monkeypatch.setenv("JITTOR_TEST_THREAD_TEAM_LOG", str(path))
        before = probe.thread_team_compile_count()
        value = build().numpy()
        after = probe.thread_team_compile_count()
        assert path.is_file(), "the generated kernel did not execute an OpenMP region"
        teams = [int(line) for line in path.read_text().splitlines()]
        assert teams
        return value, teams, before, after

    observe.tag = zlib.crc32(str(tmp_path).encode()) & 0x7FFFFFFF
    try:
        yield observe
    finally:
        try:
            jt.sync_all(True)
        finally:
            probe.thread_team_restore()
            jt.core.set_num_threads(original)


@pytest.mark.parametrize("fresh", (False, True))
@pytest.mark.parametrize("reduction", (False, True))
def test_default_cpu_kernels_observe_changed_thread_pool(kernel_team_probe, fresh, reduction):
    data = np.arange(512 * 1024, dtype=np.float32).reshape(512, 1024) % 8
    source = jt.array(data).sync()
    previous_compiles = None
    for threads in (1, 2, 3):
        jt.core.set_num_threads(threads)
        options = {"thread_pool_default": int(reduction), "thread_pool_run": kernel_team_probe.tag}
        if fresh:
            options["thread_pool_fresh"] = threads
        with jt.flag_scope(use_cuda=0, compile_options=options):
            got, teams, before, after = kernel_team_probe(
                lambda: (source + 1).sum(1) if reduction else source * 2 + 1
            )
        expected = (data + 1).sum(1) if reduction else data * 2 + 1
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-5)
        assert max(teams) == threads
        assert all(1 <= count <= threads for count in teams)
        if fresh or previous_compiles is None:
            assert after > before
        else:
            assert after == previous_compiles, "the test must exercise the cached kernel"
        previous_compiles = after


@pytest.mark.parametrize("level", (0, 3))
@pytest.mark.parametrize("fresh", (False, True))
@pytest.mark.parametrize("reduction", (False, True))
def test_parallel_cpu_kernels_never_exceed_thread_budget(
    kernel_team_probe, level, fresh, reduction
):
    data = np.arange(64 * 1024, dtype=np.float32).reshape(64, 1024) % 8
    source = jt.array(data).sync()
    previous_compiles = None
    for threads in (1, 2, 3):
        jt.core.set_num_threads(threads)
        options = {
            "thread_pool_run": kernel_team_probe.tag,
            "parallel": 1,
            "max_parallel_depth": 2,
            "thread_pool_parallel": int(reduction),
            "thread_pool_level": level,
        }
        if fresh:
            options["thread_pool_fresh"] = threads
        with jt.flag_scope(use_cuda=0, para_opt_level=level, compile_options=options):
            got, teams, before, after = kernel_team_probe(
                lambda: source.sum() if reduction else source * 2 + 1
            )
        expected = data.sum() if reduction else data * 2 + 1
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-5)
        # The bit-partitioned loop scheduler may use fewer workers, but must
        # never start a team larger than the runtime's requested upper bound.
        assert all(1 <= count <= threads for count in teams), (threads, teams)
        if fresh or previous_compiles is None:
            assert after > before
        else:
            assert after == previous_compiles, "the test must exercise the cached kernel"
        previous_compiles = after


def test_explicit_cpu_thread_loop_option_keeps_precedence(kernel_team_probe):
    data = np.arange(64 * 1024, dtype=np.float32).reshape(64, 1024) % 8
    source = jt.array(data).sync()
    jt.core.set_num_threads(1)
    for reduction in (False, True):
        options = {
            "thread_pool_run": kernel_team_probe.tag,
            "parallel": 1,
            "max_parallel_depth": 2,
            "cpu_thread_num": 4,
            "thread_pool_override": int(reduction),
        }
        with jt.flag_scope(use_cuda=0, para_opt_level=3, compile_options=options):
            got, teams, _, _ = kernel_team_probe(
                lambda: source.sum() if reduction else source * 2 + 1
            )
        expected = data.sum() if reduction else data * 2 + 1
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-5)
        assert max(teams) == 4
        assert all(1 <= count <= 4 for count in teams)
