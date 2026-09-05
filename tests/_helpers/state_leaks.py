"""Report runtime state one test file leaves behind for the next one.

Three failures in this tree were the same failure. Each looked like a bug in the
test that reported it, and in each case the state had been left behind by a file
that had already passed:

* ``tests/compiler/test_fused_op.py::TestFusedOp::test_add`` asserts
  ``number_of_hold_vars() == 0``. It passes alone and fails after
  ``tests/core`` and ``tests/ops`` have run.
* ``test_torch_compat_fsdp2::test_single_rank_fully_shard_preserves_math_and_state``
  passes alone (20 passed) and fails in a combined run.
* ``tests/ops/test_linalg.py::TestBUG4_2Op`` sets ``use_cuda=1`` and never
  restores it, so every later case in that file runs on CUDA -- where ``eigh``'s
  gradient is wrong (6.P23). The file "passes" because the tests that would
  catch it are the ones being silently redirected.

The debugging cost is what makes this worth automating: the symptom appears in a
file that is innocent, so the first day of every such investigation is spent in
the wrong place. This module snapshots the runtime around each test *file* and
names the file that changed something.

It reports; it does not fail. The survey has to run over the whole tree before
anyone can say which of these are bugs and which are load-bearing, and a check
that goes red on its first run gets switched off rather than read.
"""

import gc
import os
import sys


#: Flags that change what later tests measure rather than how fast they run.
#: ``use_cuda`` decides the device an op runs on, ``no_grad`` and ``amp_reg``
#: decide what a gradient means, and ``exclude_pass``/``use_parallel_op_compiler``
#: decide which compiled kernel is executed.
WATCHED_FLAGS = (
    "use_cuda",
    "no_grad",
    "amp_reg",
    "cuda_kernel_math",
    "use_parallel_op_compiler",
    "exclude_pass",
)

_COUNTERS = (
    "number_of_hold_vars",
    "number_of_lived_vars",
    "number_of_lived_ops",
)

#: Process-level caches *inside jittor* that legitimately hold Vars, as
#: ``(module, cache attribute, limit attribute, Vars per entry)``.
#:
#: Every one of these makes ``number_of_hold_vars`` rise and stay risen, and
#: none of them is the test file's doing. Two whole-tree surveys reported them
#: as leaks and both were read as "the test module kept a Var", which is wrong
#: in a way that matters: there is nothing to fix in the test. The measurement
#: that settled it -- ``jt.dump_all_graphs().hold_vars`` after dropping the test
#: module, pytest and ``_helpers.common`` -- showed the count unchanged, and the
#: held Vars were pairs of float32 [n,n] (DFT cos/sin matrices) and int32
#: cumulative-sequence-length vectors. Both caches are LRU with a stated limit,
#: so the floor they raise is bounded, not a leak.
#:
#: The consequence is the rule this survey exists to teach: **an absolute
#: assertion on a global counter is wrong by construction**, because the floor
#: depends on which operators the process has ever run, not on the test.
BOUNDED_VAR_CACHES = (
    ("jittor.fft", "_dft_mat_cache", "_dft_mat_cache_limit", 2),
    ("jittor.nn.attention", "_CU_SEQLENS_CACHE", "_CU_SEQLENS_CACHE_LIMIT", 1),
)


def resident_set_size_bytes():
    """Return current RSS where available, otherwise the process high-water RSS."""
    try:
        with open("/proc/self/statm") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)


def assert_rss_growth_bounded(workload, *, warmup=8, iterations=64,
                              max_growth_bytes=16 << 20, cleanup=None):
    """Exercise one allocation lifecycle and fail when retained RSS is unbounded."""
    for _ in range(warmup):
        workload()
    if cleanup is not None:
        cleanup()
    gc.collect()
    before = resident_set_size_bytes()

    for index in range(iterations):
        workload()
        if cleanup is not None and (index + 1) % 8 == 0:
            cleanup()
    if cleanup is not None:
        cleanup()
    gc.collect()
    after = resident_set_size_bytes()

    growth = max(0, after - before)
    if growth > max_growth_bytes:
        raise AssertionError(
            "RSS grew by %.2f MiB across %d iterations (limit %.2f MiB)"
            % (growth / (1 << 20), iterations, max_growth_bytes / (1 << 20))
        )
    return growth


def _bounded_cache_sizes():
    """``{name: (entries, vars, limit)}`` for the caches above, if imported."""
    sizes = {}
    for module_name, attribute, limit_attribute, per_entry in BOUNDED_VAR_CACHES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        cache = getattr(module, attribute, None)
        if cache is None:
            continue
        try:
            entries = len(cache)
        except TypeError:
            continue
        limit = getattr(module, limit_attribute, None)
        sizes["%s.%s" % (module_name, attribute)] = (
            entries, entries * per_entry, limit)
    return sizes


def _jittor():
    """The runtime, only if this session already imported it.

    Importing it here would pull the core into every static structure test and
    turn a 50 ms file into a compile.
    """
    return sys.modules.get("jittor")


def snapshot(collect=True):
    jittor = _jittor()
    if jittor is None:
        return None
    if collect:
        # Without this the counters measure Python's garbage collector schedule
        # rather than the test file, and every report would be noise.
        gc.collect()
    counters = {}
    for name in _COUNTERS:
        function = getattr(jittor, name, None)
        if callable(function):
            try:
                counters[name] = function()
            except Exception:
                pass
    flags = {}
    for name in WATCHED_FLAGS:
        try:
            flags[name] = getattr(jittor.flags, name)
        except Exception:
            continue
    return {
        "counters": counters,
        "flags": flags,
        "autograd_policy": _autograd_policy(jittor),
        "caches": _bounded_cache_sizes(),
        "modules": {name: id(module) for name, module in list(sys.modules.items())},
    }


def differences(before, after):
    """Human-readable descriptions of what changed, or an empty list."""
    if not before or not after:
        return []
    report = []
    explained, notes = _cache_growth(before, after)
    for name, value in sorted(after["counters"].items()):
        previous = before["counters"].get(name)
        if previous is None or previous == value:
            continue
        line = "%s %s -> %s" % (name, previous, value)
        # Say what is jittor's own bounded memoisation and what is left over.
        # The residual is the only part anyone should go looking for.
        if name in ("number_of_hold_vars", "number_of_lived_vars") and explained:
            residual = (value - previous) - explained
            line += " -- %d of %d is %s%s" % (
                explained, value - previous, "; ".join(notes),
                "" if residual <= 0 else "; %d unexplained" % residual)
        report.append(line)
    for name, value in sorted(after["flags"].items()):
        previous = before["flags"].get(name)
        if previous is not None and previous != value:
            report.append("flags.%s %r -> %r (use jt.flag_scope)" % (name, previous, value))
    if before.get("autograd_policy") != after.get("autograd_policy"):
        report.append(
            "autograd policy %r -> %r (use jt.autograd.policy_scope)"
            % (before.get("autograd_policy"), after.get("autograd_policy"))
        )
    for name, identity in sorted(after["modules"].items()):
        previous = before["modules"].get(name)
        if previous is not None and previous != identity:
            # A different object under the same name: something replaced an
            # imported module and did not put the original back.
            report.append("sys.modules[%r] was replaced and not restored" % name)
    return report


def _autograd_policy(jittor):
    autograd = getattr(jittor, "autograd", None)
    get_policy = getattr(autograd, "get_policy", None)
    if not callable(get_policy):
        return None
    try:
        policy = get_policy()
    except Exception:
        return None
    return (
        policy.name,
        policy.stop_outputs_when_inputs_stopped,
        policy.preserve_requires_grad_on_assignment,
    )


def _cache_growth(before, after):
    """How many of the new Vars are jittor's own bounded caches warming up."""
    explained = 0
    notes = []
    for name, (entries, held, limit) in sorted(after.get("caches", {}).items()):
        was_entries, was_held, _limit = before.get("caches", {}).get(
            name, (0, 0, limit))
        if held <= was_held:
            continue
        explained += held - was_held
        notes.append("%s %d -> %d entries (bounded at %s)" % (
            name, was_entries, entries, limit))
    return explained, notes
