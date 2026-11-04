"""Actual delayed CUDA faults preserve source candidates without debug sync."""
import textwrap

import pytest

from _helpers.child_process import run_child_script
from _helpers.capability import require_accelerator


def _require_cuda():
    require_accelerator("cuda")


def test_launch_history_survives_graph_release():
    _require_cuda()
    result = run_child_script(textwrap.dedent('''
        import gc, inspect
        import jittor as jt
        with jt.runtime.scope(use_cuda=1, trace_py_var=0, auto_flush_ops=0):
            line = inspect.currentframe().f_lineno + 1
            value = jt.array([2.0]) + 3
            assert value.numpy().tolist() == [5.0]
            del value
            gc.collect()
            history = jt.core.async_launch_history("cuda", 0)
            assert __file__ + ":" + str(line) in history, history
            assert "candidates are not proof" in history
            assert "not-found" in jt.core.async_launch_history("cuda", 0, 987654)
            print("NORMAL-RELEASE-PASS", flush=True)
    '''), env={"JT_SYNC": "0", "trace_py_var": "0"}, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NORMAL-RELEASE-PASS" in result.stdout
    assert "CUDA error at" not in result.stderr


@pytest.mark.parametrize("access", ["getitem", "setitem"])
def test_an_out_of_range_index_on_the_device_is_named_in_the_error(access):
    """The kernel's report used to go to stdout only, beside a bare code=719."""
    _require_cuda()
    source = textwrap.dedent('''
        import jittor as jt
        with jt.runtime.scope(use_cuda=1):
            x = jt.array([1.0, 2.0, 3.0])
            index = jt.array([10])
            try:
                ACCESS
            except RuntimeError as error:
                report = str(error)
                assert "index 10 is out of bounds for dimension 0 with size 3" in report, report
                assert "Restart the process" in report, report
                print("INDEX-NAMED-PASS", flush=True)
            else:
                raise AssertionError("the out-of-range index was never reported")
    ''').replace("ACCESS", {
        "getitem": "x[index].numpy()",
        "setitem": "x[index] = jt.array([5.0]); x.numpy()",
    }[access])
    result = run_child_script(source, env={"JT_SYNC": "0"}, text=True, timeout=180,
                              crash_isolated=True, name="device_index_" + access)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INDEX-NAMED-PASS" in result.stdout
    assert "[jittor] index" not in result.stdout
    assert "Readback event cleanup failed" not in result.stderr


@pytest.mark.parametrize("boundary", ["device_wait", "readback"])
def test_asynchronous_fault_reports_creation_line(boundary):
    _require_cuda()
    source = textwrap.dedent('''
        import inspect
        import jittor as jt
        with jt.runtime.scope(use_cuda=1, trace_py_var=0, auto_flush_ops=0):
            x = jt.ones((1,), "float32")
            assert x.numpy().tolist() == [1.0]
            fault_line = inspect.currentframe().f_lineno + 1
            y = jt.code(x.shape, x.dtype, [x], cuda_src=r"""
                __global__ void delayed_illegal_write(float* p) {
                    unsigned long long start = clock64();
                    while (clock64() - start < 200000000ULL) {}
                    // Address 16, not p past its end: under HMM a GPU write to
                    // any mapped host page succeeds, and what lies past p depends
                    // on the stack rlimit this child inherited. Page 0 is never
                    // mapped. See tests/backends/cuda/test_backend_teardown.py.
                    *((volatile float*)16) = 1.0f;
                }
                delayed_illegal_write<<<1,1>>>(out0_p);
            """)
            y.sync()  # issue only; poison deliberately occurs later
            print("ISSUED-WITHOUT-WAIT", flush=True)
            try:
                BOUNDARY
            except RuntimeError as error:
                report = str(error)
                assert "cudaErrorIllegalAddress" in report, report
                assert "[Recent launch candidates]" in report, report
                assert __file__ + ":" + str(fault_line) in report, report
                assert "op=code" in report, report
                assert "candidates are not proof" in report, report
                assert "device=0" in report, report
                assert "sticky" in report and "Restart the process" in report, report
                print("ASYNC-ORIGIN-PASS", flush=True)
            else:
                raise AssertionError("asynchronous poison was never reported")
    ''').replace("BOUNDARY", "jt.sync_all(True)" if boundary == "device_wait" else "y.numpy()")
    result = run_child_script(source, env={"JT_SYNC": "0", "trace_py_var": "0"},
                              text=True, timeout=180, crash_isolated=True,
                              name="async_origin_" + boundary)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ISSUED-WITHOUT-WAIT" in result.stdout
    assert "ASYNC-ORIGIN-PASS" in result.stdout


def test_a_device_index_fault_leads_with_the_index_and_is_an_index_error():
    """The sentence naming the index came second, under `helper_cuda.h:NNN:
    CUDA error at ...`, and the error was a bare JittorError although the
    same index on the CPU is a JittorIndexError."""
    _require_cuda()
    result = run_child_script(textwrap.dedent('''
        import jittor as jt
        with jt.runtime.scope(use_cuda=1):
            x = jt.array([1.0, 2.0, 3.0])
            try:
                x[jt.array([10])].numpy()
            except IndexError as error:
                first = str(error).splitlines()[0]
                assert first.startswith("index 10 is out of bounds for dimension 0 with size 3"), first
                assert isinstance(error, jt.JittorError), type(error).__mro__
                assert "CUDA error at" in str(error), str(error)
                print("INDEX-FIRST-PASS", flush=True)
    '''), env={"JT_SYNC": "0"}, text=True, timeout=180,
        crash_isolated=True, name="device_index_first")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INDEX-FIRST-PASS" in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("wait", ["y.sync()", "jt.sync_all()"])
def test_a_fault_no_read_reported_fails_the_exit(wait):
    """`sync()` and `sync_all()` do not wait for the device. A kernel that
    faulted after the last read went unreported, and the process exited 0
    with only a misleading cuDNN teardown error on stderr."""
    _require_cuda()
    result = run_child_script(textwrap.dedent('''
        import jittor as jt
        jt.flags.use_cuda = 1
        x = jt.ones((5, 3))
        y = x[jt.array([7])]
        WAIT
        print("LAST-LINE", flush=True)
    ''').replace("WAIT", wait), env={"JT_SYNC": "0"}, text=True, timeout=180,
        crash_isolated=True, name="device_fault_at_exit")
    assert "LAST-LINE" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 1, result.stdout + result.stderr
    assert "a device error was still pending" in result.stderr, result.stderr
    assert "index 7 is out of bounds for dimension 0 with size 5" in result.stderr, result.stderr
    assert "teardown error" not in result.stderr, result.stderr


def test_a_fault_the_program_caught_is_not_reported_again_at_exit():
    _require_cuda()
    result = run_child_script(textwrap.dedent('''
        import jittor as jt
        jt.flags.use_cuda = 1
        try:
            jt.ones((5, 3))[jt.array([7])].numpy()
        except IndexError:
            print("CAUGHT", flush=True)
    '''), env={"JT_SYNC": "0"}, text=True, timeout=180,
        crash_isolated=True, name="device_fault_caught")
    assert "CAUGHT" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
    assert "still pending" not in result.stderr, result.stderr
    assert "teardown error" not in result.stderr, result.stderr


def test_a_clean_cuda_program_exits_zero_and_quietly():
    _require_cuda()
    result = run_child_script(textwrap.dedent('''
        import jittor as jt
        jt.flags.use_cuda = 1
        (jt.ones((5, 3)) * 2).sync()
        print("DONE", flush=True)
    '''), env={"JT_SYNC": "0"}, text=True, timeout=180, name="device_clean_exit")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "still pending" not in result.stderr, result.stderr


def _require_two_cuda_devices():
    _require_cuda()
    import jittor as jt
    from _helpers import capability
    if int(capability.device_count("cuda", backend=jt)) < 2:
        pytest.skip("needs at least two cuda devices")


def test_a_fault_on_another_device_is_reported_after_switching_back():
    """The exit wait synchronized only the current device: a kernel that
    faulted on device 1 after the program had switched back to device 0 went
    unreported and the process exited 0."""
    _require_two_cuda_devices()
    result = run_child_script(textwrap.dedent('''
        import jittor as jt
        jt.flags.use_cuda = 1
        jt.flags.device_id = 0
        (jt.ones(3) * 2).numpy()
        with jt.flag_scope(device_id=1):
            y = jt.ones((5, 3))[jt.array([7])]
            y.sync()
        (jt.ones(3) * 2).numpy()
        print("LAST-LINE", flush=True)
    '''), env={"JT_SYNC": "0"}, text=True, timeout=180,
        crash_isolated=True, name="device_fault_other_device")
    assert "LAST-LINE" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 1, result.stdout + result.stderr
    assert "a device error was still pending" in result.stderr, result.stderr
    assert "index 7 is out of bounds for dimension 0 with size 5" in result.stderr, result.stderr
    assert "teardown error" not in result.stderr, result.stderr


def test_a_caught_fault_on_one_device_does_not_hide_another_devices_fault():
    """Whether a context was lost was one flag for the whole process: once the
    program had caught device 0's fault, a later fault on device 1 was skipped
    at exit as if it were the same one."""
    _require_two_cuda_devices()
    result = run_child_script(textwrap.dedent('''
        import jittor as jt
        jt.flags.use_cuda = 1
        jt.flags.device_id = 0
        # Both contexts exist before the fault: after a sticky error CUDA
        # refuses to create another (cudaErrorDevicesUnavailable).
        for device in (0, 1):
            with jt.flag_scope(device_id=device):
                (jt.ones(3) * 2).numpy()
        try:
            jt.ones((5, 3))[jt.array([7])].numpy()
        except IndexError:
            print("CAUGHT", flush=True)
        with jt.flag_scope(device_id=1):
            y = jt.ones((4, 3))[jt.array([9])]
            y.sync()
        print("LAST-LINE", flush=True)
    '''), env={"JT_SYNC": "0"}, text=True, timeout=180,
        crash_isolated=True, name="device_fault_second_device")
    assert "CAUGHT" in result.stdout, result.stdout + result.stderr
    assert "LAST-LINE" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 1, result.stdout + result.stderr
    assert "a device error was still pending" in result.stderr, result.stderr
    # Device 1's own index is named only where its kernel still runs: CUDA
    # fails every device of a process once one has a sticky error (driver
    # 580), and then the launch never happens. Either way the report is
    # device 1's, with the line that gave it the work.
    assert "device=1" in result.stderr, result.stderr
    assert "device_fault_second_device" in result.stderr and ".py:15\n" in result.stderr, result.stderr
    assert "teardown error" not in result.stderr, result.stderr


def test_a_fault_caught_on_one_device_is_not_reported_again_through_another():
    """Once one device has a sticky error, CUDA fails the others too. A device
    whose work had all been read before the fault, and that was given nothing
    since, must not report that same error at exit as if it were pending."""
    _require_two_cuda_devices()
    result = run_child_script(textwrap.dedent('''
        import jittor as jt
        jt.flags.use_cuda = 1
        jt.flags.device_id = 0
        (jt.ones(3) * 2).numpy()
        with jt.flag_scope(device_id=1):
            try:
                jt.ones((4, 3))[jt.array([9])].numpy()
            except IndexError:
                print("CAUGHT", flush=True)
        print("LAST-LINE", flush=True)
    '''), env={"JT_SYNC": "0"}, text=True, timeout=180,
        crash_isolated=True, name="device_fault_caught_other_device")
    assert "CAUGHT" in result.stdout, result.stdout + result.stderr
    assert "LAST-LINE" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
    assert "still pending" not in result.stderr, result.stderr
    assert "teardown error" not in result.stderr, result.stderr
