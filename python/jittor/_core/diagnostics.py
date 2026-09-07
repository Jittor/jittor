"""Log capture, profiling, distributed scopes and process cleanup."""

import sys
import traceback

import jittor_core as core
from jittor_core import sync_all
from jittor_utils import dirty_fix_pytorch_runtime_error

from . import flags as _flag_state
from .flags import _call_no_record_scope, flag_scope

_core_profiler = core.profiler


single_log_capture = None

class log_capture_scope(_call_no_record_scope):
    """log capture scope

    Example::

        with jt.log_capture_scope(log_v=0) as logs:
            LOG.v("...")
        print(logs)
    """
    def __init__(self, **jt_flags):
        jt_flags["use_parallel_op_compiler"] = 0
        self.fs = flag_scope(**jt_flags)

    def __enter__(self):
        import jittor as jt
        from jittor.compiler import LOG

        global single_log_capture
        assert not single_log_capture
        single_log_capture = jt.single_log_capture = True
        self.logs = []
        LOG.log_capture_start()
        try:
            self.fs.__enter__()
            if "log_v" in self.fs.jt_flags:
                LOG.log_v = self.fs.jt_flags["log_v"]
            return self.logs
        except:
            LOG.log_capture_stop()
            single_log_capture = jt.single_log_capture = None
            raise

    def __exit__(self, *exc):
        import jittor as jt
        from jittor.compiler import LOG

        global single_log_capture
        self.fs.__exit__(*exc)
        if "log_v" in self.fs.jt_flags:
            LOG.log_v = _flag_state.flags.log_v
        LOG.log_capture_stop()
        self.logs.extend(LOG.log_capture_read())
        single_log_capture = jt.single_log_capture = None


class profile_scope(_call_no_record_scope):
    """ profile scope

    example::

        with jt.profile_scope() as report:
            ......
        print(report)
    """
    def __init__(self, warmup=0, rerun=0, **jt_flags):
        self.fs = flag_scope(**jt_flags)
        self.warmup = warmup
        self.rerun = rerun

    def __enter__(self):
        assert not _flag_state.flags.profiler_enable
        self.report = []
        try:
            self.fs.__enter__()
            _core_profiler.start(self.warmup, self.rerun)
            return self.report
        except:
            _core_profiler.stop()
            raise

    def __exit__(self, *exc):
        _core_profiler.stop()
        self.report.extend(_core_profiler.report())
        self.fs.__exit__(*exc)


class profile_mark(_call_no_record_scope):
    def __init__(self, mark_name: str):
        ''' profiler mark is used for profiling part of code,

        Example::

        a = jt.rand(1000,1000)
        b = jt.rand(1000,1000)
        jt.sync_all()
        results = []
        with jt.profile_scope() as rep:
            results.append(jt.matmul(a, b))
            with jt.profile_mark("mark1"):
                results.append(jt.matmul(a, b))
                with jt.profile_mark("mark2"):
                    results.append(jt.matmul(a, b))
            with jt.profile_mark("mark3"):
                results.append(jt.matmul(a, b))
            results.append(jt.matmul(a, b))

        Output::

        Total time:    46.8ms
        Total Memory Access:    57.2MB
        [Mark mark3] time:       9ms
        [Mark mark2] time:    8.28ms
        [Mark mark1] time:    17.7ms

        '''
        self.mark_name = mark_name
    def __enter__(self):
        self.options = _flag_state.flags.compile_options
        new_options = _flag_state.flags.compile_options
        prev_marks = "_marks:"
        for x in self.options:
            if x.startswith(prev_marks):
                prev_marks = x
                del new_options[x]
        new_marks = prev_marks + self.mark_name + ','
        new_options[new_marks] = 1
        _flag_state.flags.compile_options = new_options

    def __exit__(self, *exc):
        _flag_state.flags.compile_options = self.options

class __single_process_scope:
    def __init__(self, rank=0):
        self.rank = rank

    def __enter__(self):
        from jittor import compile_extern

        # compile_extern owns in_mpi; this module used to keep its own copy
        # (pulled in by `from jittor import *` at import time) and mutate that
        # alongside, so a later correction to compile_extern.in_mpi left this
        # copy stale and mpi_param_broadcast() below silently did nothing. 6.B15.
        self.bk_in_mpi = compile_extern.in_mpi
        mpi = self._mpi = compile_extern.get_library("mpi")
        if mpi:
            self.bk_mpi_state = mpi.get_state()
        if not self.bk_in_mpi:
            return True

        ret = self.rank == mpi.world_rank()
        compile_extern.in_mpi = False
        mpi.set_state(False)
        return ret

    def __exit__(self, *exc):
        from jittor import compile_extern

        compile_extern.in_mpi = self.bk_in_mpi
        mpi = self._mpi
        if mpi:
            mpi.set_state(self.bk_mpi_state)

def single_process_scope(rank=0):
    """ single_process_scope

    Code in this scope will only be executed by single process.

    All the mpi code inside this scope will have not affect.
    mpi.world_rank() and mpi.local_rank() will return 0, world_size() will return 1,

    example::

        @jt.single_process_scope(rank=0)
        def xxx():
            ...
    """
    def outer(func):
        def inner(*args, **kw):
            ret = None
            sync_all()
            with __single_process_scope(rank) as flag:
                if flag:
                    ret = func(*args, **kw)
            return ret
        return inner
    return outer

def clean():
    import gc
    # make sure python do a full collection
    gc.collect()
    core.gc()

def liveness_info():
    return {
        "hold_vars": core.number_of_hold_vars(),
        "lived_vars": core.number_of_lived_vars(),
        "lived_ops": core.number_of_lived_ops(),
    }


def display_memory_info():
    import inspect, os
    f = inspect.currentframe()
    fileline = inspect.getframeinfo(f.f_back)
    fileline = f"{os.path.basename(fileline.filename)}:{fileline.lineno}"
    core.display_memory_info(fileline)

class ExitHooks(object):
    def __init__(self):
        self.exit_code = None
        self.exception = None

    def hook(self):
        self._orig_exit = sys.exit
        sys.exit = self.exit
        sys.excepthook = self.exc_handler

    def exit(self, code=0):
        self.exit_code = code
        self._orig_exit(code)

    def exc_handler(self, exc_type, exc, *args):
        self.exception = exc
        traceback.print_exception(exc_type, exc, *args)

hooks = ExitHooks()

def jittor_exit():
    if hooks.exit_code is not None:
        pass
    elif hooks.exception is not None:
        pass
    else:
        pass
        # core.sync_all(True)
    core.cleanup()

# Bootstrap registers hooks.hook() and atexit.register(jittor_exit) only
# after composing the native API, preserving the original installation order.
