# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
'''The single build lock shared by the Python and the C++ side.

Both sides of Jittor serialize compilation against ``jittor.lock``. They used
to do it with two *different* kinds of POSIX lock on two *different* file
descriptors: Python called ``fcntl.flock`` (a BSD lock, attached to the open
file description) while C++ called ``fcntl(F_SETLKW)`` (a POSIX record lock,
attached to the (process, inode) pair). On Linux those two lock families are
completely independent -- taking one never blocks the other -- so a Python
download and a C++ op compilation could write the same cache directory at the
same time while each believed it held the lock. The record lock brought a
second trap along: closing *any* descriptor for a file drops *all* of that
process's record locks on it, so a garbage-collected Python lock object could
silently release the lock C++ was relying on.

This module is now the only implementation. There is exactly one descriptor,
opened here, and exactly one kind of lock (``flock``) taken on it. Once
``jittor_core`` is loaded the descriptor and the "do we hold it" flag are
handed to C++ (:meth:`Lock.bind_core`), and every later lock/unlock from
either language goes through that single shared state.
'''

import errno
import json
import os
import sys
import time

try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
        import win32file
        import pywintypes
        _OVERLAPPED = pywintypes.OVERLAPPED()
    except:
        raise Exception("""pywin32 package not found, please install it.
>>> python3.x -m pip install pywin32
If conda is used, please install with command:
>>> conda install pywin32""")

from jittor_utils import cache_path, LOG

disable_lock = os.environ.get("disable_lock", "0") == "1"


def _env_seconds(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        LOG.w(f"{name}={value!r} is not a number, using {default}")
        return default


# How long to wait for jittor.lock before giving up. A cold build of the core
# takes minutes and a second process legitimately waits for all of it, so the
# default is generous; 0 means wait forever (the old behaviour).
lock_timeout = _env_seconds("JT_LOCK_TIMEOUT", 1800.0)
# How long to wait before naming the current holder. An orphaned or wedged
# holder used to produce no output at all, for as long as it took.
lock_report_after = _env_seconds("JT_LOCK_REPORT_AFTER", 30.0)

_HOLDER_RECORD_SIZE = 4096


def _self_cmdline():
    try:
        with open("/proc/self/cmdline", "rb") as f:
            argv = f.read().split(b"\0")
    except OSError:
        argv = [a.encode("utf8", "replace") for a in sys.argv]
    cmd = " ".join(a.decode("utf8", "replace") for a in argv if a)
    return cmd[:512]


def _write_holder(fd):
    '''Record who holds the lock, so a waiter can name the process it waits for.'''
    record = json.dumps({
        "pid": os.getpid(),
        "time": time.time(),
        "cmd": _self_cmdline(),
    })
    blob = record.encode("utf8")[:_HOLDER_RECORD_SIZE - 1] + b"\n"
    try:
        os.ftruncate(fd, 0)
        os.pwrite(fd, blob, 0)
    except OSError as e:
        # A read-only or full filesystem must not stop us from building; the
        # only thing lost is the diagnostic.
        LOG.v(f"could not record lock holder: {e}")


def _pid_alive(pid):
    if os.path.isdir("/proc"):
        return os.path.isdir(f"/proc/{pid}")
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno != errno.ESRCH
    return True


def describe_holder(fd):
    '''Human readable description of whoever wrote the holder record.'''
    try:
        raw = os.pread(fd, _HOLDER_RECORD_SIZE, 0)
    except (OSError, AttributeError):
        return "holder unknown"
    text = raw.split(b"\0", 1)[0].decode("utf8", "replace").strip()
    if not text:
        return "holder unknown (no record was written)"
    try:
        record = json.loads(text)
        pid = int(record["pid"])
    except (ValueError, KeyError, TypeError):
        return f"holder record: {text[:200]}"
    held_for = ""
    if isinstance(record.get("time"), (int, float)):
        held_for = f" for {time.time() - record['time']:.0f}s"
    alive = "" if _pid_alive(pid) else \
        " -- that process is gone, the record is stale"
    return f"held by pid {pid}{held_for}: {record.get('cmd', '')}{alive}"


def _acquire(fd, path):
    '''Take the exclusive lock on ``fd``, reporting and timing out while waiting.

    ``flock`` has no timeout, so this polls with ``LOCK_NB`` instead of using a
    blocking call. Polling costs a wakeup every half second while waiting,
    which is nothing next to the compile it is waiting for, and buys the two
    things a blocking wait could never give: a message naming the holder, and
    an error instead of an unbounded hang.
    '''
    if fcntl is None:
        hfile = win32file._get_osfhandle(fd)
        win32file.LockFileEx(hfile, 2, 0, -0x10000, _OVERLAPPED)
        return
    start = time.monotonic()
    reported = False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as e:
            if e.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                raise RuntimeError(
                    f"could not lock {path}: {e}. Set disable_lock=1 to run "
                    f"without the build lock (unsafe if anything else is "
                    f"building into the same cache).") from e
        waited = time.monotonic() - start
        if not reported and waited >= lock_report_after:
            LOG.w(f"waiting for build lock {path}, {describe_holder(fd)}")
            reported = True
        if lock_timeout > 0 and waited >= lock_timeout:
            raise RuntimeError(
                f"timed out after {waited:.0f}s waiting for the build lock "
                f"{path}, {describe_holder(fd)}. Raise JT_LOCK_TIMEOUT (0 "
                f"waits forever) if a cold build really takes this long.")
        time.sleep(0.05 if waited < 1 else 0.5)


def _release(fd):
    if fcntl is None:
        hfile = win32file._get_osfhandle(fd)
        win32file.UnlockFileEx(hfile, 0, -0x10000, _OVERLAPPED)
        return
    fcntl.flock(fd, fcntl.LOCK_UN)


class Lock:
    '''Exclusive lock on one file, shared with the C++ side once it is loaded.'''

    def __init__(self, filename):
        self.filename = filename
        # O_CREAT rather than truncating: the body of the file is the holder
        # record, and a waiter must be able to read the holder's record rather
        # than an empty file left behind by its own open().
        self.fd = os.open(filename, os.O_RDWR | os.O_CREAT, 0o666)
        LOG.v(f'OPEN LOCK path: {filename} PID: {os.getpid()}')
        self._py_is_locked = False
        # jittor_core, once compiler.py has handed it this descriptor. From
        # then on it owns the state and both languages go through it.
        self.core = None

    def fileno(self):
        return self.fd

    def bind_core(self, core):
        '''Hand the descriptor and the current state to the C++ side.

        Called once, right after ``jittor_core`` is imported. Everything before
        this point ran with no C++ code loaded at all, so there was no second
        holder to disagree with.
        '''
        handle = self.fd
        if fcntl is None:
            handle = msvcrt.get_osfhandle(self.fd)
        core.set_lock_fd(handle, self._py_is_locked)
        self.core = core

    @property
    def is_locked(self):
        if self.core is not None:
            return self.core.lock_is_held()
        return self._py_is_locked

    def lock(self):
        if disable_lock:
            return
        if self.core is not None:
            self.core.lock_acquire()
            return
        if self._py_is_locked:
            return
        _acquire(self.fd, self.filename)
        _write_holder(self.fd)
        self._py_is_locked = True
        LOG.vv(f'LOCK PID: {os.getpid()}')

    def unlock(self):
        if disable_lock:
            return
        if self.core is not None:
            self.core.lock_release()
            return
        if not self._py_is_locked:
            return
        _release(self.fd)
        self._py_is_locked = False
        LOG.vv(f'UNLOCK PID: {os.getpid()}')

    # NOTE: deliberately no __del__. The descriptor lives as long as the
    # process and the kernel closes it at exit. Closing it from a finalizer is
    # what used to drop the C++ side's lock out from under it.


class _base_scope:
    '''base_scope for support @xxx syntax'''
    def __enter__(self): pass
    def __exit__(self, *exc): pass
    def __call__(self, func):
        def inner(*args, **kw):
            with self:
                ret = func(*args, **kw)
            return ret
        return inner

class lock_scope(_base_scope):
    def __enter__(self):
        self.is_locked = jittor_lock.is_locked
        if not self.is_locked:
            jittor_lock.lock()

    def __exit__(self, *exc):
        if not self.is_locked:
            jittor_lock.unlock()

class unlock_scope(_base_scope):
    def __enter__(self):
        self.is_locked = jittor_lock.is_locked
        if self.is_locked:
            jittor_lock.unlock()

    def __exit__(self, *exc):
        if self.is_locked:
            jittor_lock.lock()

lock_path = os.path.abspath(os.path.join(cache_path, "../jittor.lock"))
jittor_lock = Lock(lock_path)
