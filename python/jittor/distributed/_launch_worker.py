"""Keep a multi-node worker tied to the lifetime of its local launch agent."""

import ctypes
import os
import signal
import sys


def main():
    parent = int(sys.argv[1])
    command = sys.argv[2:]
    if sys.platform != 'linux' or parent <= 1 or not command:
        raise RuntimeError('multi-node worker lifetime binding requires Linux and a valid parent/command')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), 'cannot bind worker lifetime to launch agent')
    # Close the race where the agent died before prctl established the binding.
    if os.getppid() != parent:
        return 125
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    raise SystemExit(main())
