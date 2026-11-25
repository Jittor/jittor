import sys
import os
import shlex
cmds = sys.argv[1:]
def replace(cmds, s, t):
    return [ c.replace(s,t) for c in cmds ]
def remove(cmds, ss):
    rets = []
    for cmd in cmds:
        found = True
        for s in ss:
            if s in cmd:
                found = False
                break
        if found:
            rets.append(cmd)
    return rets

def remove_dependency_flags(cmds):
    rets = []
    skip = False
    for cmd in cmds:
        if skip:
            skip = False
            continue
        if cmd in ("-MD", "-MMD"):
            continue
        if cmd == "-MF":
            skip = True
            continue
        rets.append(cmd)
    return rets

def remove_x_cu(cmds):
    rets = []
    i = 0
    while i < len(cmds):
        if cmds[i] == "-x" and i + 1 < len(cmds) and cmds[i + 1] == "cu":
            i += 2
            continue
        rets.append(cmds[i])
        i += 1
    return rets

def join_shell_args(args):
    # Preserve tokens with shell-special characters like $ORIGIN or spaces
    # when forwarding through os.system().
    return " ".join(shlex.quote(a) for a in args)

output_path = cmds[cmds.index("-o") + 1]
cuda_path = next(cmd for cmd in cmds if cmd.endswith(".cu"))
object_path = cuda_path[:-3] + ".o"
cmds1 = remove_x_cu(remove(cmds, [".o"]))
cmds1 = replace(cmds1, output_path, object_path)
cmds2 = remove_x_cu(replace(cmds, "-dc", ""))
cmds2 = replace(cmds2, ".cu", ".o")
cmds2 = remove_dependency_flags(cmds2)
ret = os.system(join_shell_args(cmds1))
if ret:
    exit(ret)
ret = os.system(join_shell_args(cmds2))
if ret:
    exit(ret)
