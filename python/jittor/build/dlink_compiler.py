import sys
import os
import re
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

output_path = cmds[cmds.index("-o") + 1]
cuda_path = next(cmd for cmd in cmds if cmd.endswith(".cu"))
object_path = cuda_path[:-3] + ".o"
cmds1 = remove(cmds, [".o"])
cmds1 = replace(cmds1, output_path, object_path)
cmds2 = replace(cmds, "-dc", "")
cmds2 = replace(cmds2, ".cu", ".o")
cmds2 = remove_dependency_flags(cmds2)
ret = os.system(" ".join(cmds1).replace("-x cu", ""))
if ret: exit(ret)
ret = os.system(" ".join(cmds2).replace("-x cu", ""))
if ret: exit(ret)
