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

cmds1 = remove(cmds, [".o"])
cmds1 = replace(cmds1, ".so", ".o")
cmds2 = replace(cmds, "-dc", "")
cmds2 = replace(cmds2, ".cu", ".o")
ret = os.system(" ".join(cmds1).replace("-x cu", ""))
if ret: exit(ret)
ret = os.system(" ".join(cmds2).replace("-x cu", ""))
if ret: exit(ret)