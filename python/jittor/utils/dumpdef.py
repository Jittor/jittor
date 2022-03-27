import os
import sys
import subprocess as sp

def_path = sys.argv[-1]

# print(sys.argv)
dumpbin_path = os.environ.get("dumpbin_path", "dumpbin")
export_all = os.environ.get("EXPORT_ALL", "0")=="1"

syms = {}

for obj in sys.argv[1:-2]:
    cmd = f'"{dumpbin_path}" -SYMBOLS "{obj}"'
    ret = sp.getoutput(cmd)
    # print(ret)
    for l in ret.splitlines():
        if '|' in l:
            if "UNDEF" in l: continue
            if "External" not in l: continue
            sym = l.split('|')[1].strip().split()[0]
            if sym[0] in '@.': continue
            if sym.startswith("??$get_from_env"): syms[sym] = 1
            # if sym.startswith("??"): continue
            if sym.startswith("my"): syms[sym] = 1
            # for cutt
            if "custom_cuda" in sym: syms[sym] = 1
            if "cutt" in sym: syms[sym] = 1
            if "_cudaGetErrorEnum" in sym: syms[sym] = 1
            if export_all: syms[sym] = 1
            if "jittor" not in sym: continue
            syms[sym] = 1
    # print(ret)
libname = os.path.basename(def_path).rsplit(".", 1)[0]
src = f"LIBRARY {libname}\nEXPORTS\n"
for k in syms:
    src += f"    {k}\n"
# print(src)

with open(def_path, "w", encoding="utf8") as f:
    f.write(src)
