#!python3
import os, json
from pathlib import Path
dirname = os.path.dirname(__file__)

jittor_root = os.path.join(dirname, "..", "..")
print(jittor_root)

all_src_md = []

for r, _, f in os.walk(jittor_root):
    for fname in f:
        if not fname.endswith(".src.md"): continue
        all_src_md.append(os.path.realpath(os.path.join(r, fname)))

def check_is_en(src):
    en_cnt = 0
    for c in src: en_cnt += str.isascii(c)
    return en_cnt == len(src)

def check_is_both(src):
    return len(src) < 2

for mdname in all_src_md:
    print(mdname)
    with open(mdname, "r") as f:
        src = f.read()
    src = src.split("```")
    en_src = []
    cn_src = []
    for i, s in enumerate(src):
        if i%2==1:
            en_src.append(s)
            cn_src.append(s)
        else:
            en_s = []
            cn_s = []
            for line in s.split('\n'):
                if check_is_both(line):
                    en_s.append(line)
                    cn_s.append(line)
                elif check_is_en(line):
                    en_s.append(line)
                else:
                    cn_s.append(line)
            en_src.append("\n".join(en_s))
            cn_src.append("\n".join(cn_s))
    en_src = "```".join(en_src)
    cn_src = "```".join(cn_src)
    with open(mdname.replace(".src.md", ".md"), 'w') as f:
        f.write(en_src)
    with open(mdname.replace(".src.md", ".cn.md"), 'w') as f:
        f.write(cn_src)
                    