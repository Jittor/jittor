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
    if src.startswith("!"):
        return True
    return len(src) < 2

def splite_markdown_blocks(src):
    ''' split markdown document into text, code, table blocks
    '''
    blocks = []
    block = ""
    status = "text"

    def commit_block():
        blocks.append((block, status))

    for line in src.split('\n'):
        line = line + "\n"
        if line.startswith("```"):
            assert status in ["text", "code"]
            if status == "text":
                commit_block()
                status = "code"
                block = line
            elif status == "code":
                block += line
                commit_block()
                status = "text"
                block = ""
        elif line.strip().startswith('|') and line.strip().endswith('|'):
            assert status in ["text", "table"]
            if status == "text":
                commit_block()
                status = "table"
                block = line
            else:
                block += line
        else:
            if status == "table":
                commit_block()
                status = "text"
                block = line
            else:
                block += line
    if status != "code":
        commit_block()
    return blocks

for mdname in all_src_md:
    print(mdname)
    with open(mdname, "r", encoding='utf8') as f:
        src = f.read()

    src_blocks = splite_markdown_blocks(src)

    en_src = ""
    cn_src = ""
    for block, status in src_blocks:
        if status == "code" or status == "table":
            en_src += block
            cn_src += block
        else:
            en_s = []
            cn_s = []
            for line in block.split('\n'):
                if check_is_both(line):
                    en_s.append(line)
                    cn_s.append(line)
                elif check_is_en(line):
                    en_s.append(line)
                else:
                    cn_s.append(line)
            en_src += "\n".join(en_s)
            cn_src += "\n".join(cn_s)
    
    with open(mdname.replace(".src.md", ".md"), 'w', encoding='utf8') as f:
        f.write(en_src)
    with open(mdname.replace(".src.md", ".cn.md"), 'w', encoding='utf8') as f:
        f.write(cn_src)
                    