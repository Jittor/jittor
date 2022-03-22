#!python3
import os, json
import jittor_utils as jit_utils
notebook_dir = os.path.join(jit_utils.home(), ".cache","jittor","notebook")
if not os.path.isdir(notebook_dir):
    os.mkdir(notebook_dir)
dirname = os.path.dirname(__file__)
all_md = []
for r, _, f in os.walk(dirname):
    for fname in f:
        if not fname.endswith(".md"): continue
        all_md.append(os.path.join(r, fname))
for mdname in all_md:
    with open(os.path.join(dirname, mdname), "r", encoding="utf-8") as f:
        src = f.read()
    blocks = []
    for i, b in enumerate(src.split("```")):
        b = b.strip()
        is_markdown_block = i%2==0
        if not is_markdown_block and not b.startswith("python"):
            is_markdown_block = True
            b = "```\n"+b+"\n```"
        if is_markdown_block:
            # in a markdown block
            if len(blocks)%2==0:
                # prev code block
                blocks.append(b)
            else:
                # prev markdown block
                blocks[-1] += "\n\n" + b
        else:
            # in a code block
            if b.startswith("python"):
                b = b[6:].strip()
                # prev markdown block
                assert len(blocks)%2==1
                blocks.append(b)
    cells = []
    for i, b in enumerate(blocks):
        b = b.strip()
        if len(b)==0: continue
        b = b.split("\n")
        for j in range(len(b)-1):
            b[j] += '\n'
        cell = {
            "source": b,
            "metadata": {},
        }
        if i%2==0:
            cell["cell_type"] = "markdown"
        else:
            cell["cell_type"] = "code"
            cell["outputs"] = []
            cell["execution_count"] = 0
        cells.append(cell)
    ipynb = {
        "cells":cells,
        "nbformat": 4,
        "nbformat_minor": 2,
        "metadata": {
        },
    }
    ipynb_name = os.path.basename(mdname[:-2])+"ipynb"
    ipynb_name = os.path.join(notebook_dir, ipynb_name)
    print(mdname, len(src), len(blocks), len(cells), "--->", ipynb_name)
    with open(ipynb_name, "w", encoding='utf8') as f:
        f.write(json.dumps(ipynb))