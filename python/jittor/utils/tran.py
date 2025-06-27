# usage: PYTHONPATH=/PATH/TO/JITTOR python tran.py
import re
import os

path = os.environ.get("PYTHONPATH")

def walk_dir(dir_name):
    for abname in os.listdir(dir_name):
        nfn = os.path.join(dir_name,abname)
        if os.is_dir(nfn):
            walk_dir(nfn)
            continue
        if not nfn.endswith(".py"): continue
        f = open(nfn,"r+",encoding="utf-8")
        startline = f.tell()
        content = f.read()
        loc_lang = locale.getdefaultlocale()
        if loc_lang[0] == "zh-CN":
            content = re.sub('\'\'\'(.*?)@@@', '\'\'\'', content,  flags=re.DOTALL)
        elif loc_lang[0] == "en-US":
            content = re.sub('@@@(.*?)\'\'\'', '\'\'\'', content,  flags=re.DOTALL)
        f.seek(startline)
        f.truncate(0)
        f.write(content)
        f.close()


if __name__ == "__main__":
    walk_dir(path)
