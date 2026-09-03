"""Rewrite an already-built jittor-blog site for offline, file:// browsing.

This does not build anything in this repository. It post-processes the output
of the separate jittor-blog Jekyll site, which is built with the placeholder
``JITTOR_BASEURL`` as its base URL, and turns every occurrence of that
placeholder into a relative path ending in ``index.html`` -- plus protocol
relative ``//`` links into ``http://`` -- so the tree opens without a server.

It came out of ``python/jittor/utils/`` (task 5.25), where it was shipped
inside the wheel and chdir'd into one machine's home directory on import. The
site directory is now an argument.

    # build the site with the placeholder base URL, then:
    python tools/docs/local_doc_builder.py <site-output-dir>
"""

import os
import sys

BASE_URL_PLACEHOLDER = "JITTOR_BASEURL"

def check(dirname, fname):
    with open(os.path.join(dirname, fname), 'r') as f:
        src = f.read()
    ac = BASE_URL_PLACEHOLDER
    rep = (
        ("href=\"//", "href=\"http://"), 
        ("src=\"//", "src=\"http://"),
        ('https://cg.cs.tsinghua.edu.cn/jittor', ac)
    )
    found = False
    for a,b in rep:
        if a in src:
            src = src.replace(a, b)
            found = True
    if ac not in src and not found: return
    n = len(dirname.split(os.path.sep))-1
    s = '.' + '/..' * n
    new_src = ""
    i = -1
    print("="*20)
    print(dirname, fname)
    while True:
        i += 1
        if i >= len(src):
            break
        if src[i] != 'J':
            new_src += src[i]
            continue
        if src[i:i+len(ac)] != ac:
            new_src += src[i]
            continue
        j = i
        while j<len(src) and src[j] != ' ' and src[j] != '"' and src[j] != "'":
            j += 1
        x = src[i:j]
        y = x.replace(ac, s)
        if '#' in y:
            y, l = y.split('#')
            l = '#'+l
        else:
            l = ""
        # replace xx/xx/ --> xx/xx/index.html
        if y.endswith('/'):
            y += 'index.html'
        else:
            z = y.split('/')[-1]
            # replace xx/xx --> xx/xx/index.html
            if '.' not in z:
                y += '/index.html'
        y += l
        print("found", x, '-->', y)
        new_src += y
        i = j-1
    with open(os.path.join(dirname, fname), 'w') as f:
        f.write(new_src)

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    os.chdir(argv[0])
    for r, _, f in os.walk('.'):
        for fname in f:
            ext = fname.split('.')[-1]
            if ext not in ['html', 'css', 'js']:
                continue
            check(r, fname)
    return 0


if __name__ == "__main__":
    sys.exit(main())

