# how to run:
# docker run -v "${HOME}/Documents/jittor-blog":/srv/jittor-blog -v /home/jittor/Documents/site:/mnt/jittor-blog -e LC_ALL=C.UTF-8 --rm jittor-blog-compiler bash -c "jekyll build --baseurl=JITTOR_BASEURL -d /mnt/jittor-blog/ && chmod -R 777 /mnt/jittor-blog"
# python /home/jittor/Documents/jittor-blog/local_doc_builder.py

import os

os.chdir("/home/jittor/Documents/site")

def check(dirname, fname):
    with open(os.path.join(dirname, fname), 'r') as f:
        src = f.read()
    ac = "JITTOR_BASEURL"
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

for r, _, f in os.walk('.'):
    for fname in f:
        ext = fname.split('.')[-1]
        if ext not in ['html', 'css', 'js']:
            continue
        # print(r, fname)
        check(r, fname)

