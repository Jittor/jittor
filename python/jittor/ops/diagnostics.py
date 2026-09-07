"""Diagnostics tensor operations."""


def print_tree(now, max_memory_size, prefix1, prefix2, build_by):
    import jittor as jt
    def format_size(s, end='B'):
        if (s < 1024):
            s = str(s)
            return s + ' '+end

        if (s < 1024*1024):
            s = format(s/1024, '.2f')
            return s + ' K'+end

        if (s < 1024*1024*1024):
            s = format(s/1024/1024, '.2f')
            return s + ' M'+end

        s = format(s/1024/1024/1024, '.2f')
        return s + ' G'+end

    out = ''
    tab = '   '
    out += prefix1+now['name']+'('+now['type']+')\n'
    out += prefix2+'['+format_size(now['size'])+'; '+format(now['size']/max_memory_size*100, '.2f')+'%; cnt:'+format_size(now['cnt'],'') + ']\n'
    if len(now['children']) == 0 and len(now['vinfo']):
        out += prefix2+now['vinfo'][0]
        if len(now['vinfo']) > 1: out += "..."
        out += '\n'
    if (build_by == 0):
        for p in now['path']:
            out += prefix2+p+'\n'
    else:
        out += prefix2+now['path'] + '\n'
    if (len(now['children']) > 0):
        out += prefix2 + tab + '| ' + '\n'
    else:
        out += prefix2 + '\n'
    for i in range(len(now['children'])):
        c = now['children'][i]
        if i < len(now['children']) - 1:
            prefix1_ = prefix2 + tab + '├─'
            prefix2_ = prefix2 + tab + '| '
        else:
            prefix1_ = prefix2 + tab + '└─'
            prefix2_ = prefix2 + tab + '  '
        out += jt.misc.print_tree(
            c, max_memory_size, prefix1_, prefix2_, build_by,
        )
    return out


def get_max_memory_treemap(build_by=0, do_print=True):
    '''show treemap of max memory consumption

Example::

        net = jt.models.resnet18()
        with jt.flag_scope(trace_py_var=3, profile_memory_enable=1):
            imgs = jt.randn((1,3,224,224))
            net(imgs).sync()
            jt.get_max_memory_treemap()

Output::

    |
    ├─./tests/compiler/test_memory_profiler.py:100(test_sample)
    | [19.03 MB; 29.67%]
    | ./tests/compiler/test_memory_profiler.py:100
    |    |
    |    └─./python/jittor/__init__.py:730(__call__)
    |      [19.03 MB; 29.67%]
    |      ./python/jittor/__init__.py:730
    |         |
    |         └─./python/jittor/models/resnet.py:152(execute)
    |           [19.03 MB; 29.67%]
    |           ./python/jittor/models/resnet.py:152
    |              |
    |              ├─./python/jittor/models/resnet.py:142(_forward_impl)
    |              | [6.13 MB; 9.55%]
    |              | ./python/jittor/models/resnet.py:142
    |              |    |



    '''
    import jittor as jt
    div1 = "[!@#div1!@#]"
    div2 = "[!@#div2!@#]"
    div3 = "[!@#div3!@#]"
    info = jt.get_max_memory_info()

    vars = []
    vars_ = info.split(div1)
    max_memory_size = int(vars_[0])
    vars_ = vars_[1:]
    for v_ in vars_:
        v__ = v_.split(div2)
        vinfo = v__[0].split("{")[0]
        var = {'size':int(v__[1]), 'stack':[], 'cnt':1, "vinfo":vinfo}
        v__ = v__[2:-1]
        for s_ in v__:
            s__ = s_.split(div3)
            s = {'path':s__[0], 'name':s__[1], 'type':s__[2]}
            var['stack'].append(s)
        vars.append(var)
    if (build_by == 0): # build tree by name
        tree = {'name':'root', "children":[], 'size':0, 'cnt':1, 'path':[], 'type':'', 'vinfo':[]}

        def find_child(now, key):
            for c in now['children']:
                if (c['name'] == key):
                    return c
            return None
        for v in vars:
            now = tree
            now['size'] += v['size']
            now['cnt'] += v['cnt']
            now['vinfo'].append(v['vinfo'])
            for s in v['stack']:
                ch = find_child(now, s['name'])
                if (ch is not None):
                    if (not s['path'] in ch['path']):
                        ch['path'].append(s['path'])
                    assert(ch['type']==s['type'])
                    now = ch
                    now['size'] += v['size']
                    now['cnt'] += v['cnt']
                    now['vinfo'].append(v['vinfo'])
                else:
                    now_ = {'name':s['name'], "children":[], 'size':v['size'], 'cnt':v['cnt'], 'path':[s['path']], 'type':s['type'], 'vinfo':[v['vinfo']]}
                    now['children'].append(now_)
                    now = now_
    elif (build_by == 1): # build tree by path
        tree = {'name':'root', "children":[], 'size':0, 'cnt':0, 'path':'_root_', 'type':'', 'vinfo':[]}

        def find_child(now, key):
            for c in now['children']:
                if (c['path'] == key):
                    return c
            return None
        for v in vars:
            now = tree
            now['size'] += v['size']
            now['cnt'] += v['cnt']
            now['vinfo'].append(v['vinfo'])
            for s in v['stack']:
                ch = find_child(now, s['path'])
                if (ch is not None):
                    now = ch
                    now['size'] += v['size']
                    now['cnt'] += v['cnt']
                    now['vinfo'].append(v['vinfo'])
                else:
                    now_ = {'name':s['name'], "children":[], 'size':v['size'],  'cnt':v['cnt'], 'path':s['path'], 'type':s['type'], 'vinfo':[v['vinfo']]}
                    now['children'].append(now_)
                    now = now_
    else:
        assert(False)

    def sort_tree(now):
        def takeSize(elem):
            return elem['size']
        now['children'].sort(key=takeSize, reverse=True)
        for c in now['children']:
            sort_tree(c)
    sort_tree(tree)
    out = jt.misc.print_tree(tree, max_memory_size, '', '', build_by)
    if (do_print):
        print(out)
    return tree, out
