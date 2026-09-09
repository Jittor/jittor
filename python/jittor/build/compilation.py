"""Compiler invocation and cached extension/product construction."""
from jittor_utils import lock


def _source_path(name):
    """Resolve a source named relative to the package, or pass an absolute one.

    `4.15` moved the C++ core out of the Python package to the top level, so a
    name beginning with ``src/`` no longer sits under ``jittor_path``. Routing
    it through ``core_root`` here is why the dozens of relative names in the
    core file lists below did not each have to learn where the core went --
    and why ``jit_utils_core_files`` can stay in the same spelling as the list
    it is later removed from (``files.remove`` matches strings, not paths).
    """
    from . import compiler as _compiler_state
    if _compiler_state.os.path.isabs(name):
        return name
    head = name.replace("\\", "/").split("/", 1)
    if head[0] == "src" and len(head) == 2:
        return _compiler_state.os.path.join(_compiler_state.core_root(_compiler_state.jittor_path), head[1])
    return _compiler_state.os.path.join(_compiler_state.jittor_path, name)


def compile(compiler, flags, inputs, output, combind_build=False, cuda_flags="", obj_dirname="obj_files", return_cmd=False):
    from . import compiler as _compiler_state
    def do_compile(cmd):
        if _compiler_state.jit_utils.cc:
            return _compiler_state.jit_utils.cc.cache_compile(cmd, _compiler_state.cache_path, _compiler_state.jittor_path)
        else:
            _compiler_state.run_cmd(cmd)
            return True
    base_output = _compiler_state.os.path.basename(output).split('.')[0]
    if _compiler_state.os.name == 'nt':
        # windows do not combind build, need gen def
        combind_build = False
        # windows need xxxx.lib
        afile = output.rsplit('.', 1)[0] + ".lib"
        afile = _compiler_state.os.path.join(_compiler_state.cache_path, afile)
        # The MinGW branch that used to live here read a name (`link`) that is
        # defined neither in this function nor at module scope, so executing it
        # raised NameError unconditionally: it has never run. Rather than leave
        # code that cannot work, Windows builds go through cl.
    if not _compiler_state.os.path.isabs(output):
        output = _compiler_state.os.path.join(_compiler_state.cache_path, output)
    # don't recompile object file in inputs
    obj_files = []
    ex_obj_files = []
    new_inputs = []
    obj_dir = _compiler_state.os.path.join(_compiler_state.cache_path, obj_dirname)
    _compiler_state.os.makedirs(obj_dir, exist_ok=True)
    for name in inputs:
        if name[-1] in 'oab':
            ex_obj_files.append(name)
        else:
            new_inputs.append(_compiler_state._source_path(name))
            obj_files.append(_compiler_state.os.path.join(
                obj_dir, _compiler_state.os.path.basename(name)+".o"))
    inputs = new_inputs
    cm = lambda s: f"\"{s}\""
    cms = lambda arr: [f"\"{s}\"" for s in arr ]

    if len(inputs) == 1 or combind_build:
        cmd = f"\"{compiler}\" {' '.join(cms(inputs))} {flags} -o {cm(output)}"
        if return_cmd:
            return _compiler_state.fix_cl_flags(cmd)
        return do_compile(_compiler_state.fix_cl_flags(cmd))
    if return_cmd:
        # The split compile/link path runs several commands; there is no single
        # one to hand back, and silently compiling instead would be worse.
        raise ValueError("return_cmd requires a single-command build")
    # split compile object file and link
    # remove -l -L flags when compile object files
    oflags = _compiler_state.remove_flags(flags, ['-l', '-L', '-Wl,', '.lib', '-shared'])
    cmds = []
    for input, obj_file in zip(inputs, obj_files):
        cc = compiler
        nflags = oflags
        cmd = f"{cm(input)} {nflags} {_compiler_state.lto_flags} -c -o {cm(obj_file)}"
        if input.endswith(".cu"):
            if _compiler_state.has_cuda or _compiler_state.has_rocm:
                cmd = f"\"{_compiler_state.nvcc_path}\" {cuda_flags} {cmd}"
                cmd = _compiler_state.convert_nvcc_flags(_compiler_state.fix_cl_flags(cmd))
            else:
                continue
        else:
            cmd = f"\"{cc}\" {cmd}"
            cmd = _compiler_state.fix_cl_flags(cmd)
        if "nan_checker" in input:
            # nan checker needs to disable fast_math
            if "--use_fast_math" in cmd:
                cmd = cmd.replace("--use_fast_math", "")
            if "-Ofast" in cmd:
                cmd = cmd.replace("-Ofast", "-O2")
        cmds.append(cmd)
    _compiler_state.jit_utils.run_cmds(cmds, _compiler_state.cache_path, _compiler_state.jittor_path, "Compiling "+base_output)
    obj_files += ex_obj_files
    if _compiler_state.os.name == 'nt':
        # dumpdef is a Windows build resource and ships with the build package.
        # Keep the lookup relative to the installed package so wheel builds do
        # not depend on repository-only tools/ files.
        dumpdef_path = _compiler_state.os.path.join(_compiler_state.jittor_path, "build", "dumpdef.py")
        cmd = f"\"{_compiler_state.sys.executable}\" \"{dumpdef_path}\" {' '.join(cms(obj_files))} -Fo: \"{output}.def\""
        do_compile(_compiler_state.fix_cl_flags(cmd))
    cmd = f"\"{compiler}\" {' '.join(cms(obj_files))} -o {cm(output)} {flags} {_compiler_state.lto_flags}"
    return do_compile(_compiler_state.fix_cl_flags(cmd))


@lock.lock_scope()
def compile_custom_op(header, source, op_name, warp=True):
    """Compile a single custom op
    header: code of op header, not path
    source: code of op source, not path
    op_name: op_name of this op, it will used for
        generation of header and source files, if the
        type name of op is XxxXxxOp, op_name should be
        xxx_xxx
    warp: if true, warp a snippet for header and source
    """
    from . import compiler as _compiler_state
    if warp:
        header = f"""
        #pragma once
        #include "core/op.h"
        #include "core/var.h"
        namespace jittor {{
        {header}
        }}
        """
        source = f"""
        #include "{op_name}_op.h"
        namespace jittor {{
        {source}
        }}
        """
    cops_dir = _compiler_state.os.path.join(_compiler_state.cache_path, "custom_ops")
    _compiler_state.make_cache_dir(cops_dir)
    hname = _compiler_state.os.path.join(cops_dir, op_name+"_op.h")
    ccname = _compiler_state.os.path.join(cops_dir, op_name+"_op.cc")
    with open(hname, 'w', encoding='utf8') as f:
        f.write(header)
    with open(ccname, 'w', encoding='utf8') as f:
        f.write(source)
    m = _compiler_state.compile_custom_ops([hname, ccname])
    return getattr(m, op_name)


@lock.lock_scope()
def compile_custom_ops(
    filenames,
    extra_flags="",
    return_module=False,
    dlopen_flags=None,
    gen_name_ = "",
    backend=None):
    """Compile custom ops
    filenames: path of op source files, filenames must be
        pairs of xxx_xxx_op.cc and xxx_xxx_op.h, and the
        type name of op must be XxxXxxOp.
    extra_flags: extra compile flags
    return_module: return module rather than ops(default: False)
    backend: None uses the op class declaration; cpu, accelerator, or both
        explicitly selects the registered backend family for this library.
    return: compiled ops
    """
    from . import compiler as _compiler_state
    if dlopen_flags is None:
        dlopen_flags = _compiler_state.os.RTLD_GLOBAL | _compiler_state.os.RTLD_NOW
        if _compiler_state.platform.system() == 'Linux':
            dlopen_flags |= _compiler_state.os.RTLD_DEEPBIND

    srcs = {}
    headers = {}
    builds = []
    includes = []
    pyjt_includes = []
    for name in filenames:
        name = _compiler_state.os.path.realpath(name)
        if name.endswith(".cc") or name.endswith(".cpp") or name.endswith(".cu"):
            builds.append(name)
        if name.endswith(".h"):
            dirname = _compiler_state.os.path.dirname(name)
            if dirname.endswith("inc"):
                includes.append(dirname)
            with open(name, "r", encoding='utf8') as f:
                if "@pyjt" in f.read():
                    pyjt_includes.append(name)
        bname = _compiler_state.os.path.basename(name)
        bname = _compiler_state.os.path.splitext(bname)[0]
        if bname.endswith("_op"):
            bname = bname[:-3]
            if name.endswith(".cc"):
                srcs[bname] = name
            elif name.endswith(".h"):
                includes.append(_compiler_state.os.path.dirname(name))
                headers[bname] = name
    assert len(srcs) == len(headers), "Source and header names not match"
    for name in srcs:
        assert name in headers, f"Header of op {name} not found"
    gen_name = "gen_ops_" + "_".join(headers.keys())
    if gen_name_ != "":
        gen_name = gen_name_
    if len(gen_name) > 50:
        gen_name = gen_name[:50] + "___hash" + _compiler_state.hashlib.md5(gen_name.encode()).hexdigest()[:6]

    include_dirs = sorted(list(set(includes)))
    include_flags = "".join(map(lambda x: f" -I\"{x}\" ", include_dirs))
    _compiler_state.LOG.vvvv(f"Include flags:{include_flags}")

    op_extra_flags = include_flags + extra_flags

    lib_path = _compiler_state.os.path.join(_compiler_state.cache_path, "custom_ops")
    _compiler_state.make_cache_dir(lib_path)
    gen_src_fname = _compiler_state.os.path.join(lib_path, gen_name+".cc")
    gen_head_fname = _compiler_state.os.path.join(lib_path, gen_name+".h")
    gen_lib = _compiler_state.os.path.join(lib_path, gen_name+_compiler_state.extension_suffix)
    libname = gen_name + _compiler_state.lib_suffix
    op_extra_flags += f" -L\"{lib_path}\" -l\"{libname}\" "

    # Everything below -- generating the op maker, running pyjt over the
    # headers, and handing the translation units to the compile pool -- is
    # skipped when the stamp says this library was already built from exactly
    # these inputs. See :func:`product_build_is_current` for why the cheap
    # answer is worth having.
    stamp_sources = {
        "files": _compiler_state._stat_signature([_compiler_state.os.path.realpath(name)
                                  for name in filenames]),
        "includes": _compiler_state._include_tree_signature(_compiler_state._custom_op_include_dirs(
            filenames, include_dirs, extra_flags)),
        "core_sources": _compiler_state.core_source_signature(),
    }
    stamp_ingredients = _compiler_state.custom_op_build_ingredients(
        gen_name, extra_flags, includes, backend)
    if _compiler_state.product_build_is_current(gen_lib, stamp_sources, stamp_ingredients):
        _compiler_state.LOG.v(f"custom op lib {gen_name} is current, skipping build")
        return _compiler_state._import_custom_op_lib(gen_name, dlopen_flags, return_module)
    if not _compiler_state.build_is_allowed():
        _compiler_state._refuse_build("the custom op library " + gen_name)

    gen_src = _compiler_state.gen_jit_op_maker(headers.values(), export=gen_name,
                             extra_flags=op_extra_flags, backend=backend)
    _compiler_state.pyjt_compiler.compile_single(gen_head_fname, gen_src_fname, src=gen_src)
    # gen src initialize first
    builds.insert(0, gen_src_fname)

    def insert_anchor(gen_src, anchor_str, insert_str):
        # insert insert_str after anchor_str into gen_src
        return gen_src.replace(anchor_str, anchor_str+insert_str, 1)

    for name in pyjt_includes:
        _compiler_state.LOG.v("handle pyjt_include ", name)
        bname = _compiler_state.os.path.basename(name).split(".")[0]
        gen_src_fname = _compiler_state.os.path.join(_compiler_state.cache_path, "custom_ops", gen_name+"_"+bname+".cc")
        _compiler_state.pyjt_compiler.compile_single(name, gen_src_fname)
        builds.insert(1, gen_src_fname)
        gen_src = insert_anchor(gen_src,
            "namespace jittor {",
            f"extern void pyjt_def_{bname}(PyObject* m);")
        gen_src = insert_anchor(gen_src,
            "init_module(PyModuleDef* mdef, PyObject* m) {",
            f"jittor::pyjt_def_{bname}(m);")

    with open(gen_head_fname, "w", encoding='utf8') as f:
        f.write(gen_src)

    _compiler_state.LOG.vvv(f"Build custum ops lib:{gen_lib}")
    _compiler_state.LOG.vvvv(f"Build sources:{builds}")
    _compiler_state.compile(_compiler_state.cc_path, extra_flags+_compiler_state.cc_flags+_compiler_state.opt_flags+include_flags, builds, gen_lib)

    # After the compile, so the stamp records the product that exists now. The
    # source signature is the pre-build one on purpose: a source edited during
    # the build must leave the stamp stale rather than claim to cover the edit.
    _compiler_state._write_product_build_stamp(gen_lib, stamp_sources, stamp_ingredients)

    return _compiler_state._import_custom_op_lib(gen_name, dlopen_flags, return_module)


def _import_custom_op_lib(gen_name, dlopen_flags, return_module):
    """dlopen a built custom-op library and hand back its ops."""
    from . import compiler as _compiler_state
    _compiler_state.LOG.vvv(f"Import custum ops lib:{gen_name}")
    lib_path = _compiler_state.os.path.join(_compiler_state.cache_path, "custom_ops")
    if lib_path not in _compiler_state.os.sys.path:
        _compiler_state.os.sys.path.append(lib_path)
    # unlock scope when initialize
    # NOTE: use __import__ (returns the module object) instead of
    # `exec("import X"); locals()["X"]`. Since Python 3.13 (PEP 667) exec() no
    # longer writes back into an optimized function's locals(), so the old
    # pattern raised KeyError and broke `import jittor` from source on 3.13.
    with _compiler_state.lock.unlock_scope():
        with _compiler_state.jit_utils.import_scope(dlopen_flags):
            mod = __import__(gen_name)
    if return_module:
        return mod
    return mod.ops


def product_build_stamp_path(product):
    """Where the stamp for a build product lives: right next to it.

    Beside the product rather than in an index, so that deleting the product
    -- by hand, or with ``jittor_utils.clean_cache`` -- cannot leave a stamp
    behind that claims it is still there.
    """
    return product + ".build_stamp.json"


def _stat_signature(paths):
    """``{path: [mtime_ns, size]}``, with ``None`` for a path that is gone.

    A missing path is recorded explicitly rather than skipped: dropping it
    would make "the file was deleted" indistinguishable from "the file was
    never listed", and the second one must not be treated as up to date.
    """
    from . import compiler as _compiler_state
    from typing import Dict, List, Optional, Union
    record: Dict[str, Optional[Union[List[int], None]]] = {}
    for path in paths:
        try:
            info = _compiler_state.os.stat(path)
        except OSError:
            record[path] = None
            continue
        record[path] = [info.st_mtime_ns, info.st_size]
    return record


def _is_within(path, root):
    from . import compiler as _compiler_state
    return path == root or path.startswith(root + _compiler_state.os.sep)


def _include_tree_signature(directories):
    """Stat every source and header under each of the caller's include dirs.

    ``filenames`` names the op sources and their paired headers; it does not
    name the sibling headers those include. The per-file dependency scan
    inside :func:`compile` does find them -- and that scan is exactly what the
    stamp exists to skip, so the stamp has to cover them another way.

    Two kinds of directory are deliberately not walked, because walking them
    costs more than it can ever detect:

    * anything inside the Jittor tree -- :func:`core_source_signature` already
      covers it, including ``backends/*/libraries/*/include``;
    * the CUDA SDK's include directories. ``cache_path`` is partitioned by the
      cuda key, so a different toolkit builds into a different directory and
      gets a cold build rather than a stale one. This is the same argument
      :func:`core_source_signature` makes for external headers -- and it is
      what keeps this off the import path: those trees hold ~1400 headers
      each, and walking six of them cost 90 ms of every warm CUDA import.

    What is left is the caller's own directories, which is the case that
    matters for a custom op built outside this repository.
    """
    from . import compiler as _compiler_state
    signature = {}
    tree = _compiler_state.os.path.realpath(_compiler_state.jittor_path)
    toolkit = [_compiler_state.os.path.realpath(path) for path in _compiler_state.cuda_include_dirs]
    for directory in sorted(set(directories)):
        if not directory:
            continue
        real = _compiler_state.os.path.realpath(directory)
        if _compiler_state._is_within(real, tree):
            continue
        if any(_compiler_state._is_within(real, path) or _compiler_state._is_within(path, real)
               for path in toolkit):
            continue
        for current, _, names in _compiler_state.os.walk(real):
            for name in names:
                if not name.endswith(_compiler_state._SOURCE_EXTENSIONS):
                    continue
                path = _compiler_state.os.path.join(current, name)
                try:
                    info = _compiler_state.os.stat(path)
                except OSError:
                    continue
                signature[_compiler_state.os.path.relpath(path, real)] = \
                    [info.st_mtime_ns, info.st_size]
    return signature


def _custom_op_include_dirs(filenames, include_dirs, extra_flags):
    """Every directory the custom-op compile can reach a header through.

    The auto-derived ones, the directories the named files live in, and the
    ``-I`` the caller passed by hand in ``extra_flags`` -- that last one is
    the only way a public caller can point the compile at a tree Jittor knows
    nothing about, so parsing it out is what keeps the stamp honest for code
    outside this repository.
    """
    from . import compiler as _compiler_state
    directories = list(include_dirs)
    directories += [_compiler_state.os.path.dirname(_compiler_state.os.path.realpath(name))
                    for name in filenames]
    for match in _compiler_state.re.finditer(r'-I\s*(?:"([^"]*)"|(\S+))', extra_flags or ""):
        directories.append(match.group(1) or match.group(2))
    return directories


def custom_op_build_ingredients(gen_name, extra_flags, include_flags, backend):
    """Everything besides the sources that the compile commands use.

    ``cc_flags`` is safe to record verbatim here, unlike for the core: by the
    time any custom op is built the core is linked and ``-ljittor_core`` is
    already on it, so re-reading it after the import yields the same string
    the build used.
    """
    from . import compiler as _compiler_state
    return {
        "version": _compiler_state.__version__,
        "gen_name": gen_name,
        "cc_path": _compiler_state.cc_path,
        "cc_type": _compiler_state.cc_type,
        "cc_flags": _compiler_state.cc_flags,
        "opt_flags": _compiler_state.opt_flags,
        "extra_flags": extra_flags,
        "include_flags": include_flags,
        "backend": backend,
        "nvcc_path": _compiler_state.nvcc_path,
        "nvcc_flags": _compiler_state.nvcc_flags,
        "extension_suffix": _compiler_state.extension_suffix,
        "lib_suffix": _compiler_state.lib_suffix,
        "has_cuda": int(bool(_compiler_state.has_cuda)),
        "has_accelerator": int(bool(_compiler_state.has_accelerator)),
        "core_output": _compiler_state._core_output_signature(),
        "generators": _compiler_state.core_generator_signature(),
    }


def product_build_is_current(product, sources, ingredients):
    """True when ``product`` was already built from exactly these inputs.

    Same bargain as the core's stamp, for the same reason: finding out the
    expensive way cost about 0.35 s of every warm CUDA import, because the
    bundled cuDNN/cuBLAS/cuRAND/cuFFT/cuSPARSE/CUB op libraries handed ~50
    compile commands to the pool so each worker could hash a dependency
    closure and report that there was nothing to do.

    Conservative in one direction only. A mismatch means "build", and
    building re-runs the full per-file dependency check that was always there,
    so a stamp that wrongly reports "stale" costs time and nothing else.

    The callers decide what goes in ``sources`` and ``ingredients``; this only
    promises that a build is current when *all* of it matches. Getting those
    two right is the whole safety argument, because the direction that is not
    safe -- claiming current when it is not -- does not fail, it silently
    keeps running the previous build.
    """
    from . import compiler as _compiler_state
    try:
        with open(_compiler_state.product_build_stamp_path(product), "r",
                  encoding="utf8") as handle:
            stamp = _compiler_state.json.load(handle)
    except (OSError, ValueError):
        return False
    if not isinstance(stamp, dict):
        return False
    if stamp.get("stamp_version") != _compiler_state.PRODUCT_BUILD_STAMP_VERSION:
        return False
    if stamp.get("output") != _compiler_state._stat_signature([product]).get(product):
        return False
    if stamp.get("ingredients") != ingredients:
        return False
    if stamp.get("sources") != sources:
        return False
    return True


def _write_product_build_stamp(product, sources, ingredients):
    """Record what a build product was just built from.

    Written to a temporary name and renamed, so a reader arriving mid-write
    sees either the old stamp or the new one, never a truncated file.
    """
    from . import compiler as _compiler_state
    stamp = {
        "stamp_version": _compiler_state.PRODUCT_BUILD_STAMP_VERSION,
        "output": _compiler_state._stat_signature([product]).get(product),
        "ingredients": ingredients,
        "sources": sources,
    }
    if stamp["output"] is None:
        return
    path = _compiler_state.product_build_stamp_path(product)
    temporary = path + ".tmp." + str(_compiler_state.os.getpid())
    try:
        with open(temporary, "w", encoding="utf8") as handle:
            _compiler_state.json.dump(stamp, handle)
        _compiler_state.os.replace(temporary, path)
    except OSError as error:
        # A read-only or full cache directory must not fail a build that
        # otherwise succeeded; the cost is one expensive check next time.
        _compiler_state.LOG.v("could not write build stamp for %s: %s" % (path, error))
        try:
            _compiler_state.os.remove(temporary)
        except OSError:
            pass


def compile_if_stale(what, compiler_path, flags, sources, output,
                     extra_ingredients=None):
    """:func:`compile`, skipped when the stamp says ``output`` is current.

    For the products that are neither the core nor a custom op library but
    still sit on the import path. ``libcuda_extern`` is the whole reason this
    exists: two translation units, nothing to do on a warm import, and 0.073 s
    of every one of them spent proving it -- the last compile fan-out left on
    the import path once the core and the op libraries had stamps.

    Returns True if it built.
    """
    from . import compiler as _compiler_state
    stamp_sources = {
        "files": _compiler_state._stat_signature([_compiler_state.os.path.realpath(name) for name in sources]),
        "core_sources": _compiler_state.core_source_signature(),
    }
    ingredients = {
        "version": _compiler_state.__version__,
        "what": what,
        "cc_path": compiler_path,
        "flags": flags,
        "opt_flags": _compiler_state.opt_flags,
        "extension_suffix": _compiler_state.extension_suffix,
        "generators": _compiler_state.core_generator_signature(),
    }
    if extra_ingredients:
        ingredients.update(extra_ingredients)
    if _compiler_state.product_build_is_current(output, stamp_sources, ingredients):
        _compiler_state.LOG.v("%s is current, skipping build" % what)
        return False
    if not _compiler_state.build_is_allowed():
        _compiler_state._refuse_build(what)
    _compiler_state.compile(compiler_path, flags, sources, output)
    _compiler_state._write_product_build_stamp(output, stamp_sources, ingredients)
    return True
