# Cached `jittor_core` runtime path (2026-09-09)

The repository does not ship a `jittor_core` Python stub or source module. A
matching CUDA extension is present in the local build cache:

```
/home/zy/.cache/jittor/jt1.3.11/g++12.3.0/py3.11.15/Linux-6.8.0-13xb9/arch0a30a5949a/353bb79d8e09/default/cfgd436a523/cu12.2.140_pipcu122_b9c03cb2d2fb_sm_89/jittor_core.cpython-311-x86_64-linux-gnu.so
```

The minimal extension import succeeds when its directory is first on
`PYTHONPATH` and libgomp is preloaded:

```bash
CORE_DIR=/home/zy/.cache/jittor/jt1.3.11/g++12.3.0/py3.11.15/Linux-6.8.0-13xb9/arch0a30a5949a/353bb79d8e09/default/cfgd436a523/cu12.2.140_pipcu122_b9c03cb2d2fb_sm_89
LD_PRELOAD=/lib/x86_64-linux-gnu/libgomp.so.1 \
PYTHONPATH="$CORE_DIR:python" python -c 'import jittor_core; print(hasattr(jittor_core, "Var"))'
```

Importing the full Jittor package with the same environment reaches CUDA
initialisation, detects architecture 89, and succeeds. The earlier
`TypeError: can only concatenate str (not "list") to str` came from
`python/jittor/build/compilation.py:267`, where the include directory list was
passed into the compiler after the typing rename introduced `include_flags`.
That repository regression was fixed in `b9d3c1cab`; the same command now
prints `jittor ok`.

The CPU cached extension is intentionally not a substitute for this command:
when CUDA is discoverable, Jittor rejects a CPU `jittor_core` as a build shadow.
