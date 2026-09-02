#!/usr/bin/env bash
# Four scenarios that a change to Jittor's build/cache/lock machinery has to
# survive. Everything it needs comes from the environment, so it is safe to
# run from any checkout:
#
#   JITTOR_SRC   checkout to test          (default: $PWD)
#   JITTOR_HOME  private cache root        (required; it gets deleted)
#   PYTHON       interpreter               (default: python3)
#   NVCC         nvcc path for the CUDA rounds (default: found on PATH)
#   JOBS         concurrent importers in scenario 3 (default: 4)
#
# Exit status is 0 only if all four scenarios pass. Read the log it names.
set -u

SRC=${JITTOR_SRC:-$PWD}
PY=${PYTHON:-python3}
NVCC=${NVCC:-$(command -v nvcc || true)}
JOBS=${JOBS:-4}
: "${JITTOR_HOME:?set JITTOR_HOME to a private cache directory; it is deleted}"
: "${TMPDIR:=$JITTOR_HOME/tmp}"
export TMPDIR
LOGDIR=${LOGDIR:-$TMPDIR/verify-build}
mkdir -p "$LOGDIR"

fail=0
step() { printf '\n=== %s ===\n' "$*"; }
bad()  { printf 'FAIL: %s\n' "$*"; fail=1; }
ok()   { printf 'ok: %s\n' "$*"; }

# One import, timed, with an explicit PYTHONPATH: jittor is usually installed
# editable against some *other* checkout, so a bare `python -c` silently tests
# the wrong source tree.
run_import() { # <logfile> <extra env assignments...>
  local log=$1; shift
  local start end
  start=$(date +%s)
  env PYTHONPATH="$SRC/python" JITTOR_HOME="$JITTOR_HOME" TMPDIR="$TMPDIR" \
      "$@" "$PY" -c '
import os, jittor, jittor.compiler as c
print("SRCTREE", os.path.dirname(jittor.__file__))
print("CACHE", c.cache_path)
# cache_path descends one further level for the CUDA key, so the directory
# that carries the build-configuration fingerprint is found by walking up to
# the build_config.json that names it.
d = c.cache_path
while d != os.path.dirname(d) and not os.path.exists(os.path.join(d, "build_config.json")):
    d = os.path.dirname(d)
print("CONFIGDIR", d)
print("HAS_CUDA", jittor.has_cuda)
print("OK", (jittor.ones(3)*2).sum().item())
' >"$log" 2>&1
  local rc=$?
  end=$(date +%s)
  printf 'rc=%s %ss %s\n' "$rc" "$((end - start))" "$log"
  # The source tree check is not optional: every other assertion is void if
  # the run imported somebody else's checkout.
  if ! grep -q "SRCTREE $SRC/python/jittor" "$log"; then
    bad "imported the wrong source tree (see $log)"
  fi
  return $rc
}

cache_of()  { sed -n 's/^CACHE //p' "$1"; }
config_of() { sed -n 's/^CONFIGDIR //p' "$1"; }

step "1/4 cold cache: an empty JITTOR_HOME must build and run"
rm -rf "$JITTOR_HOME"
mkdir -p "$JITTOR_HOME" "$TMPDIR"
t0=$(date +%s)
run_import "$LOGDIR/1-cold.log" ${NVCC:+nvcc_path="$NVCC"} || bad "cold import failed"
cold_secs=$(( $(date +%s) - t0 ))
COLD_CACHE=$(cache_of "$LOGDIR/1-cold.log")
COLD_CONFIG=$(config_of "$LOGDIR/1-cold.log")
echo "cold import: ${cold_secs}s -> $COLD_CACHE"
# The products and the record of what configuration produced them live
# together, so a stale directory can always be explained after the fact.
[ -n "$COLD_CONFIG" ] && [ -f "$COLD_CONFIG/build_config.json" ] \
  || bad "no build_config.json above the products"
[ -f "$JITTOR_HOME/.cache/jittor/probe.json" ] || bad "no probe.json"
grep -q "please rerun" "$LOGDIR/1-cold.log" && bad "cold build asked for a rerun"
ok "cold cache"

step "2/4 hot cache: a second import must not compile anything"
run_import "$LOGDIR/2-hot.log" ${NVCC:+nvcc_path="$NVCC"} || bad "hot import failed"
[ "$(cache_of "$LOGDIR/2-hot.log")" = "$COLD_CACHE" ] || bad "hot run landed in a different cache directory"
# "Compiling" lines in a warm run mean something invalidated a product that
# nothing changed -- a nondeterministic key, a timestamp in a command line.
if grep -qE "Compiling [0-9]+ files|jit_utils updated" "$LOGDIR/2-hot.log"; then
  bad "hot import recompiled; see $LOGDIR/2-hot.log"
else
  ok "hot cache: nothing recompiled"
fi

step "3/4 concurrent cold start: $JOBS importers racing into one empty config dir"
# The whole configuration directory, not just the CUDA subdirectory: the
# products several processes can race to replace (jit_utils_core, the core
# .so each of them then dlopens) live at this level.
rm -rf "$COLD_CONFIG"
pids=()
for i in $(seq 1 "$JOBS"); do
  run_import "$LOGDIR/3-conc-$i.log" ${NVCC:+nvcc_path="$NVCC"} &
  pids+=($!)
done
conc_fail=0
for p in "${pids[@]}"; do wait "$p" || conc_fail=1; done
[ "$conc_fail" = 0 ] || bad "a concurrent importer failed; see $LOGDIR/3-conc-*.log"
# A racing writer that replaced a .so another process had already dlopen'd
# shows up here and nowhere else.
grep -lE "Segmentation fault|core dumped|undefined symbol|file not found|truncated" \
     "$LOGDIR"/3-conc-*.log && bad "concurrent run shows cache corruption"
# And the cache has to be usable *afterwards*.
run_import "$LOGDIR/3-after.log" ${NVCC:+nvcc_path="$NVCC"} || bad "cache unusable after the concurrent run"
ok "concurrent cold start"

step "4/4 flag switch: two build configurations must not evict each other"
# nvcc_path="" is the documented CPU-only build. It produces different object
# code from the CUDA round, so it needs its own directory -- otherwise the two
# rebuild the products they share and hand each other a "please rerun".
run_import "$LOGDIR/4-cpu-1.log" nvcc_path="" || bad "CPU-only import failed"
CPU_CONFIG=$(config_of "$LOGDIR/4-cpu-1.log")
[ "$CPU_CONFIG" != "$COLD_CONFIG" ] || bad "CPU-only and CUDA share one cache directory"
grep -qE "HAS_CUDA (0|False)" "$LOGDIR/4-cpu-1.log" || bad "nvcc_path=\"\" still built with CUDA"
run_import "$LOGDIR/4-cuda-2.log" ${NVCC:+nvcc_path="$NVCC"} || bad "CUDA import after CPU-only failed"
run_import "$LOGDIR/4-cpu-2.log" nvcc_path=""                 || bad "CPU-only import after CUDA failed"
# The whole point: after each side has been built once, alternating is free.
for f in "$LOGDIR/4-cuda-2.log" "$LOGDIR/4-cpu-2.log"; do
  grep -qE "Compiling [0-9]+ files|jit_utils updated" "$f" \
    && bad "switching configurations recompiled: $f"
done
# Both configurations must still share one lock: it guards the downloads that
# every configuration on this toolchain has in common.
env PYTHONPATH="$SRC/python" JITTOR_HOME="$JITTOR_HOME" NVCC_FOR_CHECK="$NVCC" "$PY" -c '
import os, importlib, jittor_utils
locks = []
for v in ("", os.environ.get("NVCC_FOR_CHECK", "")):
    os.environ["nvcc_path"] = v
    importlib.reload(jittor_utils)
    jittor_utils.find_cache_path()
    locks.append(jittor_utils.lock_path)
assert locks[0] == locks[1], locks
print("one lock for both configurations:", locks[0])
' || bad "the two configurations do not share one lock"
ok "flag switch"

printf '\n=== summary ===\n'
echo "cold import: ${cold_secs}s"
echo "logs: $LOGDIR"
if [ "$fail" = 0 ]; then echo "ALL FOUR SCENARIOS PASSED"; else echo "SOMETHING FAILED"; fi
exit $fail
