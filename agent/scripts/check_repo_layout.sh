#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
status=0

fail() {
  printf '%s\n' "$1" >&2
  status=1
}

# These are live runtime or repository contracts. Names removed by past
# refactors belong to Git history and structure tests, not to this script.
required_paths=(
  python/jittor/selftest.py tests examples tools docs/conf.py docs/index.md
  backends/comm/__init__.py backends/acl/include/aclops
  backends/acl/include/aclnn backends/acl/kernels/native backends/comm/hccl
  backends/corex/__init__.py backends/cuda/include
  backends/cuda/src backends/cuda/kernels/cub
  backends/cuda/kernels/cublas backends/cuda/kernels/cudnn
  backends/cuda/kernels/cufft backends/cuda/kernels/curand
  backends/cuda/kernels/cusparse backends/cuda/libraries/cutt
  backends/comm/nccl backends/cpu/libraries/mkl
  backends/comm/mpi/inc backends/comm/mpi/ops
  backends/comm/mpi/src backends/rocm
)
for path in "${required_paths[@]}"; do
  [[ -e "$REPO_ROOT/$path" ]] || fail "missing required repository path: $path"
done

[[ ! -e "$REPO_ROOT/tests/__init__.py" ]] || \
  fail 'repository tests must not be an importable package: tests/__init__.py'

# Generated notebooks are products, including when a broad ignore rule would
# otherwise hide them. Jupytext materializes them outside the checkout.
while IFS= read -r path; do
  fail "notebook product must stay outside the checkout: ${path#"$REPO_ROOT"/}"
done < <(find "$REPO_ROOT" -path "$REPO_ROOT/.git" -prune -o -type f \
  \( -name '*.ipynb' -o -name '*.src.md' \) -print)

# Experiment directories have a durable external owner: JITTOR_LAB_ROOT.
while IFS= read -r path; do
  fail "experiment directory must live under JITTOR_LAB_ROOT: ${path#"$REPO_ROOT"/}"
done < <(find "$REPO_ROOT" -mindepth 1 -maxdepth 1 -type d \
  \( -name 'jittor_fsdp2' -o -name '*_work' -o -name '*_probe' \) -print)

# A module and package with the same import spelling make resolution depend on
# the importer and installation layout.
while IFS= read -r module_path; do
  package_path="${module_path%.py}"
  [[ ! -d "$package_path" ]] || fail \
    "module/package path collision: ${module_path#"$REPO_ROOT"/} and ${package_path#"$REPO_ROOT"/}/"
done < <(find "$REPO_ROOT/python/jittor" -type f -name '*.py' ! -name '__init__.py')

if [[ -d "$REPO_ROOT/.claude/worktrees" ]] &&
   [[ -n "$(find "$REPO_ROOT/.claude/worktrees" -mindepth 1 -print -quit)" ]]; then
  fail 'Git worktrees must live under JITTOR_LAB_ROOT/worktrees.'
fi

python3 "$REPO_ROOT/agent/scripts/check_docs_governance.py" || status=1

if (( status != 0 )); then
  fail 'Move experiments and runtime state outside the repository.'
  exit "$status"
fi

echo "repository layout OK: $REPO_ROOT"
