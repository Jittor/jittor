#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
status=0

if [[ ! -f "$REPO_ROOT/python/jittor/selftest.py" ]]; then
  echo 'missing installed smoke test: python/jittor/selftest.py' >&2
  status=1
fi
if [[ ! -d "$REPO_ROOT/tests" ]]; then
  echo 'missing repository test suite: tests/' >&2
  status=1
fi
if [[ ! -d "$REPO_ROOT/examples" ]]; then
  echo 'missing repository examples: examples/' >&2
  status=1
fi
if [[ ! -d "$REPO_ROOT/tools" ]]; then
  echo 'missing repository tools: tools/' >&2
  status=1
fi
if [[ ! -f "$REPO_ROOT/docs/conf.py" ]] || [[ ! -f "$REPO_ROOT/docs/index.md" ]]; then
  echo 'missing canonical Sphinx/MyST documentation tree: docs/' >&2
  status=1
fi
if find "$REPO_ROOT" -path "$REPO_ROOT/.git" -prune -o -type f \
  \( -name '*.ipynb' -o -name '*.src.md' \) -print -quit | grep -q .; then
  echo 'notebook products and legacy .src.md files must stay outside the checkout.' >&2
  find "$REPO_ROOT" -path "$REPO_ROOT/.git" -prune -o -type f \
    \( -name '*.ipynb' -o -name '*.src.md' \) -print >&2
  status=1
fi
if find "$REPO_ROOT" -path "$REPO_ROOT/.git" -prune -o \
  \( -type d -name '__pycache__' -o -type f \( -name '*.pyc' -o -name '*.pyo' \) \) \
  -print -quit | grep -q .; then
  echo 'Python bytecode caches must stay outside the checkout.' >&2
  find "$REPO_ROOT" -path "$REPO_ROOT/.git" -prune -o \
    \( -type d -name '__pycache__' -o -type f \( -name '*.pyc' -o -name '*.pyo' \) \) \
    -print >&2
  status=1
fi
if [[ -e "$REPO_ROOT/tests/__init__.py" ]]; then
  echo 'repository tests must not be an importable distribution package: tests/__init__.py' >&2
  status=1
fi

while IFS= read -r name; do
  case "$name" in
    .git|.github|.agents|.codex|.claude|agent|benchmarks|docs|examples|python|requirements|tests|tools|\
    .dockerignore|.gitignore|AGENTS.md|\
    AWESOME-JITTOR-LIST.cn.md|AWESOME-JITTOR-LIST.md|\
    asv.conf.json|CODE_OF_CONDUCT.md|CONTRIBUTING.md|Dockerfile|GOVERNANCE.md|\
    LICENSE.txt|MANIFEST.in|README.md|\
    .pre-commit-config.yaml|noxfile.py|pyproject.toml|setup.py)
      ;;
    *)
      printf 'unexpected repository-root entry: %s\n' "$name" >&2
      status=1
      ;;
  esac
done < <(find "$REPO_ROOT" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)

for forbidden_path in \
  "$REPO_ROOT/doc" \
  "$REPO_ROOT/jittor_fsdp2" \
  "$REPO_ROOT/README.cn.md" \
  "$REPO_ROOT/README.src.md" \
  "$REPO_ROOT/agent/manuals/design" \
  "$REPO_ROOT/python/jittor/attention.py" \
  "$REPO_ROOT/python/jittor/torch_fsdp2_compat.py" \
  "$REPO_ROOT/python/jittor/torch_fsdp2_compat" \
  "$REPO_ROOT/python/jittor/_torch_fsdp2" \
  "$REPO_ROOT/python/jittor/nn.py" \
  "$REPO_ROOT/python/jittor/_nn" \
  "$REPO_ROOT/python/jittor/misc.py" \
  "$REPO_ROOT/python/jittor/_misc" \
  "$REPO_ROOT/python/jittor/pool.py" \
  "$REPO_ROOT/python/jittor/_pool" \
  "$REPO_ROOT/python/jittor/optim.py" \
  "$REPO_ROOT/python/jittor/torch_compat.py" \
  "$REPO_ROOT/python/jittor/_torch_compat" \
  "$REPO_ROOT/python/jittor/triton_shim" \
  "$REPO_ROOT/python/jittor/depthwise_conv.py" \
  "$REPO_ROOT/python/jittor/test" \
  "$REPO_ROOT/python/jittor/monkeypatch_ops.py" \
  "$REPO_ROOT/python/jittor/torch_shim" \
  "$REPO_ROOT/python/jittor/script" \
  "$REPO_ROOT/python/jittor/demo" \
  "$REPO_ROOT/python/jittor/notebook" \
  "$REPO_ROOT/python/jittor/vcompiler" \
  "$REPO_ROOT/python/jittor/version" \
  "$REPO_ROOT/python/jittor/utils/polish.py" \
  "$REPO_ROOT/python/jittor/utils/polish_centos.py" \
  "$REPO_ROOT/python/jittor/extern/llvm" \
  "$REPO_ROOT/python/jittor_utils/translator.py" \
  "$REPO_ROOT/tools/docs/legacy" \
  "$REPO_ROOT/python/jittor_utils/pack_offline.py"; do
  if [[ -e "$forbidden_path" ]]; then
    printf 'forbidden legacy path: %s\n' "${forbidden_path#"$REPO_ROOT"/}" >&2
    status=1
  fi
done

while IFS= read -r legacy_path; do
  printf 'forbidden legacy path name anywhere in repository: %s\n' \
    "${legacy_path#"$REPO_ROOT"/}" >&2
  status=1
done < <(
  find "$REPO_ROOT" -path "$REPO_ROOT/.git" -prune -o \
    \( -name 'jittor_fsdp2' -o -name 'torch_fsdp2_compat.py' -o \
       -name 'torch_fsdp2_compat' \) -print | sort
)

while IFS= read -r module_path; do
  package_path="${module_path%.py}"
  if [[ -d "$package_path" ]]; then
    printf 'module/package path collision: %s and %s\n' \
      "${module_path#"$REPO_ROOT"/}" "${package_path#"$REPO_ROOT"/}/" >&2
    status=1
  fi
done < <(find "$REPO_ROOT/python/jittor" -type f -name '*.py' ! -name '__init__.py' | sort)

for required_path in \
  "$REPO_ROOT/python/jittor/extern/__init__.py" \
  "$REPO_ROOT/python/jittor/extern/acl/aclops" \
  "$REPO_ROOT/python/jittor/extern/acl/aclnn" \
  "$REPO_ROOT/python/jittor/extern/acl/hccl" \
  "$REPO_ROOT/python/jittor/extern/corex/corex_compiler.py" \
  "$REPO_ROOT/python/jittor/extern/cuda/inc" \
  "$REPO_ROOT/python/jittor/extern/cuda/src" \
  "$REPO_ROOT/python/jittor/extern/cuda/cub" \
  "$REPO_ROOT/python/jittor/extern/cuda/cublas" \
  "$REPO_ROOT/python/jittor/extern/cuda/cudnn" \
  "$REPO_ROOT/python/jittor/extern/cuda/cufft" \
  "$REPO_ROOT/python/jittor/extern/cuda/curand" \
  "$REPO_ROOT/python/jittor/extern/cuda/cusparse" \
  "$REPO_ROOT/python/jittor/extern/cuda/cutt" \
  "$REPO_ROOT/python/jittor/extern/cuda/nccl" \
  "$REPO_ROOT/python/jittor/extern/mkl/ops" \
  "$REPO_ROOT/python/jittor/extern/mpi/inc" \
  "$REPO_ROOT/python/jittor/extern/mpi/ops" \
  "$REPO_ROOT/python/jittor/extern/mpi/src" \
  "$REPO_ROOT/python/jittor/extern/rocm"; do
  if [[ ! -e "$required_path" ]]; then
    printf 'missing required runtime path: %s\n' "${required_path#"$REPO_ROOT"/}" >&2
    status=1
  fi
done

if grep -q '^def compile_extern():' "$REPO_ROOT/python/jittor/compiler.py"; then
  echo 'retired compiler.compile_extern LLVM hook must not return.' >&2
  status=1
fi

legacy_selftest_module='jittor.test.'
legacy_selftest_module+='test_example'
if grep -R -n \
  --include='*.py' \
  --include='*.sh' \
  -- "$legacy_selftest_module" "$REPO_ROOT/python"; then
  echo 'installed smoke tests must use python -m jittor.selftest.' >&2
  status=1
fi

for active_doc in \
  "$REPO_ROOT/Dockerfile" \
  "$REPO_ROOT/CONTRIBUTING.md" \
  "$REPO_ROOT/README.md"; do
  if grep -n -- "$legacy_selftest_module" "$active_doc"; then
    echo 'installation documentation must use python -m jittor.selftest.' >&2
    status=1
  fi
done

active_reference_paths=(
  "$REPO_ROOT/python"
  "$REPO_ROOT/tools"
  "$REPO_ROOT/examples"
  "$REPO_ROOT/.github"
  "$REPO_ROOT/docs"
  "$REPO_ROOT/README.md"
  "$REPO_ROOT/CONTRIBUTING.md"
  "$REPO_ROOT/Dockerfile"
  "$REPO_ROOT/MANIFEST.in"
  "$REPO_ROOT/pyproject.toml"
  "$REPO_ROOT/noxfile.py"
  "$REPO_ROOT/.pre-commit-config.yaml"
)
old_package_root='python/jittor'
old_notebook_module='jittor'
old_notebook_module+='.notebook'
old_test_prefix='jittor'
old_test_prefix+='.test.'
old_polish_module='jittor.utils.'
old_polish_module+='polish'
old_offline_module='jittor_utils.'
old_offline_module+='pack_offline'
for old_reference in \
  "$old_package_root/script" \
  "$old_package_root/demo" \
  "$old_package_root/notebook" \
  "$old_package_root/vcompiler" \
  "$old_notebook_module" \
  "$old_test_prefix" \
  "$old_polish_module" \
  "$old_offline_module"; do
  if grep -I -R -n -F -- "$old_reference" "${active_reference_paths[@]}"; then
    printf 'active file still references retired runtime path: %s\n' \
      "$old_reference" >&2
    status=1
  fi
done

old_vcompiler_module='jittor.'
old_vcompiler_module+='vcompiler'
if grep -I -R -n -F -- "$old_vcompiler_module" \
  "$REPO_ROOT/python" "$REPO_ROOT/tools" "$REPO_ROOT/examples" \
  "$REPO_ROOT/.github" "$REPO_ROOT/Dockerfile" "$REPO_ROOT/pyproject.toml" \
  "$REPO_ROOT/noxfile.py"; then
  echo 'active code still imports retired jittor.vcompiler.' >&2
  status=1
fi

if [[ -d "$REPO_ROOT/.claude/worktrees" ]] &&
   [[ -n "$(find "$REPO_ROOT/.claude/worktrees" -mindepth 1 -print -quit)" ]]; then
  echo 'Git worktrees must live under ${JITTOR_LAB_ROOT}/worktrees, not .claude/worktrees.' >&2
  status=1
fi

if ! python3 "$REPO_ROOT/agent/scripts/check_docs_governance.py"; then
  status=1
fi

if (( status != 0 )); then
  echo 'Move experiments and runtime state to ${JITTOR_LAB_ROOT:-../jittor-lab}.' >&2
  exit "$status"
fi

echo "repository layout OK: $REPO_ROOT"
