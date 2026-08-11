#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
status=0

while IFS= read -r name; do
  case "$name" in
    .git|.github|.agents|.codex|.claude|agent|benchmarks|doc|docs|python|requirements|\
    .dockerignore|.gitignore|.gitlab-ci.yml|AGENTS.md|\
    AWESOME-JITTOR-LIST.cn.md|AWESOME-JITTOR-LIST.md|\
    asv.conf.json|CODE_OF_CONDUCT.md|CONTRIBUTING.md|Dockerfile|GOVERNANCE.md|\
    LICENSE.txt|MANIFEST.in|README.cn.md|README.md|README.src.md|\
    .pre-commit-config.yaml|noxfile.py|pyproject.toml|setup.py)
      ;;
    *)
      printf 'unexpected repository-root entry: %s\n' "$name" >&2
      status=1
      ;;
  esac
done < <(find "$REPO_ROOT" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)

for forbidden_path in \
  "$REPO_ROOT/jittor_fsdp2" \
  "$REPO_ROOT/python/jittor/torch_fsdp2_compat.py" \
  "$REPO_ROOT/python/jittor/nn.py" \
  "$REPO_ROOT/python/jittor/_nn" \
  "$REPO_ROOT/python/jittor/misc.py" \
  "$REPO_ROOT/python/jittor/_misc" \
  "$REPO_ROOT/python/jittor/pool.py" \
  "$REPO_ROOT/python/jittor/_pool" \
  "$REPO_ROOT/python/jittor/torch_compat.py" \
  "$REPO_ROOT/python/jittor/_torch_compat"; do
  if [[ -e "$forbidden_path" ]]; then
    printf 'forbidden legacy path: %s\n' "${forbidden_path#"$REPO_ROOT"/}" >&2
    status=1
  fi
done

if [[ -d "$REPO_ROOT/.claude/worktrees" ]] &&
   [[ -n "$(find "$REPO_ROOT/.claude/worktrees" -mindepth 1 -print -quit)" ]]; then
  echo 'Git worktrees must live under ${JITTOR_LAB_ROOT}/worktrees, not .claude/worktrees.' >&2
  status=1
fi

if (( status != 0 )); then
  echo 'Move experiments and runtime state to ${JITTOR_LAB_ROOT:-../jittor-lab}.' >&2
  exit "$status"
fi

echo "repository layout OK: $REPO_ROOT"
