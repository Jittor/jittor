#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
status=0

while IFS= read -r name; do
  case "$name" in
    .git|.github|.agents|.codex|.claude|agent|doc|python|\
    .dockerignore|.gitignore|.gitlab-ci.yml|AGENTS.md|\
    AWESOME-JITTOR-LIST.cn.md|AWESOME-JITTOR-LIST.md|\
    CODE_OF_CONDUCT.md|CONTRIBUTING.md|Dockerfile|GOVERNANCE.md|\
    LICENSE.txt|MANIFEST.in|README.cn.md|README.md|README.src.md|setup.py)
      ;;
    *)
      printf 'unexpected repository-root entry: %s\n' "$name" >&2
      status=1
      ;;
  esac
done < <(find "$REPO_ROOT" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)

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
