#!/usr/bin/env bash
set -euo pipefail

source_root=${JITTOR_SOURCE_ROOT:?set JITTOR_SOURCE_ROOT}
full_root=${VERL_FULL_PPO_ROOT:?set VERL_FULL_PPO_ROOT}
export PYTHONPATH="${source_root}/agent/skills/verl-torch-compat/runtime:${PYTHONPATH:-}"
exec "${full_root}/candidate/bin/python" "$@"
