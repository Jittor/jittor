#!/usr/bin/env bash
set -euo pipefail

source_root=${JITTOR_SOURCE_ROOT:?set JITTOR_SOURCE_ROOT}
full_root=${VERL_FULL_PPO_ROOT:?set VERL_FULL_PPO_ROOT}
verl_root=${VERL_SOURCE_ROOT:?set VERL_SOURCE_ROOT}
oracle_root=${source_root}/agent/skills/verl-torch-compat/runtime/oracle
oracle_python=${VERL_ORACLE_PYTHON:-${full_root}/oracle/bin/python}
# Do not append inherited site-packages or the Jittor Python tree. A separate
# executable with a shared, polluted module path is not an independent oracle.
export PYTHONPATH="${oracle_root}:${verl_root}:${source_root}/adapters"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset JITTOR_TORCH_SHIM JITTOR_TORCH_SKIP_EXT_BUILD JITTOR_TORCH_KEEP_HOME
unset JITTOR_TRANSFORMERS_MATERIALIZED_INIT JITTOR_VERL_NATIVE_NPU_KERNELS
unset JITTOR_TORCH_DISTRIBUTED_AUTO_INIT RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES
export JITTOR_VERL_HF_ASYNC_ROLLOUT=1
exec "${oracle_python}" -u -s "${oracle_root}/entry.py" "$@"
