#!/usr/bin/env bash
# Provision ONLY a fresh, hash-locked official oracle; never modify shared envs.
set -euo pipefail

base_python=${VERL_ORACLE_BASE_PYTHON:?set an existing Python matching the NPU wheels}
venv_dir=${VERL_ORACLE_VENV_DIR:?set a new isolated venv directory}
wheelhouse=${VERL_ORACLE_WHEELHOUSE:?set the verified local wheel directory}
requirements=${VERL_ORACLE_REQUIREMENTS_LOCK:?set a fully pinned --require-hashes requirements file}
source_root=${JITTOR_SOURCE_ROOT:?set JITTOR_SOURCE_ROOT}
verl_root=${VERL_SOURCE_ROOT:?set VERL_SOURCE_ROOT}

if [[ -e ${venv_dir} ]]; then
    echo "refusing to overwrite existing oracle: ${venv_dir}" >&2
    exit 2
fi
if [[ ! -d ${wheelhouse} || ! -s ${requirements} ]]; then
    echo "verified wheelhouse and requirements lock are required" >&2
    exit 2
fi
# Pins and wheel hashes must be extracted/verified from the target deployment,
# not invented here. pip checks ALL transitive dependencies; an incomplete lock
# fails instead of resolving a new version or replacing the candidate's Torch.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
unset JITTOR_TORCH_SHIM JITTOR_TRANSFORMERS_MATERIALIZED_INIT
unset JITTOR_VERL_NATIVE_NPU_KERNELS JITTOR_TORCH_DISTRIBUTED_AUTO_INIT
"${base_python}" -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --isolated --no-index \
    --find-links "${wheelhouse}" --require-hashes -r "${requirements}"
"${venv_dir}/bin/python" -m pip check
export VERL_ORACLE_PYTHON=${venv_dir}/bin/python
# The wrapper also requires the PPO environment root even with explicit Python.
export VERL_FULL_PPO_ROOT=${VERL_FULL_PPO_ROOT:-${venv_dir}/..}
export JITTOR_SOURCE_ROOT=${source_root}
export VERL_SOURCE_ROOT=${verl_root}
bash "${source_root}/agent/skills/verl-torch-compat/runtime/oracle_worker_python.sh" \
    --check-only
