#!/usr/bin/env bash
set -euo pipefail

framework=${1:?framework: torch or jittor}
topology=${2:?topology: single or multi}
task=${3:?task: train, infer, or ddp-sync}
attempt=${4:?unique attempt name}

state_root=${VERL_NPU_STATE_ROOT:?set VERL_NPU_STATE_ROOT}
source_root=${JITTOR_SOURCE_ROOT:?set JITTOR_SOURCE_ROOT}
model=${VERL_MODEL_PATH:?set VERL_MODEL_PATH}
cards_csv=${BENCH_CARDS:?set BENCH_CARDS to one or two free logical device ids}
base_env=${VERL_CANN_ENV_ROOT:-${state_root}/cann9-py39}
torch_env=${VERL_TORCH_NPU_ENV_ROOT:-${state_root}/torch-npu-2.7.1}
run_state_root=${VERL_NPU_RUN_STATE_ROOT:-${state_root}}
verl_source_root=${VERL_SOURCE_ROOT:-}
script=${source_root}/agent/skills/verl-torch-compat/scripts/benchmark_llm.py
if [[ ${task} == ddp-sync ]]; then
    script=${source_root}/agent/skills/verl-torch-compat/scripts/probe_ddp_sync.py
fi
result_namespace=${BENCH_RESULT_NAMESPACE:-qwen35-08b}
result_root=${run_state_root}/benchmarks/${result_namespace}/${attempt}
steps=${BENCH_STEPS:-5}
warmup=${BENCH_WARMUP:-2}
seq_len=${BENCH_SEQ_LEN:-128}
batch_size=${BENCH_BATCH_SIZE:-1}
new_tokens=${BENCH_NEW_TOKENS:-32}
objective=${BENCH_OBJECTIVE:-causal}
dtype=${BENCH_DTYPE:-bfloat16}
learning_rate=${BENCH_LEARNING_RATE:-1e-6}
adam_eps=${BENCH_ADAM_EPS:-1e-5}
run_timeout=${BENCH_TIMEOUT_SECONDS:-600}

if [[ ${framework} != torch && ${framework} != jittor ]]; then
    echo "framework must be torch or jittor" >&2
    exit 2
fi
if [[ ${topology} != single && ${topology} != multi ]]; then
    echo "topology must be single or multi" >&2
    exit 2
fi
if [[ ${task} != train && ${task} != infer && ${task} != ddp-sync ]]; then
    echo "task must be train, infer, or ddp-sync" >&2
    exit 2
fi
if [[ ${task} == ddp-sync && ${topology} != multi ]]; then
    echo "ddp-sync requires multi topology" >&2
    exit 2
fi
if [[ ! ${run_timeout} =~ ^[1-9][0-9]*$ ]] || (( run_timeout < 30 || run_timeout > 1800 )); then
    echo "BENCH_TIMEOUT_SECONDS must be an integer from 30 through 1800" >&2
    exit 2
fi
command -v timeout >/dev/null || { echo "timeout is required" >&2; exit 2; }
if [[ -e ${result_root} ]]; then
    echo "Attempt already exists; use a fresh name: ${result_root}" >&2
    exit 2
fi
IFS=',' read -r -a cards <<<"${cards_csv}"
world_size=1
[[ ${topology} == multi ]] && world_size=2
if [[ ${#cards[@]} -ne ${world_size} ]]; then
    echo "BENCH_CARDS must name exactly ${world_size} card(s)" >&2
    exit 2
fi
for card in "${cards[@]}"; do
    process_info=$(npu-smi info -t proc-mem -i "${card}" -c 0)
    if ! grep -q "No process in device" <<<"${process_info}"; then
        echo "Refusing to use occupied NPU ${card}:" >&2
        echo "${process_info}" >&2
        exit 20
    fi
done

mkdir -p "${result_root}/logs" "${result_root}/tmp"
: "${LD_LIBRARY_PATH:=}"
: "${CMAKE_PREFIX_PATH:=}"
: "${PYTHONPATH:=}"
source "${base_env}/Ascend/cann-9.0.0/set_env.sh"
export VERL_EXTRA_SITE_PACKAGES=${VERL_EXTRA_SITE_PACKAGES:-${base_env}/lib/python3.11/site-packages}

if [[ -n ${verl_source_root} ]]; then
    export PYTHONPATH="${verl_source_root}:${PYTHONPATH}"
fi

export ASCEND_RT_VISIBLE_DEVICES
ASCEND_RT_VISIBLE_DEVICES=$(IFS=,; echo "${cards[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_DEACTIVATE_ASYNC_LOAD=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${BENCH_MASTER_PORT:-29622}
export WORLD_SIZE=${world_size}
export TMPDIR="${result_root}/tmp"

jittor_home_base=
xdg_cache_home_base=
if [[ ${framework} == torch ]]; then
    python=${torch_env}/bin/python
else
    python=${base_env}/bin/python
    export PYTHONPATH="${source_root}/adapters:${source_root}/python:${source_root}:${PYTHONPATH}"
    export JITTOR_TORCH_SHIM=1
    export JITTOR_TORCH_SKIP_EXT_BUILD=1
    export JITTOR_TORCH_KEEP_HOME=1
    export JITTOR_TRANSFORMERS_MATERIALIZED_INIT=1
    export DISABLE_MULTIPROCESSING=0
    export backend_fallback=error
    export JT_BUILD_USE_MKL=0
    export cache_name=${JITTOR_CACHE_NAME:-verl-qwen35-npu}
    if [[ -n ${JITTOR_SYNC_RUN:-} ]]; then
        export sync_run=${JITTOR_SYNC_RUN}
    else
        unset sync_run
    fi
    if [[ ${topology} == single ]]; then
        export XDG_CACHE_HOME=${JITTOR_XDG_CACHE_HOME_OVERRIDE:-${run_state_root}/runs/single/xdg-cache}
        export JITTOR_HOME=${JITTOR_HOME_OVERRIDE:-${run_state_root}/runs/single/jittor-home}
    else
        local_cache_scope=verl-benchmark
        [[ ${task} == ddp-sync ]] && local_cache_scope=verl-ddp-correctness
        jittor_home_base=${run_state_root}/runs/${local_cache_scope}/jittor-home
        xdg_cache_home_base=${run_state_root}/runs/${local_cache_scope}/xdg-cache
        export JT_HCCL_WORLD_SIZE=${world_size}
        export JT_HCCL_ROOTINFO_FILE="${result_root}/hccl_root.bin"
        export JT_HCCL_COLLECTIVE_SYNC=full
    fi
fi

run_rank() {
    local rank=$1
    local output=${result_root}/${framework}-${topology}-${task}
    local rank_jittor_home=${JITTOR_HOME:-}
    local rank_xdg_cache_home=${XDG_CACHE_HOME:-}
    if [[ ${framework} == jittor && ${topology} == multi ]]; then
        rank_jittor_home=${jittor_home_base}/rank${rank}
        rank_xdg_cache_home=${xdg_cache_home_base}/rank${rank}
        mkdir -p "${rank_jittor_home}" "${rank_xdg_cache_home}"
    fi
    local -a script_args=(--framework "${framework}" --output "${output}")
    if [[ ${task} != ddp-sync ]]; then
        script_args+=(--task "${task}" --objective "${objective}"
            --device npu --dtype "${dtype}" --model "${model}"
            --learning-rate "${learning_rate}" --adam-eps "${adam_eps}"
            --seq-len "${seq_len}"
            --batch-size "${batch_size}" --warmup "${warmup}"
            --steps "${steps}" --new-tokens "${new_tokens}")
    fi
    RANK=${rank} LOCAL_RANK=${rank} \
    JT_HCCL_RANK=${rank} JT_HCCL_LOCAL_RANK=${rank} \
    JITTOR_HOME=${rank_jittor_home} XDG_CACHE_HOME=${rank_xdg_cache_home} \
        timeout --kill-after=15s "${run_timeout}s" \
        "${python}" -u "${script}" "${script_args[@]}" \
        >"${result_root}/logs/rank${rank}.log" 2>&1
}

if [[ ${world_size} -eq 1 ]]; then
    status=0
    run_rank 0 || status=$?
    cat "${result_root}/logs/rank0.log"
    exit "${status}"
fi

run_rank 0 &
rank0_pid=$!
run_rank 1 &
rank1_pid=$!
set +e
wait "${rank0_pid}"
rank0_status=$?
wait "${rank1_pid}"
rank1_status=$?
set -e

echo "===== rank 0 (exit ${rank0_status}) ====="
cat "${result_root}/logs/rank0.log"
echo "===== rank 1 (exit ${rank1_status}) ====="
cat "${result_root}/logs/rank1.log"
if [[ ${rank0_status} -ne 0 || ${rank1_status} -ne 0 ]]; then
    exit 1
fi
