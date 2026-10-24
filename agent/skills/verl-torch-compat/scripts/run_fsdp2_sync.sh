#!/usr/bin/env bash
# Caches must have been warmed serially before launching these two ranks.
set -euo pipefail
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
    echo "Bash >=5.1 is required for fail-fast owned-child waits" >&2
    exit 2
fi
framework=${1:?torch or jittor}
attempt=${2:?fresh attempt name}
load_mode=${3:-no-load}
state_root=${VERL_NPU_STATE_ROOT:?set VERL_NPU_STATE_ROOT}
source_root=${JITTOR_SOURCE_ROOT:?set JITTOR_SOURCE_ROOT}
cards=${BENCH_CARDS:?set two idle NPU ids}
[[ $framework == torch || $framework == jittor ]]
[[ $load_mode == no-load || $load_mode == load ]]
[[ $attempt =~ ^[a-zA-Z0-9_-]+$ ]]
[[ $cards =~ ^[0-9]+,[0-9]+$ ]]
IFS=, read -r first_card second_card <<< "$cards"
[[ $first_card != "$second_card" ]]
result_root=$state_root/l2-l5/fsdp2-sync/$attempt
[[ ! -e $result_root ]] || { echo "Attempt already exists" >&2; exit 2; }
for card in "$first_card" "$second_card"; do
    process_info=$(npu-smi info -t proc-mem -i "$card" -c 0)
    grep -q "No process in device" <<< "$process_info" || {
        echo "Refusing occupied NPU $card" >&2; exit 20;
    }
done
mkdir -p "$result_root"
: "${LD_LIBRARY_PATH:=}" "${CMAKE_PREFIX_PATH:=}" "${PYTHONPATH:=}"
source "$state_root/cann9-py39/Ascend/cann-9.0.0/set_env.sh"
export ASCEND_RT_VISIBLE_DEVICES=$cards OMP_NUM_THREADS=1 WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1 MASTER_PORT=${BENCH_MASTER_PORT:?set unused rendezvous port}
script=$source_root/agent/skills/verl-torch-compat/scripts/probe_fsdp2_sync.py
python=$state_root/torch-npu-2.7.1/bin/python
unset PYTHONPATH JITTOR_TORCH_SHIM
export VERL_EXTRA_SITE_PACKAGES=$state_root/cann9-py39/lib/python3.11/site-packages
if [[ $framework == jittor ]]; then
    python=$state_root/cann9-py39/bin/python
    export PYTHONPATH=$source_root/adapters:$source_root/python:$source_root
    export JITTOR_TORCH_SHIM=1 JITTOR_TORCH_SKIP_EXT_BUILD=1 JITTOR_TORCH_KEEP_HOME=1
    export JITTOR_TORCH_DISTRIBUTED_AUTO_INIT=1 DISABLE_MULTIPROCESSING=0
    export JT_SYNC=1 trace_py_var=3 sync_run=1 backend_fallback=error
    export JT_HCCL_ROOTINFO_FILE=$result_root/hccl-root.bin JT_HCCL_COLLECTIVE_SYNC=full
fi
args=(--framework "$framework" --output "$result_root/result")
[[ $load_mode == no-load ]] || args+=(--load-cpu-state)
pids=()
signal_owned_group() {
    local pid=$1 signal=$2 info parent group
    info=$(ps -o ppid= -o pgid= -p "$pid" 2>/dev/null) || return 0
    read -r parent group <<< "$info"
    if [[ $parent == "$$" && $group == "$pid" ]]; then
        kill -"$signal" -- "-$pid" 2>/dev/null || true
    fi
}
cleanup() {
    local pid
    trap - EXIT INT TERM
    for pid in "${pids[@]}"; do signal_owned_group "$pid" TERM; done
    if (( ${#pids[@]} )); then sleep 1; fi
    for pid in "${pids[@]}"; do signal_owned_group "$pid" KILL; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for rank in 0 1; do
    (
        export RANK=$rank LOCAL_RANK=$rank
        if [[ $framework == jittor ]]; then
            export JITTOR_HOME=$state_root/runs/verl-ddp-correctness/jittor-home/rank$rank
            export XDG_CACHE_HOME=$state_root/runs/verl-ddp-correctness/xdg-cache/rank$rank
        fi
        exec setsid timeout --kill-after=15s 300s "$python" "$script" "${args[@]}"
    ) > "$result_root/rank$rank.log" 2>&1 &
    pids+=("$!")
done
status=0
while (( ${#pids[@]} )); do
    finished=""
    wait -n -p finished "${pids[@]}" || status=$?
    (( status == 0 )) || break
    remaining=()
    for pid in "${pids[@]}"; do
        [[ $pid == "$finished" ]] || remaining+=("$pid")
    done
    pids=("${remaining[@]}")
done
exit "$status"
