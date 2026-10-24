#!/usr/bin/env bash
# Usage: bash run_fsdp2_worker.sh torch|jittor actor|critic fresh_attempt
# Required: STATE (or VERL_NPU_STATE_ROOT), SOURCE (or JITTOR_SOURCE_ROOT),
# BENCH_CARDS=card0,card1, BENCH_MASTER_PORT, CHECKPOINT (or VERL_FSDP2_CHECKPOINT).
# Candidate additionally requires VERL_FSDP2_CACHES_PREWARMED=1: BOTH rank caches
# below must have been warmed serially. This launcher never performs cold JIT.
# Optional: VERL_FSDP2_ALGORITHM=fixed|ppo|grpo|rlhf|rlvr;
# VERL_FSDP2_STOP_AFTER=train; VERL_FSDP2_TIMEOUT_SECONDS=600 (hard cap,
# including ten seconds of TERM grace); VERL_FSDP2_DISTRIBUTED_TIMEOUT=180.
# Invoke asynchronously from the orchestrator. All rank logs/artifacts remain
# under STATE/l2-l5/fsdp2-worker-v1/attempt; no arrays are printed by this launcher.
set -euo pipefail
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
    echo "Bash >=5.1 is required for fail-fast owned-child waits" >&2
    exit 2
fi
[[ $# == 3 ]] || { echo "Expected framework role fresh_attempt" >&2; exit 2; }
framework=$1
role=$2
attempt=$3
state_root=${STATE:-${VERL_NPU_STATE_ROOT:?set STATE or VERL_NPU_STATE_ROOT}}
source_root=${SOURCE:-${JITTOR_SOURCE_ROOT:?set SOURCE or JITTOR_SOURCE_ROOT}}
checkpoint=${CHECKPOINT:-${VERL_FSDP2_CHECKPOINT:?set CHECKPOINT or VERL_FSDP2_CHECKPOINT}}
cards=${BENCH_CARDS:?set two idle physical NPU ids}
port=${BENCH_MASTER_PORT:?set an unused local rendezvous port}
stage=${VERL_FSDP2_STOP_AFTER:-train}
algorithm=${VERL_FSDP2_ALGORITHM:-fixed}
deadline=${VERL_FSDP2_TIMEOUT_SECONDS:-600}
distributed_timeout=${VERL_FSDP2_DISTRIBUTED_TIMEOUT:-180}
[[ $framework == torch || $framework == jittor ]]
[[ $role == actor || $role == critic ]]
[[ $algorithm == fixed || $algorithm == ppo || $algorithm == grpo || $algorithm == rlhf || $algorithm == rlvr ]]
if [[ $role == critic && ( $algorithm == grpo || $algorithm == rlvr ) ]]; then
    echo "GRPO/RLVR do not use the critic role in this protocol" >&2
    exit 2
fi
[[ $attempt =~ ^[a-zA-Z0-9_-]+$ && $cards =~ ^[0-9]+,[0-9]+$ ]]
[[ $port =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 ))
[[ $deadline =~ ^[1-9][0-9]{1,2}$ ]] && (( deadline >= 30 && deadline <= 600 ))
[[ $distributed_timeout =~ ^[1-9][0-9]{0,2}$ ]] && (( distributed_timeout < deadline - 10 ))
case "$stage" in import|init|reset|infer|train) ;; *) echo "Invalid stop stage" >&2; exit 2 ;; esac
IFS=, read -r first_card second_card <<< "$cards"
(( 10#$first_card != 10#$second_card ))
[[ $state_root == /* && $source_root == /* && $checkpoint == /* ]]
[[ -d $state_root && -d $source_root && -f $checkpoint/config.json ]]
script=$source_root/agent/skills/verl-torch-compat/scripts/probe_fsdp2_worker.py
[[ -f $script ]]
for command in npu-smi setsid timeout ps; do
    command -v "$command" >/dev/null || { echo "Missing launcher dependency: $command" >&2; exit 2; }
done
python=$state_root/torch-npu-2.7.1/bin/python
if [[ $framework == jittor ]]; then
    python=$state_root/cann9-py39/bin/python
    [[ ${VERL_FSDP2_CACHES_PREWARMED:-0} == 1 ]] || {
        echo "Warm both candidate rank caches serially before launching" >&2; exit 2;
    }
    for rank in 0 1; do
        [[ -d $state_root/runs/verl-ddp-correctness/jittor-home/rank$rank ]]
        [[ -d $state_root/runs/verl-ddp-correctness/xdg-cache/rank$rank ]]
    done
    for cache_kind in jittor-home xdg-cache; do
        cache0=$(cd "$state_root/runs/verl-ddp-correctness/$cache_kind/rank0" && pwd -P)
        cache1=$(cd "$state_root/runs/verl-ddp-correctness/$cache_kind/rank1" && pwd -P)
        [[ $cache0 != "$cache1" ]] || { echo "Rank caches resolve to the same directory" >&2; exit 2; }
    done
fi
[[ -x $python ]]
# Read-only preflight apart from a momentary localhost bind, immediately closed.
# The actual launcher still owns rendezvous; this check is not a port reservation.
"$python" -I -S - "$port" <<'PY'
import socket
import sys
with socket.socket() as listener:
    listener.bind(("127.0.0.1", int(sys.argv[1])))
PY
result_root=$state_root/l2-l5/fsdp2-worker-v1/$attempt
[[ ! -e $result_root ]] || { echo "Attempt already exists" >&2; exit 2; }
for card in "$first_card" "$second_card"; do
    process_info=$(npu-smi info -t proc-mem -i "$card" -c 0)
    [[ $process_info == *"No process in device"* ]] || {
        echo "Refusing occupied NPU $card" >&2; exit 20;
    }
done
mkdir -p "$state_root/l2-l5/fsdp2-worker-v1"
mkdir "$result_root"  # atomic refusal if another launcher claimed this attempt
: "${LD_LIBRARY_PATH:=}" "${CMAKE_PREFIX_PATH:=}" "${PYTHONPATH:=}"
source "$state_root/cann9-py39/Ascend/cann-9.0.0/set_env.sh"
export ASCEND_RT_VISIBLE_DEVICES=$cards OMP_NUM_THREADS=1 WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$port
unset PYTHONPATH JITTOR_TORCH_SHIM JITTOR_TRANSFORMERS_MATERIALIZED_INIT
unset JITTOR_VERL_NATIVE_NPU_KERNELS
unset JT_HCCL_WORLD_SIZE JT_HCCL_RANK JT_HCCL_LOCAL_RANK JT_HCCL_ROOTINFO_FILE
export VERL_EXTRA_SITE_PACKAGES=$state_root/cann9-py39/lib/python3.11/site-packages
if [[ $framework == jittor ]]; then
    export PYTHONPATH=$source_root/adapters:$source_root/python:$source_root
    export JITTOR_TORCH_SHIM=1 JITTOR_TORCH_SKIP_EXT_BUILD=1 JITTOR_TORCH_KEEP_HOME=1
    export JITTOR_TRANSFORMERS_MATERIALIZED_INIT=1
    export JITTOR_VERL_NATIVE_NPU_KERNELS=1
    export JITTOR_TORCH_DISTRIBUTED_AUTO_INIT=1 DISABLE_MULTIPROCESSING=0
    export JT_SYNC=1 trace_py_var=3 sync_run=1 backend_fallback=error
    export JT_HCCL_ROOTINFO_FILE=$result_root/hccl-root.bin JT_HCCL_COLLECTIVE_SYNC=full
fi
args=(--framework "$framework" --role "$role" --checkpoint "$checkpoint"
      --algorithm "$algorithm" --output "$result_root/worker.json"
      --stop-after "$stage" --timeout "$distributed_timeout")
pids=()
signal_owned_group() {
    local pid=$1 signal=$2 info parent group
    info=$(ps -o ppid= -o pgid= -p "$pid" 2>/dev/null) || return 0
    read -r parent group <<< "$info"
    # setsid creates a fresh session/group for this direct child only. Verify
    # both relationships each time; never signal a stale or unrelated PID.
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
        # GNU timeout manages the group rooted at its own PID. setsid isolates
        # that group/session from the caller and any other user's processes.
        exec setsid timeout --kill-after=10s "$((deadline - 10))s" "$python" "$script" "${args[@]}"
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
printf 'framework=%s role=%s algorithm=%s stage=%s status=%s\n' \
    "$framework" "$role" "$algorithm" "$stage" "$status" > "$result_root/launcher-status.txt"
exit "$status"
