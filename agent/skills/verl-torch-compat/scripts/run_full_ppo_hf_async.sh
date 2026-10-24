#!/usr/bin/env bash
set -euo pipefail

framework=${1:?framework: jittor or torch}
device=${VERL_DEVICE:-npu}
if [[ ${device} != cuda && ${device} != npu ]]; then
    echo "VERL_DEVICE must be cuda or npu" >&2
    exit 2
fi
# This is a fresh driver, not a rollout actor. Never inherit another run's
# actor-local backend binding or Ray visibility override. Upstream platform
# registry names are vendor names (nvidia/huawei), not device names (cuda/npu).
unset VERL_PLATFORM RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
unset RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO
unset JITTOR_VERL_RAY_CUDA_CONTEXT JITTOR_VERL_RAY_NPU_CONTEXT
if [[ ${device} == cuda ]]; then
    export VERL_PLATFORM=nvidia
else
    export VERL_PLATFORM=huawei
fi
allow_shared_cuda=${VERL_ALLOW_SHARED_CUDA-0}
if [[ ${allow_shared_cuda} != 0 && ${allow_shared_cuda} != 1 ]]; then
    echo "VERL_ALLOW_SHARED_CUDA must be absent, 0 or 1" >&2
    exit 2
fi
attempt=${2:?unique attempt name}
cards=${3:?comma-separated free accelerator ids}
if [[ ! ${cards} =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "invalid accelerator ids: ${cards}" >&2
    exit 2
fi
IFS=',' read -r -a card_list <<<"${cards}"
n_gpus=${#card_list[@]}
if (( n_gpus < 1 )); then
    echo "at least one logical NPU id is required" >&2
    exit 2
fi

state_root=${VERL_STATE_ROOT:-${VERL_NPU_STATE_ROOT:-}}
if [[ -z ${state_root} ]]; then
    echo "set VERL_STATE_ROOT (or legacy VERL_NPU_STATE_ROOT)" >&2
    exit 2
fi
source_root=${JITTOR_SOURCE_ROOT:?set JITTOR_SOURCE_ROOT}
verl_root=${VERL_SOURCE_ROOT:?set VERL_SOURCE_ROOT}
model=${VERL_MODEL_PATH:?set VERL_MODEL_PATH}
critic_model=${VERL_CRITIC_MODEL_PATH:-${model}}
train_dtype=${VERL_TRAIN_DTYPE:-fp32}
training_steps=${VERL_TOTAL_TRAINING_STEPS:-5}
if [[ ! ${training_steps} =~ ^[1-9][0-9]*$ ]]; then
    echo "VERL_TOTAL_TRAINING_STEPS must be a positive integer" >&2
    exit 2
fi
full_root=${VERL_FULL_PPO_ROOT:-${state_root}/full-ppo-hf-async}
run_root=${full_root}/runs/${attempt}
dataset_root=${full_root}/dataset
# Ray launches workers through candidate_worker_python.sh, which must resolve
# the same isolated environment even when the caller did not pre-export the
# optional override used above.
export VERL_FULL_PPO_ROOT="${full_root}"

if [[ ${framework} != jittor && ${framework} != torch ]]; then
    echo "framework must be jittor or torch" >&2
    exit 2
fi
if [[ ${train_dtype} != fp32 && ${train_dtype} != bf16 ]]; then
    echo "VERL_TRAIN_DTYPE must be fp32 or bf16" >&2
    exit 2
fi
strict_fp32=${VERL_CUDA_STRICT_FP32-0}
if [[ ${strict_fp32} != 0 && ${strict_fp32} != 1 ]]; then
    echo "VERL_CUDA_STRICT_FP32 must be absent, 0 or 1" >&2
    exit 2
fi
if [[ ${device} == cuda && ${train_dtype} == fp32 ]]; then
    export VERL_CUDA_STRICT_FP32=1
else
    if [[ ${strict_fp32} == 1 ]]; then
        echo "VERL_CUDA_STRICT_FP32 requires CUDA fp32 training" >&2
        exit 2
    fi
    unset VERL_CUDA_STRICT_FP32
fi
export VERL_TRAIN_DTYPE="${train_dtype}"
rollout_dtype=float32
if [[ ${train_dtype} == bf16 ]]; then
    rollout_dtype=bfloat16
fi
if [[ -e ${run_root} ]]; then
    echo "attempt already exists: ${run_root}" >&2
    exit 2
fi
seen_cards=,
shared_cuda_context=()
for card in "${card_list[@]}"; do
    if [[ ! ${card} =~ ^[0-9]+$ ]]; then
        echo "invalid accelerator id: ${card}" >&2
        exit 2
    fi
    normalized_card=$((10#${card}))
    if [[ ${seen_cards} == *",${normalized_card},"* ]]; then
        echo "duplicate accelerator id: ${card}" >&2
        exit 2
    fi
    seen_cards+=${normalized_card},
done
for card in "${card_list[@]}"; do
    if [[ ${device} == npu ]]; then
        process_info=$(npu-smi info -t proc-mem -i "${card}" -c 0)
        if ! grep -q "No process in device" <<<"${process_info}"; then
            echo "refusing occupied NPU ${card}" >&2
            echo "${process_info}" >&2
            exit 20
        fi
    else
        process_info=$(nvidia-smi -i "${card}" --query-compute-apps=pid --format=csv,noheader)
        if [[ -n ${process_info} && ${allow_shared_cuda} != 1 ]]; then
            echo "refusing occupied CUDA device ${card}: ${process_info}" >&2
            exit 20
        fi
        if [[ ${allow_shared_cuda} == 1 ]]; then
            # Explicit authorization allows functional tests on a shared card,
            # never an exclusive-device performance claim. Observe only; do
            # not change, signal or terminate any existing GPU process.
            gpu_status=$(nvidia-smi -i "${card}" \
                --query-gpu=index,uuid,utilization.gpu,memory.free,memory.total \
                --format=csv,noheader,nounits)
            if [[ -z ${gpu_status} ]]; then
                echo "empty CUDA GPU status for shared device ${card}" >&2
                exit 21
            fi
            shared_cuda_context+=("requested_gpu=${card}" "gpu_status=${gpu_status}"
                                  "compute_pids=${process_info:-none}")
            echo "shared CUDA device ${card}: ${gpu_status}; occupied PIDs: ${process_info:-none}" >&2
        fi
    fi
done

# Ray appends a long session/socket suffix; keep the parent exceptionally short
# so Linux's 107-byte AF_UNIX limit is respected.  Give every attempt its own
# parent as well: an interrupted local Ray cluster leaves session_latest and
# sockets behind, and reusing that parent can make the next ray.init wait on a
# dead control plane for minutes.
attempt_tag=$(printf '%s' "${attempt}" | cksum)
attempt_tag=${attempt_tag%% *}
card_tag=${cards//,/}
ray_tmp=${VERL_RAY_TMPDIR:-/tmp/vh${card_tag}-${UID}-${attempt_tag}}
short_tmp=${VERL_PROCESS_TMPDIR:-/tmp/vt${card_tag}-${UID}-${attempt_tag}}
mkdir -p "${run_root}/tmp" "${run_root}/checkpoints" "${dataset_root}" "${ray_tmp}" "${short_tmp}"
if [[ ${device} == cuda && ${allow_shared_cuda} == 1 ]]; then
    export VERL_L4_PERFORMANCE_ELIGIBLE=0
    echo "shared CUDA functional validation only: performance is ineligible for L4" >&2
    printf '%s\n' \
        "mode=shared_cuda" "shared_cuda=true" "performance_eligible=false" \
        "l4_performance_eligible=0" \
        "observation_scope=preflight snapshot only; not continuous GPU monitoring" \
        "gpu_status_fields=index,uuid,utilization.gpu,memory.free,memory.total" \
        "utilization_unit=percent" "memory_unit=MiB" \
        "${shared_cuda_context[@]}" > "${run_root}/shared-cuda-context.txt"
fi
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PYTHONPATH=${PYTHONPATH:-}
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:-}
# This is a Ray driver, not an already launched distributed rank. Ray assigns
# worker ranks later; a previous shell's HCCL/NCCL/MPI-free rendezvous must not
# contaminate this import (the core bootstrap gives HCCL precedence).
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK MASTER_ADDR MASTER_PORT
unset JT_HCCL_WORLD_SIZE JT_HCCL_RANK JT_HCCL_LOCAL_RANK JT_HCCL_ROOTINFO_FILE
unset JT_NCCL_WORLD_SIZE JT_NCCL_RANK JT_NCCL_LOCAL_RANK JT_NCCL_ROOTINFO_FILE
if [[ ${device} == npu ]]; then
    source "${CANN_SET_ENV:-${state_root}/cann9-py39/Ascend/cann-9.0.0/set_env.sh}"
    export ASCEND_RT_VISIBLE_DEVICES=${cards}
else
    export CUDA_VISIBLE_DEVICES=${cards}
    unset ASCEND_RT_VISIBLE_DEVICES JITTOR_VERL_NATIVE_NPU_KERNELS
    export JT_BACKEND=cuda
    export JT_BUILD_NVCC_PATH=${VERL_NVCC_PATH:?set VERL_NVCC_PATH}
    export nvcc_path=${VERL_NVCC_PATH}
fi
export TOKENIZERS_PARALLELISM=false
export HF_DEACTIVATE_ASYNC_LOAD=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VERL_LOGGING_LEVEL=INFO
# verl's built-in NumPy DataProto transport serializes each tensor separately.
# TensorDict's default consolidate()+torch.save path requests one shared storage
# across Jittor tensors and SIGBUSes inside the Ray async actor on this aarch64
# NPU runtime. The NumPy path is an upstream-supported transport choice and is
# pinned for both candidate and oracle so later correctness comparisons use the
# same process boundary semantics.
export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy
export JITTOR_VERL_FATAL_TRACE=${JITTOR_VERL_FATAL_TRACE:-1}
export RAY_TMPDIR="${ray_tmp}"
export TMPDIR="${short_tmp}"
# On the shared NPU host, raylet registration can exceed Ray's default startup
# window while the local GCS is otherwise healthy.  Keep this configurable but
# give the isolated full-PPO run enough time to register under host load.
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-120}
if [[ ${framework} == jittor ]]; then
    export PYTHONPATH="${source_root}/compat/shim/resources:${verl_root}:${source_root}/adapters:${source_root}/python:${source_root}:${PYTHONPATH:-}"
    python=${full_root}/candidate/bin/python
    export JITTOR_TORCH_SHIM=1
    export JITTOR_TORCH_SKIP_EXT_BUILD=1
    export JITTOR_TORCH_KEEP_HOME=1
    export JITTOR_TRANSFORMERS_MATERIALIZED_INIT=1
    if [[ ${device} == npu ]]; then
        export JITTOR_VERL_NATIVE_NPU_KERNELS=1
    fi
    export JITTOR_VERL_HF_ASYNC_ROLLOUT=1
    if (( n_gpus > 1 )); then
        # Ray creates ranks dynamically rather than through
        # jittor.distributed.launch.  Give every attempt a fresh shared HCCL
        # rendezvous and explicitly authorize the compat layer's post-import
        # bootstrap path.
        export JITTOR_TORCH_DISTRIBUTED_AUTO_INIT=1
        # Retain the complete physical mask in every actor and let Jittor pick
        # the logical device from RANK.  Ray's normal in-process rewrite occurs
        # after Jittor imports CANN and would make every actor's logical device
        # zero, so disable only that rewrite.
        if [[ ${device} == npu ]]; then
            export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
            export JT_HCCL_ROOTINFO_FILE="${run_root}/hccl-root.bin"
            export JT_HCCL_COLLECTIVE_SYNC=${JT_HCCL_COLLECTIVE_SYNC:-full}
        else
            export JT_NCCL_ROOTINFO_FILE="${run_root}/nccl-root.bin"
        fi
    fi
    export backend_fallback=error
    export JT_BUILD_USE_MKL=0
    worker_python=${source_root}/agent/skills/verl-torch-compat/runtime/candidate_worker_python.sh
    unset cache_name
    export JITTOR_HOME=${JITTOR_WARM_HOME:-${run_root}/jittor-home}
    export XDG_CACHE_HOME=${JITTOR_WARM_XDG_CACHE:-${run_root}/xdg-cache}
    export DISABLE_MULTIPROCESSING=1
else
    # Official torch_npu workers need the same project-owned hf_async rollout
    # registration as the Jittor candidate, without loading Jittor's module
    # patcher into the oracle process.
    worker_python=${source_root}/agent/skills/verl-torch-compat/runtime/oracle_worker_python.sh
    # Nested Ray actors may replace rather than inherit the parent's
    # py_executable runtime field.  Put the oracle bootstrap on the global
    # module path as well so rollout, agent-loop and reward actors cannot
    # discover the host's Jittor-backed torchvision facade.
    oracle_bootstrap=${source_root}/agent/skills/verl-torch-compat/runtime/oracle
    export PYTHONPATH="${oracle_bootstrap}:${verl_root}:${source_root}/adapters"
    export PYTHONNOUSERSITE=1
    unset JITTOR_TORCH_SHIM JITTOR_TORCH_SKIP_EXT_BUILD JITTOR_TORCH_KEEP_HOME
    unset JITTOR_TRANSFORMERS_MATERIALIZED_INIT JITTOR_VERL_NATIVE_NPU_KERNELS
    unset JITTOR_TORCH_DISTRIBUTED_AUTO_INIT RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES
    python=${worker_python}
    export JITTOR_VERL_HF_ASYNC_ROLLOUT=1
fi

if [[ ! -s ${dataset_root}/train.parquet || ! -s ${dataset_root}/test.parquet ]]; then
    "${python}" "${source_root}/agent/skills/verl-torch-compat/scripts/build_minimal_ppo_dataset.py" \
        --output "${dataset_root}"
fi

cd "${verl_root}"
"${python}" -u "${source_root}/agent/skills/verl-torch-compat/runtime/launch_main_ppo.py" \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=false \
    data.train_files="['${dataset_root}/train.parquet']" \
    data.val_files="['${dataset_root}/test.parquet']" \
    data.train_batch_size=2 \
    data.shuffle=false \
    data.seed=42 \
    data.dataloader_num_workers=0 \
    data.max_prompt_length=64 \
    data.max_response_length=8 \
    data.filter_overlong_prompts=false \
    data.truncation=error \
    actor_rollout_ref.model.path="${model}" \
    actor_rollout_ref.rollout.dtype="${rollout_dtype}" \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    actor_rollout_ref.model.use_fused_kernels=false \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype="${train_dtype}" \
    "+actor_rollout_ref.actor.fsdp_config.mixed_precision={param_dtype:${train_dtype},reduce_dtype:${train_dtype},buffer_dtype:${train_dtype}}" \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.ref.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.model_dtype="${train_dtype}" \
    "+actor_rollout_ref.ref.fsdp_config.mixed_precision={param_dtype:${train_dtype},reduce_dtype:${train_dtype},buffer_dtype:${train_dtype}}" \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.name=hf_async \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.full_determinism=true \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_seqs=2 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
    critic.model.path="${critic_model}" \
    critic.model.use_remove_padding=false \
    critic.model.enable_gradient_checkpointing=false \
    critic.model.use_fused_kernels=false \
    +critic.model.override_config.attn_implementation=eager \
    critic.strategy=fsdp2 \
    critic.fsdp.strategy=fsdp2 \
    critic.fsdp.model_dtype="${train_dtype}" \
    "+critic.fsdp.mixed_precision={param_dtype:${train_dtype},reduce_dtype:${train_dtype},buffer_dtype:${train_dtype}}" \
    critic.fsdp.use_torch_compile=false \
    critic.fsdp.param_offload=false \
    critic.fsdp.optimizer_offload=false \
    critic.ppo_mini_batch_size=2 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.use_dynamic_bsz=false \
    critic.optim.lr=1e-5 \
    reward.num_workers=1 \
    trainer.device="${device}" \
    trainer.use_v1=false \
    trainer.v1.trainer_mode=sync \
    trainer.n_gpus_per_node="${n_gpus}" \
    trainer.nnodes=1 \
    trainer.total_epochs="${training_steps}" \
    trainer.total_training_steps="${training_steps}" \
    trainer.critic_warmup=0 \
    trainer.val_before_train=false \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.logger='["console"]' \
    trainer.project_name=verl_jittor_full_ppo \
    trainer.experiment_name="${attempt}" \
    trainer.default_local_dir="${run_root}/checkpoints" \
    trainer.resume_mode=disable \
    transfer_queue.enable=false \
    +ray_kwargs.ray_init.include_dashboard=false \
    ray_kwargs.ray_init.num_cpus=8 \
    ray_kwargs.ray_init.runtime_env.py_executable="${worker_python}"
