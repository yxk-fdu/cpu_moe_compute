#!/usr/bin/env bash
# 153普通CPU MoE：单卡BS1，CPU计算开启，DSpark和下一层预取均关闭。
# 共享 NZ 单卡配置，部署前按实际机器配置 NUMA 和线程数。
set -o pipefail
if [[ -f "$HOME/.bashrc" ]]; then source "$HOME/.bashrc"; fi
WORKSPACE=${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ASCEND_REPO=${ASCEND_REPO:-$WORKSPACE/vllm-ascend}
MODEL_PATH=${MODEL_PATH:-/data/libvirt/images/weights/DeepSeek-V4-Flash-0731}

CARDS=${NPU:-4}
PORT=${PORT:-8144}
MAX_SEQS=1
SERVED_NAME=deepseek_v4
GPU_MEM=0.90
MOE_DEBUG=false

export VLLM_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1

PROFILE_ROOT=${LOG_ROOT:-$WORKSPACE/../codex_logs/cpu_moe_basic}
RUN_STAMP=$(date +%Y%m%d-%H%M%S)
PROF_RAW_DIR="$PROFILE_ROOT/$RUN_STAMP"

mkdir -p "$PROF_RAW_DIR"
ln -sfn "$PROF_RAW_DIR" "$PROFILE_ROOT/latest"

LOG_DIR="$PROF_RAW_DIR/logs"
mkdir -p "$LOG_DIR"

export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$TMPDIR"

if ! python3 -c "import acl" >/dev/null 2>&1; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "[env] source CANN"
fi

KT_BUILD=${KT_BUILD:?Set KT_BUILD to the built shared NZ package directory}
export KT_MOE_MATCH_NPU_MXFP8=1
export KT_MOE_MXFP4_TOKEN_REUSE=0
export KT_GROUP_MIN_LEN=10

export KT_MOE_PIN_WEIGHTS=1

export KT_MOE_COMPLETION_RT_PRIORITY=0
export KT_MOE_COMPLETION_SPIN_US=3000
unset KT_MOE_RT_PRIORITY  # Continuous FIFO triggered RT throttling; pending refinement.

if [ "${CPU_MOE_PHASE_TRACE:-1}" = "1" ]; then
    export KT_MOE_CPU_TRACE="$PROF_RAW_DIR/cpu_compute_%p.jsonl"
else
    unset KT_MOE_CPU_TRACE
fi
unset KT_MOE_LIFECYCLE_TRACE
unset KT_MOE_LIFECYCLE_WORKERS
unset KT_MOE_PHASE_TIMING
unset CPU_MOE_CLOCK_CALIBRATION
export CPU_MOE_CLOCK_ANCHORS=1
unset CPU_MOE_ASYNC_RESULT CPU_MOE_ASYNC_VALIDATE_OUTPUTS

unset KT_MOE_TASK_TRACE KT_MOE_PAGE_TRACE
unset VLLM_ASCEND_CPU_MOE_TRACE VLLM_ASCEND_PROFILE_DIAG_LOG

export PYTHONPATH="$KT_BUILD:$ASCEND_REPO/csrc/lrc_policy:$ASCEND_REPO:${PYTHONPATH:-}"

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_ENGINE_READY_TIMEOUT_S=10000
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

export ASCEND_RT_VISIBLE_DEVICES=$CARDS

export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=0

export ASCEND_CUSTOM_OPP_PATH=$ASCEND_REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH:-}

export LD_LIBRARY_PATH=$ASCEND_REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-}

export VLLM_TORCH_PROFILER_DIR=$PROF_RAW_DIR
export VLLM_SERVER_DEV_MODE=1

export VLLM_ASCEND_OFFLOAD_THRESHOLD=32

NDE_ARRAY="84"
NDE_JSON="[$NDE_ARRAY]"

COMPILATION_CONFIG=$(cat <<JSON
{
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [
        1,2,3,4,8,16,24,32,48
    ]
}
JSON
)

PROFILER_CONFIG=$(cat <<JSON
{
    "profiler": "torch",
    "torch_profiler_dir": "$PROF_RAW_DIR",
    "torch_profiler_with_stack": false,
    "ignore_frontend": true,
    "delay_iterations": 2,
    "max_iterations": 200
}
JSON
)

# Unset overrides are JSON null, so the worker sizes the CPU MoE pool.
for name in CPU_THREADS NUMA_NODE HOST_WEIGHT_NUMA_NODE DMA_CPU; do
    if [[ -n ${!name:-} && ! ${!name:-} =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "$name must be a non-negative integer" >&2
        exit 2
    fi
done
CPU_MOE_NUMA_NODES=null
if [[ -n ${NUMA_NODE:-} ]]; then
    CPU_MOE_NUMA_NODES="[$NUMA_NODE]"
fi

ADDITIONAL_CONFIG=$(cat <<JSON
{
    "enable_cpu_binding": false,
    "decode_stats_enabled": true,
    "benchmark_route_stats_enabled": false,
    "decode_stats_path": "$LOG_DIR/cpu_moe_${RUN_STAMP}_stats",
    "expert_offload_config": {
        "expert_offload": true,
        "h2d_backend": "torch",
        "prefill_delta_load": true,
        "prefill_split_gmm": true,
        "enable_multi_card": false,
        "num_device_experts": $NDE_JSON,
        "num_device_layers": 2,
        "cache_policy_enabled": true,
        "expert_prefetch_enabled": false,
        "expert_prefetch_num": 1,
        "hot_expert_preload": true,
        "hot_experts_file": "$ASCEND_REPO/vllm_ascend/expert_offload/expert_rank_gsm8k.json",
        "moe_offload_debug": $MOE_DEBUG,
        "cpu_moe": {
            "enabled": true,
            "host_weight_numa_node": ${HOST_WEIGHT_NUMA_NODE:-null},
            "native_dispatch_dma_cpu": ${DMA_CPU:-null},
            "balance_max_transfers": 32,
            "balance_npu_route_us": 2.0,
            "balance_npu_fixed_us": 200.0,
            "balance_min_gain_us": 20.0,
            "max_num_tokens": 48,
            "cpuinfer_threads": ${CPU_THREADS:-null},
            "threadpool_count": 1,
            "numa_nodes": $CPU_MOE_NUMA_NODES
        }
    }
}
JSON
)

ulimit -l unlimited
exec vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name "$SERVED_NAME" \
    --max-num-seqs "$MAX_SEQS" \
    --max-model-len 267000 \
    --max-num-batched-tokens 32000 \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_MEM" \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --aggregate-engine-logging \
    --api-server-count 1 \
    --safetensors-load-strategy prefetch \
    --kv-cache-dtype fp8 \
    --compilation-config "$COMPILATION_CONFIG" \
    --profiler-config "$PROFILER_CONFIG" \
    --shutdown-timeout 60 \
    --additional-config "$ADDITIONAL_CONFIG"
