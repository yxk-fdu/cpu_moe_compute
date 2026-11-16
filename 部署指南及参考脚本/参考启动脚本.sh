#!/usr/bin/env bash
# vLLM 0.30.0, single NPU, CPU MoE; shared NZ expert weights.
set -euo pipefail
export WORKSPACE=${WORKSPACE:?Set WORKSPACE to the vLLM workspace}
export ASCEND_REPO=${ASCEND_REPO:-$WORKSPACE/vllm-ascend}
export KT_BUILD=${KT_BUILD:?Set KT_BUILD to the built shared NZ package directory}
export MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the existing model directory}
export NPU=${NPU:-4} PORT=${PORT:-8144} BS=${BS:-1}
export DSPARK=${DSPARK:-7} PREFETCH=${PREFETCH:-0}
export NUMA_NODE=${NUMA_NODE:-}
export HOST_WEIGHT_NUMA_NODE=${HOST_WEIGHT_NUMA_NODE:-}
export CPU_THREADS=${CPU_THREADS:-}
export DMA_CPU=${DMA_CPU:-}
export LOG_ROOT=${LOG_ROOT:?Set LOG_ROOT to the test log directory}
export RUN_DIR="$LOG_ROOT/$(date +%Y%m%d-%H%M%S)_bs${BS}_d${DSPARK}_pre${PREFETCH}"
mkdir -p "$RUN_DIR"
if ! python -c 'import acl' >/dev/null 2>&1; then
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi
export ASCEND_RT_VISIBLE_DEVICES="$NPU"
export PYTHONPATH="$KT_BUILD:$ASCEND_REPO/csrc/lrc_policy:$ASCEND_REPO:${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$ASCEND_REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="$ASCEND_REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-}"
export KT_MOE_MATCH_NPU_MXFP8=1 KT_MOE_MXFP4_TOKEN_REUSE=0 KT_GROUP_MIN_LEN=10
export KT_MOE_PIN_WEIGHTS=1
export KT_MOE_COMPLETION_RT_PRIORITY=0 KT_MOE_COMPLETION_SPIN_US=3000
export CPU_MOE_CLOCK_ANCHORS=1
export VLLM_LOGGING_LEVEL=INFO PYTHONUNBUFFERED=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_SERVER_DEV_MODE=1
export VLLM_ASCEND_OFFLOAD_THRESHOLD=32
export ASCEND_CONNECT_TIMEOUT=10000 ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_ENGINE_READY_TIMEOUT_S=10000 VLLM_RPC_TIMEOUT=3600000 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
unset KT_MOE_RT_PRIORITY KT_MOE_LIFECYCLE_TRACE KT_MOE_LIFECYCLE_WORKERS
unset KT_MOE_PHASE_TIMING CPU_MOE_CLOCK_CALIBRATION CPU_MOE_ASYNC_RESULT CPU_MOE_ASYNC_VALIDATE_OUTPUTS
unset KT_MOE_TASK_TRACE KT_MOE_PAGE_TRACE VLLM_ASCEND_CPU_MOE_TRACE VLLM_ASCEND_PROFILE_DIAG_LOG
if [[ ${CPU_MOE_PROFILE:-1} == 1 ]]; then
    export KT_MOE_CPU_TRACE="$RUN_DIR/cpu_compute_%p.jsonl"
else
    unset KT_MOE_CPU_TRACE
fi
python - <<'PYCODE'
import importlib.metadata as metadata
import os
from pathlib import Path
from packaging.version import Version
version=metadata.version('vllm')
if Version(version).base_version != '0.30.0':
    raise SystemExit(f'This experiment is pinned to vLLM 0.30.0; installed: {version}')
for name in ('BS','CPU_THREADS'):
    if os.environ[name] and int(os.environ[name]) < 1: raise SystemExit(f'{name} must be positive')
if os.environ['PREFETCH'] not in ('0','1'): raise SystemExit('PREFETCH must be 0 or 1')
if os.environ['DSPARK'] not in ('0','7'): raise SystemExit('This launcher supports measured DSPARK=0 or 7')
if int(os.environ['BS'])*(int(os.environ['DSPARK'])+1)>48:
    raise SystemExit('Requested batch exceeds the configured CPU MoE token capacity 48')
from kt_kernel import kt_kernel_ext
if not hasattr(kt_kernel_ext.moe, 'NZMOE'):
    raise SystemExit('kt-kernel lacks NZMOE; rebuild the shared NZ package')
for name in ('native_dispatch_api','native_cpu_handle','available_worker_pool_cpus'):
    if not hasattr(kt_kernel_ext,name): raise SystemExit(f'kt-kernel lacks {name}; rebuild the matching native interface')
if os.environ.get('CPU_MOE_PROFILE', '1') == '1':
    for name in ('set_lifecycle_enabled','lifecycle_enabled'):
        if not hasattr(kt_kernel_ext,name): raise SystemExit(f'kt-kernel lacks profiling export {name}; select the matching profile package')
import vllm_ascend
if Path(vllm_ascend.__file__).resolve().parent != Path(os.environ['ASCEND_REPO']).resolve()/'vllm_ascend':
    raise SystemExit(f'Loaded unexpected Ascend source: {vllm_ascend.__file__}')
for relative in ('vllm_ascend/vllm_ascend_C',):
    if not list(Path(os.environ['ASCEND_REPO']).glob(relative+'*.so')):
        raise SystemExit('Ascend native extension missing; install/build the matching Ascend checkout first')
combine = Path(os.environ['ASCEND_REPO'])/'vllm_ascend/expert_offload/cpu_backend/lib/libcpu_moe_mapped_combine_dispatch.so'
if not combine.is_file():
    raise SystemExit('Missing mapped combine library; run csrc/cpu_moe_mapped_combine/build.sh first')
import _vllm_ascend_lrc
if os.environ['PREFETCH'] == '1':
    import _vllm_ascend_prefetch_wait
print(f'[experiment] vLLM={version}, NPU={os.environ["NPU"]}, BS={os.environ["BS"]}, DSpark={os.environ["DSPARK"]}, prefetch={os.environ["PREFETCH"]}')
PYCODE
ADDITIONAL_CONFIG=$(python - <<'PYCODE'
import json,os
def number(key):
    value = os.environ[key]
    return int(value) if value else None

numa_node = number('NUMA_NODE')
cpu=dict(enabled=True,native_dispatch_dma_cpu=number('DMA_CPU'),
         host_weight_numa_node=number('HOST_WEIGHT_NUMA_NODE'),
         balance_max_transfers=32,balance_npu_route_us=2.0,
         balance_npu_fixed_us=200.0,balance_min_gain_us=20.0,
         max_num_tokens=48,
         cpuinfer_threads=number('CPU_THREADS'),threadpool_count=1,numa_nodes=[numa_node] if numa_node is not None else None)
offload=dict(expert_offload=True,h2d_backend='torch',prefill_delta_load=True,prefill_split_gmm=True,
             enable_multi_card=False,num_device_experts=[84],num_device_layers=2,cache_policy_enabled=True,
             expert_prefetch_enabled=os.environ['PREFETCH']=='1',expert_prefetch_num=1,hot_expert_preload=True,
             hot_experts_file=os.environ['ASCEND_REPO']+'/vllm_ascend/expert_offload/expert_rank_gsm8k.json',
             moe_offload_debug=False,cpu_moe=cpu)
print(json.dumps(dict(enable_cpu_binding=False,decode_stats_enabled=True,benchmark_route_stats_enabled=False,
                     decode_stats_path=os.environ['RUN_DIR']+'/decode_stats',expert_offload_config=offload)))
PYCODE
)
COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,8,16,24,32,48]}'
PROFILER_CONFIG=$(python - <<'PYCODE'
import json,os
print(json.dumps(dict(profiler='torch',torch_profiler_dir=os.environ['RUN_DIR'],torch_profiler_with_stack=False,
                     ignore_frontend=True,delay_iterations=2,max_iterations=200)))
PYCODE
)
export ADDITIONAL_CONFIG COMPILATION_CONFIG PROFILER_CONFIG
python - <<'PYCODE'
import json, os
from pathlib import Path
run = Path(os.environ['RUN_DIR'])
(run/'additional_config.json').write_text(os.environ['ADDITIONAL_CONFIG']+'\n')
(run/'launch-config.json').write_text(json.dumps({
    'selection': {k: os.environ[k] for k in ('NPU','PORT','BS','DSPARK','PREFETCH','NUMA_NODE','CPU_THREADS','HOST_WEIGHT_NUMA_NODE','DMA_CPU')},
    'affinity_overrides': {k: os.environ.get(k) for k in ('KT_MOE_POLL_CPU', 'KT_MOE_COMPLETION_CPU')},
    'compilation': json.loads(os.environ['COMPILATION_CONFIG']),
    'profiler': json.loads(os.environ['PROFILER_CONFIG'])}, indent=2)+'\n')
PYCODE
if [[ ${CPU_MOE_CHECK_ONLY:-0} == 1 ]]; then
    echo "Environment imports and configuration checked; model not loaded. Config: $RUN_DIR"
    exit 0
fi
exec > >(tee -a "$RUN_DIR/server.log") 2>&1
speculative_args=()
if [[ $DSPARK == 7 ]]; then
    speculative_args=(--speculative-config '{"method":"dspark","num_speculative_tokens":7}')
fi
ulimit -l unlimited
exec vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$PORT" \
    --data-parallel-size 1 --tensor-parallel-size 1 --pipeline-parallel-size 1 --enable-expert-parallel \
    --seed 1024 --served-model-name deepseek_v4 --max-num-seqs "$BS" \
    --max-model-len 267000 --max-num-batched-tokens 32000 --trust-remote-code \
    --gpu-memory-utilization 0.90 --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice --reasoning-parser deepseek_v4 --aggregate-engine-logging \
    --api-server-count 1 --safetensors-load-strategy prefetch --kv-cache-dtype fp8 \
    "${speculative_args[@]}" --compilation-config "$COMPILATION_CONFIG" \
    --profiler-config "$PROFILER_CONFIG" --shutdown-timeout 60 --additional-config "$ADDITIONAL_CONFIG"
