# profile合成指南

- 合成入口：[脚本/merge_cpu_compute_latest.sh](脚本/merge_cpu_compute_latest.sh)
- 合成实现：[脚本/merge_cpu_compute_trace.py](脚本/merge_cpu_compute_trace.py)
- [参考profile：scale搬运修复版](参考profile/README.md)
- [如何让agent合成profile：离线指令，整段复制即可](如何让agent合成profile.md)

配套运行代码见 [Ascend CPU分支](https://github.com/yxk-fdu/vllm-ascend/tree/moe_cpu_compute_v1-on-0928)，当前启动与短采集步骤见 [脚本设定指南](../部署指南及参考脚本/脚本设定指南.md#验证与短-profile)。本仓库既保留 `cpu_moe/` 内的合成脚本，也在本目录的 `脚本/` 提供显式配对、校时和裁剪工具。

153已验证配置使用原生派发与按需开启的CPU事件。Ascend固定提交已集成校时钩子，**不需要再重复应用下面的历史校时补丁**。后文的补丁接入方法供缺少钩子的旧代码使用。合成工具需与输入事件格式配套；本次原生生命周期事件和DMA阶段日志的专项分析脚本仍在实验任务目录，当前合成器不能据此宣称已覆盖全部原生事件。


目标是呈现下图这种 CPU 计算条形与 NPU 原始设备流并排阅读的效果，参考 `L4_E1_C1`，新增真实传输计数后为 `L4_E1_C1_T1`。截图用于说明展示方式；合成代码和图中重叠现象不构成模型正确性、计算精度或加速收益的证明。

可下载查看的参考文件：[scale搬运修复版合成profile](参考profile/README.md)。大文件以gzip保存，解压后为原始合成JSON。

原始文件备份在 `cpu_moe/`，原始参考入口相对于该目录为 `merge_cpu_compute_latest.sh`；`cpu_moe` 的绝对路径按实际环境确定。本目录的 `脚本/`保存加入校时和 T 标签的参考版本，原始包装器保存在 `profile合成指南/参考实现/merge_cpu_compute_latest.original.sh`。新包装器要求显式给出同次 trace、CPU 日志和校时锚点，避免 latest 指向其他运行。

## 日志合成直接交给agent

打开 [如何让agent合成profile](如何让agent合成profile.md)，填写服务器和本次采集目录，把整段离线指令发给agent即可。文本已包含字段约定、校时和合成规则，不要求agent访问GitHub；没有现成脚本时可用Python标准库实现。

**默认下载到本地当前用户桌面的本次任务子目录，只下载最终合成trace JSON**：例如 `trace_view_cpu_compute_prefetch.json`。其他统计JSON、报告、CPU日志、数据库、原始trace和整个profiler目录均不下载；需要的中间结果保留在服务器供校验。

下面是合成实现与采集配置的补充说明。

## 合成日志必须满足的最低配置

要得到本仓库合成器可接受的结果，**CPU 计算 JSONL、同次 NPU profiler trace、同 worker PID 的校时锚点三者缺一不可**。只设置 `CPU_MOE_PHASE_TRACE=1` 不够；`CPU_MOE_CLOCK_CALIBRATION=1` 也不能代替新校时锚点。

### 先接入校时补丁

先检查实际运行的 `vllm-ascend` 是否已接入 `CPU_MOE_CLOCK_ANCHORS` 钩子；未接入时，在隔离源码副本中应用 `profile合成指南/脚本/clock_anchor_profiler.patch`，再把 `profile合成指南/脚本/clock_anchor_helper.py` 复制到实际运行的 `vllm_ascend/expert_offload/cpu_backend/`。应用、启动和停止的完整方法见 [校时实现说明](参考实现/CLOCK_ALIGNMENT_IMPLEMENTATION.md)。**未接入补丁时，只 export `CPU_MOE_CLOCK_ANCHORS=1` 不会自动产生所需锚点。**

### 启动脚本中必须保留这些设置

下面是应写入启动脚本的配置片段。用任务独立目录替换原脚本的 profiler 目录设置；环境变量段放在 `PROF_RAW_DIR` 确定之后、`vllm serve` 之前，避免后面的旧配置覆盖它。

```bash
PROFILE_ROOT=/实际日志根目录/cpu_moe_profile
RUN_STAMP=$(date +%Y%m%d-%H%M%S)
PROF_RAW_DIR="$PROFILE_ROOT/$RUN_STAMP"
mkdir -p "$PROF_RAW_DIR"

# 当前权重传输方案必须开启；未设为 1 的吞吐明显下降。
export KT_MOE_PIN_WEIGHTS=1

# CPU 最小计算日志：两个都显式设置，%p 按 worker PID 展开。
export CPU_MOE_PHASE_TRACE=1
export KT_MOE_CPU_TRACE="$PROF_RAW_DIR/cpu_compute_%p.jsonl"

# 必须先接入上述 profiler 钩子；记录 start/stop 配对锚点。
export CPU_MOE_CLOCK_ANCHORS=1
export VLLM_TORCH_PROFILER_DIR="$PROF_RAW_DIR"
export VLLM_SERVER_DEV_MODE=1

# 最小合成不需要内部耗时诊断。
unset KT_MOE_LIFECYCLE_TRACE KT_MOE_LIFECYCLE_WORKERS KT_MOE_PHASE_TIMING
unset KT_MOE_TASK_TRACE KT_MOE_PAGE_TRACE KT_MOE_POLL_TRACE
unset VLLM_ASCEND_CPU_MOE_TRACE VLLM_ASCEND_PROFILE_DIAG_LOG
# 旧校准记录不是本合成器输入，可关闭。
unset CPU_MOE_CLOCK_CALIBRATION
```

同时，启动脚本传给 `--profiler-config` 的 JSON 建议先改成以下内容；`$PROF_RAW_DIR` 由脚本构造 JSON 时展开：

```json
{
  "profiler": "torch",
  "torch_profiler_dir": "$PROF_RAW_DIR",
  "torch_profiler_with_stack": false,
  "ignore_frontend": true,
  "delay_iterations": 0,
  "max_iterations": 0
}
```

这里的 `max_iterations=0` 按当前实现表示由 `/start_profile` 与 `/stop_profile` 手动控制。先 warmup，再 start，运行一次有限请求，等待全部响应结束后 stop；随后等待 Ascend 解析完成。必须采集实际计算 worker 的 CPU/NPU activities，而不是只记录 API 前端。`additional_config.enable_cpu_binding=false`、`cpu_moe.max_num_tokens=32`、`expert_prefetch_enabled=false` 分别用于避免重复绑核、沿用当前后端批量设置及建立无预取基线。

### 合成前确认三个输入都存在

- 本次 `<ascend_pt>/ASCEND_PROFILER_OUTPUT/trace_view.json`。
- 本次 worker 的 `cpu_compute_<PID>.jsonl`，包含 `cpu_moe.compute` 记录。
- 同 PID 的 `cpu_clock_anchors_<PID>.jsonl`，且两个唯一 start/stop host marker 都在本次 trace 中。

```bash
bash profile合成指南/脚本/merge_cpu_compute_latest.sh \
  "$TRACE" "$CPU_LOG" "$CLOCK_ANCHORS" "$OUT"
```

这是默认 part compute 的最低流程。现有 C++ 区间到每个 part 的 `forward_impl()` 返回为止，不包含上层 `merge_results()` 的归并和 dtype 转换，也不含 completion 通知、Python 返回和 H2D。若要统计 CPU 计算起点到最终 CPU 输出写出的整个区间，还必须应用本指南末尾的 total 计时补丁、重编后端，并显式选择 total 事件；不能只改环境变量就假定旧二进制已有该能力。

## 必须先对齐时钟

**CPU 与 NPU 可能使用不同的时钟域、起点和频率。直接将 CPU wall_ns 塞到 NPU trace 会造成错位，甚至制造虚假的并行与等待。** 原启动脚本开启旧校准变量，但原合成器没有应用时钟对齐；旧校准日志只有 CPU 包围区间，不能据此唯一还原设备时间。

新参考方案在同一 worker 中生成两类配对数据：profiler 的唯一 host marker，以及 CPU `CLOCK_MONOTONIC_RAW` 在 marker 入口前后的区间。通过首末锚点将 CPU RAW 映射到 profiler host 时域，并报告采样不确定性和漂移。该方案还依赖 Ascend exporter 已将 NPU device 时间映射到同一个 host 时域；如果原始 trace 的 host/device 自身未对齐，必须进一步取得它们的同步元数据，不能把 host marker 当成设备同步的替代。

接入方法见 [校时实现说明](参考实现/CLOCK_ALIGNMENT_IMPLEMENTATION.md)。每个 worker PID 在真正开始采集后、最后一批计算完成且停止采集前分别调用 `clock_anchor_helper.record_clock_anchor('start', CAPTURE)` 和 `record_clock_anchor('stop', CAPTURE)`。`delay_iterations=2` 的原配置需要避免锚点落在未采集阶段；不能在开始前或 profiler 自动停止后再补标记。该 helper 的 import 必须发生在实际 worker 可访问的位置，而不是只在 API 前端调用。

合成器缺少配对锚点、锚点缺失或重复、时钟漂移异常或没有范围内 CPU 记录时会报错；常驻日志中锚点范围外的记录会排除并计数。不要移动 CPU 条形去适配 `NOTIFY_WAIT`，也不要把 wall clock 差值当成可靠的设备偏移。

## 同次输入与合成命令

等待 Ascend 解析成功并确认输出停止变化，选择：

1. 本次 `<ascend_pt>/ASCEND_PROFILER_OUTPUT/trace_view.json`。
2. 同 PID、同次运行的 `cpu_compute_<pid>.jsonl`。
3. 同次 worker 产生的 `cpu_clock_anchors_<pid>.jsonl`。可审查的生命周期接入补丁位于 `profile合成指南/脚本/clock_anchor_profiler.patch`，启用 `CPU_MOE_CLOCK_ANCHORS=1`，详细应用步骤见实施说明。

```bash
bash profile合成指南/脚本/merge_cpu_compute_latest.sh \
  "$TRACE" "$CPU_LOG" "$CLOCK_ANCHORS" "$OUT"
```

输出 `trace_view_cpu_compute.json`、`cpu_compute_stats.json`、`cpu_compute_report.txt`。也可以直接运行核心合成器，参数见 `--help`；多个 CPU 日志需要在校时 JSONL 中包含各 PID 的锚点。原始输入不要覆盖。使用 MindStudio 打开合成结果。

## 标签含义

| 字段 | 含义 |
|---|---|
| L4 | 显示层号，按当前合成器从零基 layer 加一显示。 |
| E1 | CPU compute 记录中的不同 CPU 专家数 expert_count；expert_ids_truncated 时可能不完整。 |
| C1 | CPU compute 记录的专家 route 数 expert_route_count，不是 CPU 核数。 |
| T1 | 该层本次按需准入迁入 NPU 的一个专家权重集合。T2 为两个。 |
| T0 | 日志明确记录本次按需迁入数为零。 |
| Tunknown | 旧日志没有 transfer_count，不能猜成零。 |

T 由 NPU 路由的 active.sum() 经 sentinel 传到 CPU 日志，重复 sentinel 取最大值而非求和。它不是字节数或 memcpy 次数，不包含独立预取，也不证明每次传输已经完成。查看独立预取需要对应日志另行说明。

## 合成后检查

检查 `actual_compute_spans`、`cpu_spans`、`unmatched_compute_spans` 和 `writer_status`，dropped 非零、无 CPU 记录或未匹配均应明确报告。检查校时斜率、锚点包围范围、不确定性及原始 NPU trace 的时钟同步前提，误差较大时不作微秒级结论。

导出只保留本次 CPU 计算窗口，窗口内原始设备、Python、CANN、draft、Overlap Analysis 流及事件保留；CPU actual compute 放在设备流附近上方。时间戳用 Decimal 处理，窗口内部事件保持原始 ts/dur，只有跨界事件精确裁剪，避免大绝对时间戳 float 舍入制造假重叠。裁剪窗口取本次已校时 CPU spans 与唯一匹配 roundtrip 的连续最早起点和最晚终点并集，保留匹配往返完整边界。

当前修改通过合成 fixture 验证 T 标签、时钟偏移映射、缺锚点拒绝与精确裁剪，尚未完成新采集的真机端到端验证。截图是目标效果，使用前仍须完成 worker 接入、真实采集和原始设备时钟映射核对。

## 覆盖完整 CPU 结果归并的可选补丁

现有最小开关只记录 part compute。若要求从外层 CPU 计算开始到 CPU 结果归并及类型转换完成，提供 `profile合成指南/脚本/cpu_compute_total.patch` 供隔离环境审查与编译验证。该补丁增加 `KT_MOE_CPU_TRACE_KIND=total`，写 `cpu_moe.compute_total`，并关闭原 part span，避免重复计时。边界到外层 TP_MOE forward 内 merge_results 完成为止，仍不包含 Python 返回、completion 通知和设备侧结果使用。

应用并重新编译后启用：

```bash
export KT_MOE_CPU_TRACE_KIND=total
export KT_MOE_CPU_TRACE="$CAPTURE/cpu_compute_%p.jsonl"
# 采集完成并校时后，包装器显式选择新事件：
CPU_MOE_COMPUTE_EVENT=cpu_moe.compute_total \
  bash profile合成指南/脚本/merge_cpu_compute_latest.sh "$TRACE" "$CPU_LOG" "$CLOCK_ANCHORS" "$OUT"
```

补丁尚未编译或真机验证，不能直接当作当前服务器已有能力。原有 `KT_DECODE_TIMING=1` / `KT_PREFILL_TIMING=1` 统计 submit 到 sync 的聚合时间，会混入排队，不等于这里的最小计算区间；`FORWARD_TIME_REPORT` 是编译期详细打印，也不作为本次最小采集开关。
