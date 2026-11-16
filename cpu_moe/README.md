# CPU 计算后端

> **草稿版，未经 Ascend 实卡验证。** 本版用于代码审阅和后续实卡验证。

本目录保存 CPU MoE 后端及配套源码。与 [vllm-ascend 的 CPU分支](https://github.com/yxk-fdu/vllm-ascend/tree/moe_cpu_compute_v1-on-0928) 配套：Ascend 侧负责专家分工、信号、搬运和同步；这里的 kt-kernel / CPUInfer 负责实际 CPU 专家计算。

本草稿配套的 Ascend 提交为 [`b6be7b5`](https://github.com/yxk-fdu/vllm-ascend/commit/b6be7b5b56ffef1e1c53e6dcdf229de929d98468)。

| 路径 | 内容 |
| --- | --- |
| [dependencies/ktransformers-AK/kt-kernel/](dependencies/ktransformers-AK/kt-kernel) | C++ CPU专家算子、CPUInfer线程池、共享 NZ 内核及构建入口。 |
| [kt_async_runtime/kt_kernel/](kt_async_runtime/kt_kernel) | Python运行时、后端适配。 |
| [dependencies/ktransformers-AK/third_party/](dependencies/ktransformers-AK/third_party) | 后端构建所用第三方源码和许可证。 |
| `merge_cpu_compute_latest.sh` / `merge_cpu_compute_trace.py` | CPU/NPU日志合成参考实现。当前使用入口见独立的profile合成指南。 |

## 共享 NZ 执行路径

CPU MoE 启用后，Ascend 将最终 NZ 权重和 scale 的 Tensor 所有者交给本目录的 `NZMoEWrapper`。C++ `NZMOE` 和 Ascend 原生 DMA 使用同一批地址。CPU 后端按专家分组并复用重复选中专家的解包结果。此路径使用模型原有权重，按 [共享 NZ 配对说明](../README.md#共享-nz-配对) 编译和配置。

## 启动

[普通CPU计算.sh](普通CPU计算.sh) 默认使用单卡 BS1，关闭 DSpark 和下一层专家预取。构建后设置 `KT_BUILD` 指向共享 NZ 的 Python 包目录，`ASCEND_REPO` 指向配套源码，`MODEL_PATH` 指向原始 MXFP4 模型目录。安装入口和可调参数见 [脚本设定指南](../部署指南及参考脚本/脚本设定指南.md)。
