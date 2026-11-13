# CPU MoE

本仓库分为三个部分：

| 目录 | 内容 |
| --- | --- |
| [cpu_moe/](cpu_moe/) | **CPU计算配套实现**：计算后端、Python运行时、依赖源码和GGUF工具。使用CPU计算时下载完整目录，保持内部结构。 |
| [部署指南及参考脚本/](部署指南及参考脚本/) | 仅保留[参考启动脚本](部署指南及参考脚本/参考启动脚本.sh)、[脚本设定指南](部署指南及参考脚本/脚本设定指南.md)、[权重准备指南](部署指南及参考脚本/权重准备指南.md)。 |
| [profile合成指南/](profile合成指南/) | CPU/NPU日志采集、时钟对齐、合成脚本和样例。需要分析profile时使用；[复制agent合成指令](profile合成指南/如何让agent合成profile.md)，默认只下载最终JSON到本地桌面。 |

历史 GGUF 启动脚本 [cpu_moe/普通CPU计算.sh](cpu_moe/普通CPU计算.sh) 沿用 153 的单卡 BS1 配置，关闭 DSpark 和下一层专家预取。

## 下载什么

**准备CPU计算后端，下载完整的 [`cpu_moe/`](cpu_moe/) 目录即可。** 不要只取其中的 `kt-kernel/`，它还需要同级依赖与运行时文件。另从 [部署指南及参考脚本](部署指南及参考脚本/) 取使用说明、启动脚本；日志合成工具在独立的 [profile合成指南](profile合成指南/) 中。无需为了准备后端下载其他说明或日志样例。

## 两个仓库的关系

还必须单独准备 [vllm-ascend 的 `moe_cpu_compute_v1-on-0928` 分支](https://github.com/yxk-fdu/vllm-ascend/tree/moe_cpu_compute_v1-on-0928)。**Ascend侧代码与CPU计算后端缺一不可**：前者负责CPU/NPU专家分工、LRC、搬运、预取和同步；本仓库 `cpu_moe/` 负责CPU专家计算。

### 共享 NZ 配对

| 组件 | 配套版本 |
| --- | --- |
| vLLM | **0.30.0**，实装 `0.30.0+empty`。 |
| vllm-ascend | `moe_cpu_compute_v1-on-0928` 的共享 NZ 接入提交 [`d9ddc40`](https://github.com/yxk-fdu/vllm-ascend/commit/d9ddc405db763daeba5c70083b50f9a08a8514a7)。 |
| CPU计算后端 | 用本仓库的完整 [`cpu_moe/`](cpu_moe/) 为目标架构编译 kt-kernel，确认运行包导出 `NZMOE`。共享 NZ 内核基线为 [`db0d796`](https://github.com/yxk-fdu/cpu_moe_compute/commit/db0d79674096e4a883f1e2e5687baf441acb9ab1)，当前构建脚本还包含 ARM 轮询库编译修复。 |

使用模型原有的 MXFP4 权重。Ascend 在加载时生成最终 NZ 权重和 scale，CPU 后端借用同一批 DMA 源缓冲。设置 `cpu_moe.weight_format="nz_mxfp4"`、`native_dispatch=true`、`threadpool_count=1`，并将 `weight_path` 留空。启动前设置 `KT_MOE_MATCH_NPU_MXFP8=1`。原生派发通过 CPU 后端的 C++ 接口启动。计算线程与 DMA 源内存放在同一 NUMA 节点。配置和调用流程见 [Ascend 共享 NZ 指南](https://github.com/yxk-fdu/vllm-ascend/blob/moe_cpu_compute_v1-on-0928/docs/host_hybrid_cpu_lrc_dma.md#shared-nz-cpu-weights)。

这组共享 NZ 配对支持 x86。ARM 原生派发和映射回传已在 Ascend 950DT、CANN 9.2.1 上通过 1 和 8 token 实卡小图回归。

### 历史 GGUF 配对

| 组件 | 配套版本 |
| --- | --- |
| vLLM | **0.30.0**，实装 `0.30.0+empty`，使用同一实验构建。 |
| vllm-ascend | 固定提交 [`7cf63ca`](https://github.com/yxk-fdu/vllm-ascend/commit/7cf63ca477bf6045ad5a262ff6d59b66ace463bd)。 |
| CPU计算后端 | 固定标签 [`cpu-moe-153-20261109-r2`](https://github.com/yxk-fdu/cpu_moe_compute/tree/cpu-moe-153-20261109-r2/cpu_moe)，校验清单见 [该标签的 version.json](https://github.com/yxk-fdu/cpu_moe_compute/blob/cpu-moe-153-20261109-r2/cpu_moe/version.json)。 |

历史 GGUF 配对已在 **153服务器完成整模型启动、推理和性能测试**，覆盖单卡 BS1、DSpark 关闭/7 及下一层预取关闭/开启。CPU 参考成本为 **93.16 μs / expert-token**，对应 GGUF 内核。

模型权重和编译后的运行库另行准备。更新 GitHub 时检查两边源码，有改动的一边同步更新，并记录新的配套版本。
